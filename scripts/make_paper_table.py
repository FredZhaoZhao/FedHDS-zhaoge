import argparse
import csv
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    'final_global_loss',
    'global_loss_before_unlearning',
    'global_loss_after_unlearning',
    'forget_client_loss_before_unlearning',
    'forget_client_loss_after_unlearning',
    'mia_loss_auc',
    'mia_loss_tpr_at_fpr001',
    'unlearning_time_sec',
    'actual_param_delta_l2_norm',
    'unlearn_applied_update_l2_norm_estimate',
    'unlearn_eta',
    'unlearn_max_update_norm',
    'unlearn_num_steps',
    'unlearn_steps_attempted',
    'unlearn_steps_accepted',
    'unlearn_global_loss_guard_max',
    'unlearn_best_probe_direction_forget_delta',
    'unlearn_forget_loss_min_gain',
    'unlearn_forget_loss_tolerance',
    'unlearn_forget_loss_after_last_accepted',
]

OUTPUT_COLUMNS = [
    'source',
    'method',
    'setting',
    'hessian',
    'forget_loss',
    'mia_auc',
    'mia_tpr_at_fpr001',
    'final_global_loss',
    'update_l2',
    'steps',
    'unlearning_time_sec',
    'note',
]


def _to_numeric(df):
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')


def _fmt(value, digits=4, empty='-'):
    if pd.isna(value):
        return empty
    return f'{float(value):.{digits}f}'


def _get(row, key, default=''):
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return value


def _final_global_loss(row):
    for key in ('global_loss_after_unlearning', 'final_global_loss', 'round2_global_loss'):
        value = row.get(key)
        if pd.notna(value):
            return value
    return None


def _update_l2(row):
    for key in (
        'actual_param_delta_l2_norm',
        'unlearn_applied_update_l2_norm_estimate',
        'scaled_update_l2_norm_after_clipping',
    ):
        value = row.get(key)
        if pd.notna(value):
            return value
    return None


def _setting(row):
    parts = []
    eta = row.get('unlearn_eta')
    if pd.notna(eta):
        parts.append(f'eta={_fmt(eta, digits=4)}')

    max_norm = row.get('unlearn_max_update_norm')
    if pd.notna(max_norm) and float(max_norm) > 0:
        parts.append(f'max_norm={_fmt(max_norm, digits=3)}')

    steps = row.get('unlearn_num_steps')
    if pd.notna(steps) and int(float(steps)) > 1:
        parts.append(f'steps={int(float(steps))}')

    guard = row.get('unlearn_global_loss_guard_max')
    if pd.notna(guard) and float(guard) > 0:
        parts.append(f'global_guard={_fmt(guard, digits=3)}')

    sign = str(_get(row, 'unlearn_selected_update_sign', '')).strip()
    if sign and sign not in ('nan', 'positive'):
        parts.append(f'sign={sign}')

    forget_guard = str(_get(row, 'unlearn_forget_loss_guard_enabled', '')).strip().lower()
    if forget_guard == 'true':
        min_gain = row.get('unlearn_forget_loss_min_gain')
        tolerance = row.get('unlearn_forget_loss_tolerance')
        if pd.notna(min_gain) and float(min_gain) > 0:
            parts.append(f'forget_guard=+{_fmt(min_gain, digits=4)}')
        elif pd.notna(tolerance) and float(tolerance) > 0:
            parts.append(f'forget_guard=tol{_fmt(tolerance, digits=4)}')
        else:
            parts.append('forget_guard=nondecrease')

    if parts:
        return ', '.join(parts)

    hessian = str(_get(row, 'hessian_mode', '')).strip()
    if hessian in ('', 'nan'):
        return 'training only'
    return 'default'


def _forget_loss(row):
    before = row.get('forget_client_loss_before_unlearning')
    after = row.get('forget_client_loss_after_unlearning')
    if pd.notna(before) and pd.notna(after):
        return f'{_fmt(before)} -> {_fmt(after)}'
    return '-'


def _steps(row):
    accepted = row.get('unlearn_steps_accepted')
    attempted = row.get('unlearn_steps_attempted')
    if pd.isna(attempted):
        attempted = row.get('unlearn_num_steps')
    if pd.notna(accepted) and pd.notna(attempted):
        return f'{int(float(accepted))}/{int(float(attempted))}'
    return '-'


def _method(row):
    method = str(_get(row, 'unlearning_method', '')).strip()
    if method and method != 'nan':
        return method

    hessian = str(_get(row, 'hessian_mode', '')).strip()
    if hessian in ('retrain-from-scratch',):
        return 'retrain'
    if hessian in ('', 'nan'):
        return 'training_only'
    return 'second_order'


def _note(row):
    mode = str(_get(row, 'hessian_mode', '')).strip()
    final_loss = _final_global_loss(row)
    forget_before = row.get('forget_client_loss_before_unlearning')
    forget_after = row.get('forget_client_loss_after_unlearning')
    guard_reason = str(_get(row, 'unlearn_guard_stop_reason', '')).strip()
    anomaly_flag = str(_get(row, 'is_anomalous', '')).strip().lower() == 'true'
    anomaly_reason = str(_get(row, 'anomaly_reason', '')).strip()
    anomaly_note = ''
    if anomaly_flag:
        anomaly_note = 'anomalous seed'
        if anomaly_reason and anomaly_reason != 'nan':
            anomaly_note = f'{anomaly_note} ({anomaly_reason})'

    if mode in ('', 'nan'):
        return f'{anomaly_note}; utility baseline'.strip('; ') if anomaly_note else 'utility baseline'
    if guard_reason and guard_reason not in ('completed', 'nan'):
        tail = f'guard stopped by {guard_reason}'
        return f'{anomaly_note}; {tail}'.strip('; ') if anomaly_note else tail
    best_probe_delta = row.get('unlearn_best_probe_direction_forget_delta')
    best_probe_sign = str(_get(row, 'unlearn_best_probe_update_sign', '')).strip()
    if pd.notna(best_probe_delta) and best_probe_sign and best_probe_sign != 'nan':
        if best_probe_sign == 'negative':
            tail = f'best probe sign negative, forget delta {_fmt(best_probe_delta)}'
            return f'{anomaly_note}; {tail}'.strip('; ') if anomaly_note else tail
        tail = f'best probe forget delta {_fmt(best_probe_delta)}'
        return f'{anomaly_note}; {tail}'.strip('; ') if anomaly_note else tail
    if pd.notna(final_loss) and float(final_loss) >= 5:
        tail = 'unstable in this setting'
        return f'{anomaly_note}; {tail}'.strip('; ') if anomaly_note else tail
    if pd.notna(forget_before) and pd.notna(forget_after):
        gain = float(forget_after) - float(forget_before)
        if gain > 0:
            tail = 'forget loss increased'
            return f'{anomaly_note}; {tail}'.strip('; ') if anomaly_note else tail
    return anomaly_note


def build_table(summary_csv):
    df = pd.read_csv(summary_csv)
    _to_numeric(df)

    rows = []
    for _, row in df.iterrows():
        source = str(_get(row, 'group', '')).strip()
        if not source:
            source = str(_get(row, 'path', '')).strip()

        rows.append({
            'source': source,
            'method': _method(row),
            'setting': _setting(row),
            'hessian': str(_get(row, 'hessian_mode', 'none')).strip() or 'none',
            'forget_loss': _forget_loss(row),
            'mia_auc': _fmt(row.get('mia_loss_auc')),
            'mia_tpr_at_fpr001': _fmt(row.get('mia_loss_tpr_at_fpr001')),
            'final_global_loss': _fmt(_final_global_loss(row)),
            'update_l2': _fmt(_update_l2(row)),
            'steps': _steps(row),
            'unlearning_time_sec': _fmt(row.get('unlearning_time_sec'), digits=2),
            'note': _note(row),
        })

    return rows


def write_csv(rows, output_path):
    with output_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, output_path):
    with output_path.open('w', encoding='utf-8') as file:
        file.write('| Source | Method | Setting | Hessian | Forget loss | MIA AUC | MIA TPR@1%FPR | Final global loss | Update L2 | Steps | Time (s) | Note |\n')
        file.write('|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n')
        for row in rows:
            values = [
                row['source'],
                row['method'],
                row['setting'],
                row['hessian'],
                row['forget_loss'],
                row['mia_auc'],
                row['mia_tpr_at_fpr001'],
                row['final_global_loss'],
                row['update_l2'],
                row['steps'],
                row['unlearning_time_sec'],
                row['note'],
            ]
            safe_values = [str(value).replace('|', '/') for value in values]
            file.write('| ' + ' | '.join(safe_values) + ' |\n')


def main():
    parser = argparse.ArgumentParser(description='Create paper-ready CSV/Markdown tables from summary.csv.')
    parser.add_argument('summary_csv', help='Path to summary.csv generated by summarize_experiments.py.')
    parser.add_argument('--output-prefix', default='', help='Output prefix. Defaults to <summary parent>/paper_table.')
    args = parser.parse_args()

    summary_path = Path(args.summary_csv).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f'Summary CSV does not exist: {summary_path}')

    output_prefix = Path(args.output_prefix).resolve() if args.output_prefix else summary_path.parent / 'paper_table'
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = build_table(summary_path)
    csv_path = output_prefix.with_suffix('.csv')
    md_path = output_prefix.with_suffix('.md')
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(f'Wrote {len(rows)} rows to {csv_path}')
    print(f'Wrote {len(rows)} rows to {md_path}')


if __name__ == '__main__':
    main()
