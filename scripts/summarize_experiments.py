import argparse
import csv
import json
import re
from pathlib import Path


FIELDNAMES = [
    'group',
    'path',
    'seed_id',
    'is_anomalous',
    'anomaly_reason',
    'unlearning_method',
    'round2_global_loss',
    'final_global_loss',
    'global_loss_before_unlearning',
    'global_loss_after_unlearning',
    'forget_client_loss_before_unlearning',
    'forget_client_loss_after_unlearning',
    'mia_loss_auc',
    'mia_loss_tpr_at_fpr001',
    'mia_member_count',
    'mia_nonmember_count',
    'unlearning_time_sec',
    'hessian_mode',
    'retained_reference_set_size',
    'last_hvp_l2_norm',
    'actual_param_delta_l2_norm',
    'unlearn_eta',
    'unlearn_max_update_norm',
    'unlearn_num_steps',
    'unlearn_steps_attempted',
    'unlearn_steps_accepted',
    'unlearn_guard_stop_reason',
    'unlearn_ref_loss_guard_ratio',
    'unlearn_global_loss_guard_max',
    'unlearn_update_sign_mode',
    'unlearn_selected_update_sign',
    'unlearn_direction_check_enabled',
    'unlearn_best_probe_update_sign',
    'unlearn_best_probe_direction_forget_delta',
    'unlearn_best_probe_direction_global_loss_after',
    'unlearn_forget_loss_guard_enabled',
    'unlearn_forget_guard_sample_size',
    'unlearn_forget_loss_min_gain',
    'unlearn_forget_loss_tolerance',
    'unlearn_forget_loss_before_guard',
    'unlearn_forget_loss_limit',
    'unlearn_forget_loss_after_last_accepted',
    'unlearn_reference_loss_before_guard',
    'unlearn_reference_loss_after_last_accepted',
    'unlearn_global_loss_after_last_accepted',
    'unlearn_applied_update_l2_norm_estimate',
    'update_clipping_enabled',
    'update_clip_applied',
    'update_clip_coefficient',
    'scaled_update_l2_norm_before_clipping',
    'scaled_update_l2_norm_after_clipping',
]


def _get(mapping, *keys, default=''):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _seed_id_from_group(group):
    if not group:
        return ''
    first_part = str(group).split('/', 1)[0]
    return first_part if re.fullmatch(r'seed\d+', first_part) else ''


def _group_name(group):
    return str(group).rsplit('/', 1)[-1]


def _to_float(value):
    try:
        if value in ('', None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    try:
        if value in ('', None):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _infer_group(root, result_path):
    rel_parts = result_path.relative_to(root).parts
    if 'exp' in rel_parts:
        exp_idx = rel_parts.index('exp')
        if exp_idx > 0:
            return '/'.join(rel_parts[:exp_idx])
    if len(rel_parts) > 1:
        return rel_parts[0]
    return result_path.parent.name


def _should_skip_result_path(root, result_path):
    rel_parts = result_path.relative_to(root).parts
    return any(part.startswith('appendix_') for part in rel_parts)


def _summarize_result(root, result_path):
    with result_path.open('r', encoding='utf-8') as file:
        result = json.load(file)

    eval_history = result.get('eval_history') or []
    metrics = result.get('experiment_metrics') or {}
    config = result.get('config') or {}

    return {
        'group': _infer_group(root, result_path),
        'path': str(result_path.relative_to(root)),
        'seed_id': '',
        'is_anomalous': False,
        'anomaly_reason': '',
        'unlearning_method': _get(metrics, 'unlearning_method', default=config.get('unlearn_method', '')),
        'round2_global_loss': eval_history[1] if len(eval_history) > 1 else '',
        'final_global_loss': eval_history[-1] if eval_history else '',
        'global_loss_before_unlearning': _get(metrics, 'global_loss_before_unlearning'),
        'global_loss_after_unlearning': _get(metrics, 'global_loss_after_unlearning'),
        'forget_client_loss_before_unlearning': _get(metrics, 'forget_client_loss_before_unlearning'),
        'forget_client_loss_after_unlearning': _get(metrics, 'forget_client_loss_after_unlearning'),
        'mia_loss_auc': _get(metrics, 'mia_loss_auc'),
        'mia_loss_tpr_at_fpr001': _get(metrics, 'mia_loss_tpr_at_fpr001'),
        'mia_member_count': _get(metrics, 'mia_member_count'),
        'mia_nonmember_count': _get(metrics, 'mia_nonmember_count'),
        'unlearning_time_sec': _get(metrics, 'unlearning_time_sec'),
        'hessian_mode': _get(
            metrics,
            'effective_hessian_reference_mode',
            'hessian_reference_mode',
            'requested_hessian_reference_mode',
        ),
        'retained_reference_set_size': _get(metrics, 'retained_reference_set_size'),
        'last_hvp_l2_norm': _get(metrics, 'last_hvp_l2_norm'),
        'actual_param_delta_l2_norm': _get(metrics, 'actual_param_delta_l2_norm'),
        'unlearn_eta': _get(metrics, 'unlearn_eta', default=config.get('unlearn_eta', '')),
        'unlearn_max_update_norm': _get(
            metrics,
            'unlearn_max_update_norm',
            default=config.get('unlearn_max_update_norm', ''),
        ),
        'unlearn_num_steps': _get(metrics, 'unlearn_num_steps', default=config.get('unlearn_num_steps', '')),
        'unlearn_steps_attempted': _get(metrics, 'unlearn_steps_attempted'),
        'unlearn_steps_accepted': _get(metrics, 'unlearn_steps_accepted'),
        'unlearn_guard_stop_reason': _get(metrics, 'unlearn_guard_stop_reason'),
        'unlearn_ref_loss_guard_ratio': _get(
            metrics,
            'unlearn_ref_loss_guard_ratio',
            default=config.get('unlearn_ref_loss_guard_ratio', ''),
        ),
        'unlearn_global_loss_guard_max': _get(
            metrics,
            'unlearn_global_loss_guard_max',
            default=config.get('unlearn_global_loss_guard_max', ''),
        ),
        'unlearn_update_sign_mode': _get(
            metrics,
            'unlearn_update_sign_mode',
            default=config.get('unlearn_update_sign', ''),
        ),
        'unlearn_selected_update_sign': _get(metrics, 'unlearn_selected_update_sign'),
        'unlearn_direction_check_enabled': _get(metrics, 'unlearn_direction_check_enabled'),
        'unlearn_best_probe_update_sign': _get(metrics, 'unlearn_best_probe_update_sign'),
        'unlearn_best_probe_direction_forget_delta': _get(metrics, 'unlearn_best_probe_direction_forget_delta'),
        'unlearn_best_probe_direction_global_loss_after': _get(
            metrics,
            'unlearn_best_probe_direction_global_loss_after',
        ),
        'unlearn_forget_loss_guard_enabled': _get(metrics, 'unlearn_forget_loss_guard_enabled'),
        'unlearn_forget_guard_sample_size': _get(metrics, 'unlearn_forget_guard_sample_size'),
        'unlearn_forget_loss_min_gain': _get(
            metrics,
            'unlearn_forget_loss_min_gain',
            default=config.get('unlearn_forget_loss_min_gain', ''),
        ),
        'unlearn_forget_loss_tolerance': _get(
            metrics,
            'unlearn_forget_loss_tolerance',
            default=config.get('unlearn_forget_loss_tolerance', ''),
        ),
        'unlearn_forget_loss_before_guard': _get(metrics, 'unlearn_forget_loss_before_guard'),
        'unlearn_forget_loss_limit': _get(metrics, 'unlearn_forget_loss_limit'),
        'unlearn_forget_loss_after_last_accepted': _get(metrics, 'unlearn_forget_loss_after_last_accepted'),
        'unlearn_reference_loss_before_guard': _get(metrics, 'unlearn_reference_loss_before_guard'),
        'unlearn_reference_loss_after_last_accepted': _get(
            metrics,
            'unlearn_reference_loss_after_last_accepted',
        ),
        'unlearn_global_loss_after_last_accepted': _get(metrics, 'unlearn_global_loss_after_last_accepted'),
        'unlearn_applied_update_l2_norm_estimate': _get(metrics, 'unlearn_applied_update_l2_norm_estimate'),
        'update_clipping_enabled': _get(metrics, 'update_clipping_enabled'),
        'update_clip_applied': _get(metrics, 'update_clip_applied'),
        'update_clip_coefficient': _get(metrics, 'update_clip_coefficient'),
        'scaled_update_l2_norm_before_clipping': _get(
            metrics,
            'scaled_update_l2_norm_before_clipping',
            'scaled_update_l2_norm',
        ),
        'scaled_update_l2_norm_after_clipping': _get(
            metrics,
            'scaled_update_l2_norm_after_clipping',
            'scaled_update_l2_norm',
        ),
    }


def _annotate_anomalous_rows(rows):
    annotated_rows = [dict(row) for row in rows]
    grouped = {}
    for row in annotated_rows:
        seed_id = _seed_id_from_group(row.get('group', ''))
        row['seed_id'] = seed_id
        grouped.setdefault(seed_id, []).append(row)

    baseline_groups = {'fl', 'fedhds', 'retrain_oracle'}
    for seed_id, seed_rows in grouped.items():
        if not seed_id:
            for row in seed_rows:
                row['is_anomalous'] = False
                row['anomaly_reason'] = ''
            continue

        reasons = []
        baseline_rows = [row for row in seed_rows if _group_name(row.get('group', '')) in baseline_groups]
        if baseline_rows:
            baseline_losses = []
            for row in baseline_rows:
                for key in ('round2_global_loss', 'final_global_loss'):
                    value = _to_float(row.get(key))
                    if value is not None:
                        baseline_losses.append(value)
            if any(loss >= 5.0 for loss in baseline_losses):
                reasons.append('baseline_diverged')

            member_counts = [_to_int(row.get('mia_member_count')) for row in baseline_rows]
            member_counts = [count for count in member_counts if count is not None]
            if member_counts and min(member_counts) < 150:
                reasons.append('forget_client_too_small')

        reason_text = ','.join(reasons)
        for row in seed_rows:
            row['is_anomalous'] = bool(reasons)
            row['anomaly_reason'] = reason_text

    return annotated_rows


def main():
    parser = argparse.ArgumentParser(description='Summarize final_results.json files into one CSV.')
    parser.add_argument('root', help='Experiment root directory to scan recursively.')
    parser.add_argument('--output', default='', help='CSV path. Defaults to <root>/summary.csv.')
    parser.add_argument(
        '--exclude-anomalous-seeds',
        action='store_true',
        help='Omit rows whose seed is flagged as anomalous by the baseline-health heuristic.',
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f'Experiment root does not exist: {root}')

    output_path = Path(args.output).resolve() if args.output else root / 'summary.csv'
    result_paths = sorted(
        path for path in root.rglob('final_results.json') if not _should_skip_result_path(root, path)
    )

    rows = [_summarize_result(root, result_path) for result_path in result_paths]
    rows = _annotate_anomalous_rows(rows)
    if args.exclude_anomalous_seeds:
        rows = [row for row in rows if not row.get('is_anomalous')]
    with output_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} rows to {output_path}')


if __name__ == '__main__':
    main()
