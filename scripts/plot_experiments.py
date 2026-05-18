import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    'final_global_loss',
    'global_loss_before_unlearning',
    'global_loss_after_unlearning',
    'forget_client_loss_before_unlearning',
    'forget_client_loss_after_unlearning',
    'unlearning_time_sec',
    'actual_param_delta_l2_norm',
    'unlearn_applied_update_l2_norm_estimate',
    'scaled_update_l2_norm_after_clipping',
    'unlearn_steps_attempted',
    'unlearn_steps_accepted',
]


def _read_summary(path):
    df = pd.read_csv(path)
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

    if 'group' not in df.columns:
        df['group'] = [f'run_{idx + 1}' for idx in range(len(df))]

    df['plot_label'] = df['group'].fillna('').astype(str)
    if 'path' in df.columns:
        missing = df['plot_label'].str.len() == 0
        df.loc[missing, 'plot_label'] = df.loc[missing, 'path'].astype(str)

    df['plot_label'] = df['plot_label'].map(_short_label)
    return df


def _short_label(value, max_len=36):
    label = str(value).strip().replace('\\', '/')
    if not label:
        return 'run'
    if len(label) <= max_len:
        return label
    keep = max_len - 3
    return '...' + label[-keep:]


def _final_global_loss(row):
    after = row.get('global_loss_after_unlearning', np.nan)
    if pd.notna(after):
        return after
    final = row.get('final_global_loss', np.nan)
    if pd.notna(final):
        return final
    return row.get('round2_global_loss', np.nan)


def _update_l2(row):
    for column in (
        'actual_param_delta_l2_norm',
        'unlearn_applied_update_l2_norm_estimate',
        'scaled_update_l2_norm_after_clipping',
    ):
        value = row.get(column, np.nan)
        if pd.notna(value):
            return value
    return np.nan


def _save(fig, output_dir, stem, formats):
    output_paths = []
    for fmt in formats:
        path = output_dir / f'{stem}.{fmt}'
        fig.savefig(path, dpi=220, bbox_inches='tight')
        output_paths.append(path)
    plt.close(fig)
    return output_paths


def _bar_width(count):
    return max(8, min(18, 0.55 * count + 4))


def plot_forget_loss(df, output_dir, formats):
    before_col = 'forget_client_loss_before_unlearning'
    after_col = 'forget_client_loss_after_unlearning'
    if before_col not in df.columns or after_col not in df.columns:
        return []

    subset = df[df[[before_col, after_col]].notna().any(axis=1)].copy()
    if subset.empty:
        return []

    x = np.arange(len(subset))
    width = 0.36
    fig, ax = plt.subplots(figsize=(_bar_width(len(subset)), 5))
    ax.bar(x - width / 2, subset[before_col], width, label='Before unlearning', color='#4c78a8')
    ax.bar(x + width / 2, subset[after_col], width, label='After unlearning', color='#f58518')
    ax.set_ylabel('Forget-client loss')
    ax.set_title('Forget-Client Loss Before and After Unlearning')
    ax.set_xticks(x)
    ax.set_xticklabels(subset['plot_label'], rotation=35, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    return _save(fig, output_dir, 'forget_loss_before_after', formats)


def plot_global_loss(df, output_dir, formats):
    subset = df.copy()
    subset['plot_global_loss'] = subset.apply(_final_global_loss, axis=1)
    subset = subset[subset['plot_global_loss'].notna()].copy()
    if subset.empty:
        return []

    colors = [
        '#54a24b' if pd.isna(row.get('global_loss_after_unlearning', np.nan)) else '#e45756'
        for _, row in subset.iterrows()
    ]
    x = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(_bar_width(len(subset)), 5))
    ax.bar(x, subset['plot_global_loss'], color=colors)
    ax.set_ylabel('Final/global loss')
    ax.set_title('Final Global Loss by Experiment')
    ax.set_xticks(x)
    ax.set_xticklabels(subset['plot_label'], rotation=35, ha='right')
    ax.grid(axis='y', alpha=0.25)
    return _save(fig, output_dir, 'global_loss_final', formats)


def plot_update_tradeoff(df, output_dir, formats):
    subset = df.copy()
    subset['plot_global_loss'] = subset.apply(_final_global_loss, axis=1)
    subset['plot_update_l2'] = subset.apply(_update_l2, axis=1)
    if {
        'forget_client_loss_before_unlearning',
        'forget_client_loss_after_unlearning',
    }.issubset(subset.columns):
        subset['forget_gain'] = (
            subset['forget_client_loss_after_unlearning']
            - subset['forget_client_loss_before_unlearning']
        )
    else:
        subset['forget_gain'] = np.nan

    subset = subset[
        subset['plot_update_l2'].notna()
        & subset['plot_global_loss'].notna()
    ].copy()
    if subset.empty:
        return []

    outputs = []
    mode_col = 'hessian_mode' if 'hessian_mode' in subset.columns else None

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if mode_col:
        for mode, group in subset.groupby(mode_col, dropna=False):
            ax.scatter(
                group['plot_update_l2'],
                group['plot_global_loss'],
                s=60,
                label=str(mode) if str(mode) else 'unknown',
                alpha=0.85,
            )
    else:
        ax.scatter(subset['plot_update_l2'], subset['plot_global_loss'], s=60, alpha=0.85)
    for _, row in subset.iterrows():
        ax.annotate(row['plot_label'], (row['plot_update_l2'], row['plot_global_loss']), fontsize=7)
    ax.set_xlabel('Actual update L2 norm')
    ax.set_ylabel('Final global loss')
    ax.set_title('Update Norm vs Global Loss')
    ax.grid(alpha=0.25)
    if mode_col:
        ax.legend()
    outputs.extend(_save(fig, output_dir, 'update_norm_vs_global_loss', formats))

    subset_gain = subset[subset['forget_gain'].notna()].copy()
    if not subset_gain.empty:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        if mode_col:
            for mode, group in subset_gain.groupby(mode_col, dropna=False):
                ax.scatter(
                    group['plot_update_l2'],
                    group['forget_gain'],
                    s=60,
                    label=str(mode) if str(mode) else 'unknown',
                    alpha=0.85,
                )
        else:
            ax.scatter(subset_gain['plot_update_l2'], subset_gain['forget_gain'], s=60, alpha=0.85)
        for _, row in subset_gain.iterrows():
            ax.annotate(row['plot_label'], (row['plot_update_l2'], row['forget_gain']), fontsize=7)
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Actual update L2 norm')
        ax.set_ylabel('Forget loss increase')
        ax.set_title('Update Norm vs Forgetting Strength')
        ax.grid(alpha=0.25)
        if mode_col:
            ax.legend()
        outputs.extend(_save(fig, output_dir, 'update_norm_vs_forget_gain', formats))

    return outputs


def plot_guard_steps(df, output_dir, formats):
    attempted_col = 'unlearn_steps_attempted'
    accepted_col = 'unlearn_steps_accepted'
    if attempted_col not in df.columns or accepted_col not in df.columns:
        return []

    subset = df[df[[attempted_col, accepted_col]].notna().any(axis=1)].copy()
    subset = subset[subset[attempted_col].fillna(0) > 0]
    if subset.empty:
        return []

    x = np.arange(len(subset))
    width = 0.36
    fig, ax = plt.subplots(figsize=(_bar_width(len(subset)), 5))
    ax.bar(x - width / 2, subset[attempted_col], width, label='Attempted', color='#72b7b2')
    ax.bar(x + width / 2, subset[accepted_col], width, label='Accepted', color='#b279a2')
    ax.set_ylabel('Number of steps')
    ax.set_title('Guarded Unlearning Steps')
    ax.set_xticks(x)
    ax.set_xticklabels(subset['plot_label'], rotation=35, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    return _save(fig, output_dir, 'guard_steps', formats)


def plot_unlearning_time(df, output_dir, formats):
    column = 'unlearning_time_sec'
    if column not in df.columns:
        return []

    subset = df[df[column].notna()].copy()
    if subset.empty:
        return []

    x = np.arange(len(subset))
    fig, ax = plt.subplots(figsize=(_bar_width(len(subset)), 5))
    ax.bar(x, subset[column], color='#9d755d')
    ax.set_ylabel('Seconds')
    ax.set_title('Unlearning Runtime')
    ax.set_xticks(x)
    ax.set_xticklabels(subset['plot_label'], rotation=35, ha='right')
    ax.grid(axis='y', alpha=0.25)
    return _save(fig, output_dir, 'unlearning_time', formats)


def main():
    parser = argparse.ArgumentParser(description='Generate experiment figures from summary.csv.')
    parser.add_argument('summary_csv', help='Path to a summary.csv generated by summarize_experiments.py.')
    parser.add_argument(
        '--output-dir',
        default='',
        help='Directory for figures. Defaults to <summary_csv parent>/figures.',
    )
    parser.add_argument(
        '--formats',
        default='png',
        help='Comma-separated output formats, for example: png,pdf',
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_csv).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f'Summary CSV does not exist: {summary_path}')

    output_dir = Path(args.output_dir).resolve() if args.output_dir else summary_path.parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = [item.strip().lstrip('.') for item in args.formats.split(',') if item.strip()]
    if not formats:
        formats = ['png']

    df = _read_summary(summary_path)
    outputs = []
    outputs.extend(plot_forget_loss(df, output_dir, formats))
    outputs.extend(plot_global_loss(df, output_dir, formats))
    outputs.extend(plot_update_tradeoff(df, output_dir, formats))
    outputs.extend(plot_guard_steps(df, output_dir, formats))
    outputs.extend(plot_unlearning_time(df, output_dir, formats))

    if not outputs:
        print('No figures were generated. Check that summary.csv contains numeric experiment metrics.')
        return

    print(f'Generated {len(outputs)} figure file(s):')
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()
