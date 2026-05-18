# Experiment Tooling

This folder contains helper scripts for cloud runs, result tables, and figures.

## 1. Summarize Raw Runs

Each training run writes a `final_results.json`. Recursively summarize a result root:

```bash
python scripts/summarize_experiments.py formal_cloud_results
```

Output:

```text
formal_cloud_results/summary.csv
```

## 2. Generate Paper Tables

Create compact CSV and Markdown tables from a `summary.csv`:

```bash
python scripts/make_paper_table.py formal_cloud_results/summary.csv
```

Outputs:

```text
formal_cloud_results/paper_table.csv
formal_cloud_results/paper_table.md
```

## 3. Generate Figures

Create PNG figures:

```bash
python scripts/plot_experiments.py formal_cloud_results/summary.csv
```

Create both PNG and PDF figures:

```bash
python scripts/plot_experiments.py formal_cloud_results/summary.csv --formats png,pdf
```

Default output:

```text
formal_cloud_results/figures/
```

Generated figures include:

- `forget_loss_before_after`
- `global_loss_final`
- `update_norm_vs_global_loss`
- `update_norm_vs_forget_gain`
- `guard_steps`
- `unlearning_time`

## 4. Cloud Run Template

Run a quick cloud smoke test first:

```bash
RUN_SMOKE=1 \
PYTHON_BIN=/path/to/python \
MODEL=/path/to/local/model/or/hf_id \
DATA_PATH=/path/to/databricks-dolly-15k.jsonl \
DEVICE=0 \
bash scripts/run_cloud_experiments.sh
```

Run the default formal template:

```bash
PYTHON_BIN=/path/to/python \
MODEL=/path/to/local/model/or/hf_id \
DATA_PATH=/path/to/databricks-dolly-15k.jsonl \
DEVICE=0 \
RESULT_ROOT=formal_cloud_results \
bash scripts/run_cloud_experiments.sh
```

The template runs:

- `fl`
- `fedhds`
- `forget_batch_hessian`
- `retained_guard`

Then it automatically writes:

- `summary.csv`
- `paper_table.csv`
- `paper_table.md`
- `figures/*.png`
- `figures/*.pdf`

Adjust experiment scale through environment variables such as:

```bash
NUM_CLIENTS=20
CLIENT_FRACTION=0.2
ROUNDS=10
DATA_SAMPLE=0.2
LOCAL_STEP=2
FORGET_CLIENT_IDX=0
```
