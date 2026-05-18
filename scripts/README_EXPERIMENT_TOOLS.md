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

## 3. Run Loss-Based MIA

Loss-based MIA reads saved LoRA checkpoints and writes metrics back into each
`final_results.json`, so new V3 runs should include `--save`.

Evaluate one run or a whole result root:

```bash
python scripts/mia_loss_based.py formal_cloud_results
```

Optional controls:

```bash
python scripts/mia_loss_based.py formal_cloud_results --checkpoint-label post_unlearn --sample-limit 128
```

New metrics written into `experiment_metrics` include:

- `mia_loss_auc`
- `mia_loss_tpr_at_fpr001`
- `mia_member_count`
- `mia_nonmember_count`

## 4. Generate Figures

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

## 5. Cloud Run Template

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

## 6. V3 Core Suite

For the V3 paper-strengthening runs, use the dedicated suite. It adds:

- `--save` checkpoints for later MIA
- `ga`
- `ga_guarded`
- `retrain_oracle`
- automatic `MIA -> summary -> paper_table -> figures`

Example:

```bash
PYTHON_BIN=/path/to/python \
MODEL=/path/to/local/model/or/hf_id \
DATA=/path/to/databricks-dolly-15k.jsonl \
ROOT=formal_cloud_results/v3_iid_seed5 \
SEEDS="42 43 44 45 46" \
IID=0 \
bash scripts/run_v3_core_suite.sh
```

For the non-IID validation, keep the same script and change only `IID`:

```bash
IID=dir0.5 bash scripts/run_v3_core_suite.sh
```
