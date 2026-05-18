#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/FedHDS-zhaoge

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HOME=/root/autodl-tmp/cache
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
ROOT="formal_cloud_results/stage3b_negative_norm_sweep_20260503"
PREV_ROOT="formal_cloud_results/stage3_direction_forget_guard_20260503"
MODEL="/root/autodl-tmp/models/Qwen2-0.5B-ms"
DATA="/root/autodl-tmp/FedHDS-zhaoge/data/databricks-dolly-15k.jsonl"

COMMON_ARGS=(
  --dataset dolly
  --data_path "$DATA"
  --model "$MODEL"
  --num_clients 20
  -k 0.2
  --rounds 20
  --batch_or_epoch batch
  --local_step 2
  --batch_size 1
  --max_length 64
  --data_sample 0.4
  --iid 0
  --device 0
  --log
)

FEDHDS_ARGS=(
  --zeroshot
  --filtering
  --compound_dim 8
  --n_cluster 5
  --filtering_sample_limit 20
)

BASE_UNLEARN_ARGS=(
  --use_fedhds_unlearn
  --forget_client_idx 0
  --lissa_depth 1
  --lissa_damping 0.01
  --unlearn_grad_sample_size 8
  --unlearn_eta 0.01
  --use_retained_hessian
  --hessian_ref_per_client 1
  --unlearn_num_steps 5
  --unlearn_update_sign negative
  --unlearn_forget_loss_guard
  --unlearn_forget_loss_tolerance 0.0
)

mkdir -p "$ROOT/logs"

echo "[INFO] Python: $("$PYTHON_BIN" -V)"
echo "[INFO] Result root: $ROOT"

if [[ ! -f "$PREV_ROOT/summary.csv" ]]; then
  echo "[ERROR] Missing previous stage3 summary: $PREV_ROOT/summary.csv"
  exit 1
fi

GUARD05=$("$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path

p = Path("formal_cloud_results/stage3_direction_forget_guard_20260503/summary.csv")
rows = list(csv.DictReader(p.open()))
fed = next(r for r in rows if r["group"] == "fedhds")
loss = float(fed["final_global_loss"])
print(f"{loss + 0.05:.6f}")
PY
)

GUARD10=$("$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path

p = Path("formal_cloud_results/stage3_direction_forget_guard_20260503/summary.csv")
rows = list(csv.DictReader(p.open()))
fed = next(r for r in rows if r["group"] == "fedhds")
loss = float(fed["final_global_loss"])
print(f"{loss + 0.10:.6f}")
PY
)

echo "[INFO] Using global guards: +0.05=${GUARD05}, +0.10=${GUARD10}"

run_case() {
  local name="$1"
  local max_norm="$2"
  local guard="$3"
  shift 3

  echo "[RUN] ${name}"
  "$PYTHON_BIN" main.py "${COMMON_ARGS[@]}" "${FEDHDS_ARGS[@]}" "${BASE_UNLEARN_ARGS[@]}" \
    --unlearn_max_update_norm "$max_norm" \
    --unlearn_global_loss_guard_max "$guard" \
    --log_root "$ROOT/$name" \
    "$@" \
    2>&1 | tee "$ROOT/logs/${name}.log"
}

run_case "negative_norm005_guard05" 0.05 "$GUARD05"
run_case "negative_norm010_guard10" 0.10 "$GUARD10"
run_case "negative_norm015_guard10" 0.15 "$GUARD10"

echo "[INFO] Final summary/table/figures"
"$PYTHON_BIN" scripts/summarize_experiments.py "$ROOT"
"$PYTHON_BIN" scripts/make_paper_table.py "$ROOT/summary.csv"
"$PYTHON_BIN" scripts/plot_experiments.py "$ROOT/summary.csv" --formats png,pdf

echo "[DONE] Stage3b results under $ROOT"
