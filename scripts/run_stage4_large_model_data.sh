#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/FedHDS-zhaoge

export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

DATE_TAG="${DATE_TAG:-20260504}"
ROOT="${ROOT:-formal_cloud_results/stage4_large_model_data_${DATE_TAG}}"
ARCHIVE="${ARCHIVE:-stage4_large_model_data_${DATE_TAG}.tar.gz}"

MODEL="${MODEL:-/root/autodl-tmp/models/Qwen2.5-1.5B-ms}"
DATA="${DATA:-/root/autodl-tmp/FedHDS-zhaoge/data/databricks-dolly-15k.jsonl}"

DATA_SAMPLE="${DATA_SAMPLE:-0.6}"
ROUNDS="${ROUNDS:-20}"
NUM_CLIENTS="${NUM_CLIENTS:-20}"
CLIENT_FRAC="${CLIENT_FRAC:-0.2}"
LOCAL_STEP="${LOCAL_STEP:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-64}"

UNLEARN_GRAD_SAMPLE_SIZE="${UNLEARN_GRAD_SAMPLE_SIZE:-4}"
UNLEARN_ETA="${UNLEARN_ETA:-0.01}"
HESSIAN_REF_PER_CLIENT="${HESSIAN_REF_PER_CLIENT:-1}"
UNLEARN_NUM_STEPS="${UNLEARN_NUM_STEPS:-5}"
RETAINED_MAX_UPDATE_NORM="${RETAINED_MAX_UPDATE_NORM:-0.10}"
NEGATIVE_UPDATE_NORM="${NEGATIVE_UPDATE_NORM:-0.10}"

RUN_FULL_ABLATION="${RUN_FULL_ABLATION:-0}"

if [[ ! -e "$MODEL" ]]; then
  echo "[ERROR] MODEL path does not exist: $MODEL" >&2
  echo "[ERROR] Set MODEL=/path/to/local/model before running this script." >&2
  exit 2
fi

if [[ ! -f "$DATA" ]]; then
  echo "[ERROR] DATA file does not exist: $DATA" >&2
  exit 2
fi

mkdir -p "$ROOT/logs"

COMMON_ARGS=(
  --dataset dolly
  --data_path "$DATA"
  --model "$MODEL"
  --num_clients "$NUM_CLIENTS"
  -k "$CLIENT_FRAC"
  --rounds "$ROUNDS"
  --batch_or_epoch batch
  --local_step "$LOCAL_STEP"
  --batch_size "$BATCH_SIZE"
  --max_length "$MAX_LENGTH"
  --data_sample "$DATA_SAMPLE"
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
  --unlearn_grad_sample_size "$UNLEARN_GRAD_SAMPLE_SIZE"
  --unlearn_eta "$UNLEARN_ETA"
)

RETAINED_ARGS=(
  --use_retained_hessian
  --hessian_ref_per_client "$HESSIAN_REF_PER_CLIENT"
  --unlearn_num_steps "$UNLEARN_NUM_STEPS"
)

run_case() {
  local name="$1"
  shift
  echo
  echo "========== ${name} =========="
  "$PYTHON_BIN" main.py "$@" --log_root "$ROOT/$name" 2>&1 | tee "$ROOT/logs/${name}.log"
}

summarize_now() {
  if "$PYTHON_BIN" scripts/summarize_experiments.py "$ROOT" --output "$ROOT/summary.csv"; then
    :
  else
    "$PYTHON_BIN" scripts/summarize_experiments.py "$ROOT" --out "$ROOT/summary.csv"
  fi
  "$PYTHON_BIN" scripts/make_paper_table.py "$ROOT/summary.csv" --output-prefix "$ROOT/paper_table"
}

plot_now() {
  if "$PYTHON_BIN" scripts/plot_experiments.py "$ROOT/summary.csv" --output-dir "$ROOT/figures" --formats png,pdf; then
    :
  else
    "$PYTHON_BIN" scripts/plot_experiments.py "$ROOT/summary.csv" --out-dir "$ROOT/figures"
  fi
}

read_guard() {
  local margin="$1"
  SUMMARY_PATH="$ROOT/summary.csv" GUARD_MARGIN="$margin" "$PYTHON_BIN" - <<'PY'
import csv
import os
from pathlib import Path

summary_path = Path(os.environ["SUMMARY_PATH"])
margin = float(os.environ["GUARD_MARGIN"])

with summary_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

fedhds = next(row for row in rows if row.get("group") == "fedhds")
loss = float(fedhds["final_global_loss"])
print(f"{loss + margin:.6f}")
PY
}

run_negative_case() {
  local name="$1"
  local max_norm="$2"
  local guard="$3"
  run_case "$name" \
    "${COMMON_ARGS[@]}" \
    "${FEDHDS_ARGS[@]}" \
    "${BASE_UNLEARN_ARGS[@]}" \
    "${RETAINED_ARGS[@]}" \
    --unlearn_max_update_norm "$max_norm" \
    --unlearn_update_sign negative \
    --unlearn_forget_loss_guard \
    --unlearn_forget_loss_tolerance 0.0 \
    --unlearn_global_loss_guard_max "$guard"
}

echo "[INFO] Stage 4 large-model/large-data experiment starts at $(date)"
echo "[INFO] ROOT=$ROOT"
echo "[INFO] MODEL=$MODEL"
echo "[INFO] DATA_SAMPLE=$DATA_SAMPLE ROUNDS=$ROUNDS RUN_FULL_ABLATION=$RUN_FULL_ABLATION"

run_case fl \
  "${COMMON_ARGS[@]}" \
  --zeroshot

run_case fedhds \
  "${COMMON_ARGS[@]}" \
  "${FEDHDS_ARGS[@]}"

run_case forget_batch_hessian \
  "${COMMON_ARGS[@]}" \
  "${FEDHDS_ARGS[@]}" \
  "${BASE_UNLEARN_ARGS[@]}"

summarize_now
GUARD05="$(read_guard 0.05)"
GUARD10="$(read_guard 0.10)"
echo "[INFO] FedHDS global-loss guard +0.05 = ${GUARD05}"
echo "[INFO] FedHDS global-loss guard +0.10 = ${GUARD10}"

run_case retained_auto_forget_guard_tol0_guard10 \
  "${COMMON_ARGS[@]}" \
  "${FEDHDS_ARGS[@]}" \
  "${BASE_UNLEARN_ARGS[@]}" \
  "${RETAINED_ARGS[@]}" \
  --unlearn_max_update_norm "$RETAINED_MAX_UPDATE_NORM" \
  --unlearn_update_sign auto \
  --unlearn_direction_check \
  --unlearn_forget_loss_guard \
  --unlearn_forget_loss_tolerance 0.0 \
  --unlearn_global_loss_guard_max "$GUARD10"

run_negative_case retained_negative_norm010_guard10 "$NEGATIVE_UPDATE_NORM" "$GUARD10"

if [[ "$RUN_FULL_ABLATION" == "1" ]]; then
  run_case retained_auto_direction_guard05 \
    "${COMMON_ARGS[@]}" \
    "${FEDHDS_ARGS[@]}" \
    "${BASE_UNLEARN_ARGS[@]}" \
    "${RETAINED_ARGS[@]}" \
    --unlearn_max_update_norm "$RETAINED_MAX_UPDATE_NORM" \
    --unlearn_update_sign auto \
    --unlearn_direction_check \
    --unlearn_global_loss_guard_max "$GUARD05"

  run_case retained_auto_forget_guard_tol005_guard10 \
    "${COMMON_ARGS[@]}" \
    "${FEDHDS_ARGS[@]}" \
    "${BASE_UNLEARN_ARGS[@]}" \
    "${RETAINED_ARGS[@]}" \
    --unlearn_max_update_norm "$RETAINED_MAX_UPDATE_NORM" \
    --unlearn_update_sign auto \
    --unlearn_direction_check \
    --unlearn_forget_loss_guard \
    --unlearn_forget_loss_tolerance 0.005 \
    --unlearn_global_loss_guard_max "$GUARD10"

  run_negative_case retained_negative_norm005_guard05 0.05 "$GUARD05"
  run_negative_case retained_negative_norm015_guard10 0.15 "$GUARD10"
fi

summarize_now
plot_now

tar -czf "$ARCHIVE" \
  "$ROOT" \
  main.py \
  server.py \
  client.py \
  scripts/run_stage4_large_model_data.sh \
  scripts/summarize_experiments.py \
  scripts/make_paper_table.py \
  scripts/plot_experiments.py

echo "[DONE] Stage 4 large-model/large-data experiment finished at $(date)"
echo "[DONE] Archive: /root/autodl-tmp/FedHDS-zhaoge/${ARCHIVE}"
