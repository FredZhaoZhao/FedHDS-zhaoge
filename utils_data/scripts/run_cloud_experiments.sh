#!/usr/bin/env bash
set -euo pipefail

# Cloud experiment template for FedHDS retained-set unlearning.
#
# Usage:
#   bash scripts/run_cloud_experiments.sh
#
# Recommended first run:
#   RUN_SMOKE=1 bash scripts/run_cloud_experiments.sh
#
# Before running on a rented server, adjust MODEL, DATA_PATH, DEVICE, and
# RESULT_ROOT below, or pass them as environment variables.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-Qwen/Qwen2-0.5B}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data/databricks-dolly-15k.jsonl}"
DEVICE="${DEVICE:-0}"
RESULT_ROOT="${RESULT_ROOT:-formal_cloud_results}"
RUN_SMOKE="${RUN_SMOKE:-0}"

DATASET="${DATASET:-dolly}"
NUM_CLIENTS="${NUM_CLIENTS:-20}"
CLIENT_FRACTION="${CLIENT_FRACTION:-0.2}"
ROUNDS="${ROUNDS:-10}"
LOCAL_STEP="${LOCAL_STEP:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-64}"
DATA_SAMPLE="${DATA_SAMPLE:-0.2}"
IID="${IID:-0}"
FORGET_CLIENT_IDX="${FORGET_CLIENT_IDX:-0}"
HESSIAN_REF_PER_CLIENT="${HESSIAN_REF_PER_CLIENT:-4}"
UNLEARN_GRAD_SAMPLE_SIZE="${UNLEARN_GRAD_SAMPLE_SIZE:-8}"
LISSA_DEPTH="${LISSA_DEPTH:-1}"
LISSA_DAMPING="${LISSA_DAMPING:-0.01}"

if [[ "${RUN_SMOKE}" == "1" ]]; then
  NUM_CLIENTS="${SMOKE_NUM_CLIENTS:-4}"
  CLIENT_FRACTION="${SMOKE_CLIENT_FRACTION:-1.0}"
  ROUNDS="${SMOKE_ROUNDS:-1}"
  LOCAL_STEP="${SMOKE_LOCAL_STEP:-1}"
  DATA_SAMPLE="${SMOKE_DATA_SAMPLE:-0.02}"
  UNLEARN_GRAD_SAMPLE_SIZE="${SMOKE_UNLEARN_GRAD_SAMPLE_SIZE:-2}"
  HESSIAN_REF_PER_CLIENT="${SMOKE_HESSIAN_REF_PER_CLIENT:-2}"
  RESULT_ROOT="${RESULT_ROOT%/}/smoke"
fi

mkdir -p "${RESULT_ROOT}"/logs

COMMON_ARGS=(
  --dataset "${DATASET}"
  --data_path "${DATA_PATH}"
  --model "${MODEL}"
  --num_clients "${NUM_CLIENTS}"
  -k "${CLIENT_FRACTION}"
  --rounds "${ROUNDS}"
  --batch_or_epoch batch
  --local_step "${LOCAL_STEP}"
  --batch_size "${BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
  --data_sample "${DATA_SAMPLE}"
  --iid "${IID}"
  --device "${DEVICE}"
  --log
)

UNLEARN_ARGS=(
  --use_fedhds_unlearn
  --forget_client_idx "${FORGET_CLIENT_IDX}"
  --lissa_depth "${LISSA_DEPTH}"
  --lissa_damping "${LISSA_DAMPING}"
  --unlearn_grad_sample_size "${UNLEARN_GRAD_SAMPLE_SIZE}"
)

run_case() {
  local name="$1"
  shift
  local out_dir="${RESULT_ROOT}/${name}"
  local log_path="${RESULT_ROOT}/logs/${name}.log"

  echo "[RUN] ${name}"
  echo "[LOG] ${log_path}"
  "${PYTHON_BIN}" main.py "${COMMON_ARGS[@]}" "$@" --log_root "${out_dir}" 2>&1 | tee "${log_path}"
}

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Model: ${MODEL}"
echo "[INFO] Data path: ${DATA_PATH}"
echo "[INFO] Result root: ${RESULT_ROOT}"
echo "[INFO] RUN_SMOKE=${RUN_SMOKE}"

run_case "fl" \
  --zeroshot

run_case "fedhds" \
  --zeroshot \
  --filtering \
  --compound_dim 8 \
  --n_cluster 5 \
  --filtering_sample_limit 20

run_case "forget_batch_hessian" \
  --zeroshot \
  --filtering \
  --compound_dim 8 \
  --n_cluster 5 \
  --filtering_sample_limit 20 \
  "${UNLEARN_ARGS[@]}" \
  --unlearn_eta 0.01

run_case "retained_guard" \
  --zeroshot \
  --filtering \
  --compound_dim 8 \
  --n_cluster 5 \
  --filtering_sample_limit 20 \
  "${UNLEARN_ARGS[@]}" \
  --use_retained_hessian \
  --hessian_ref_per_client "${HESSIAN_REF_PER_CLIENT}" \
  --unlearn_eta 0.01 \
  --unlearn_max_update_norm 0.3 \
  --unlearn_num_steps 5 \
  --unlearn_global_loss_guard_max 0.8

"${PYTHON_BIN}" scripts/summarize_experiments.py "${RESULT_ROOT}"
"${PYTHON_BIN}" scripts/make_paper_table.py "${RESULT_ROOT}/summary.csv"
"${PYTHON_BIN}" scripts/plot_experiments.py "${RESULT_ROOT}/summary.csv" --formats png,pdf

echo "[DONE] Results are under ${RESULT_ROOT}"
