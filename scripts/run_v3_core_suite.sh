#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT="${ROOT:-formal_cloud_results/v3_core_$(date +%Y%m%d)}"
ARCHIVE="${ARCHIVE:-$(basename "${ROOT}").tar.gz}"
MODEL="${MODEL:-Qwen/Qwen2-0.5B}"
DATA="${DATA:-${PROJECT_ROOT}/data/databricks-dolly-15k.jsonl}"
SEEDS="${SEEDS:-42}"

DATASET="${DATASET:-dolly}"
NUM_CLIENTS="${NUM_CLIENTS:-20}"
CLIENT_FRAC="${CLIENT_FRAC:-0.2}"
ROUNDS="${ROUNDS:-20}"
LOCAL_STEP="${LOCAL_STEP:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-64}"
DATA_SAMPLE="${DATA_SAMPLE:-0.4}"
IID="${IID:-0}"
FORGET_CLIENT_IDX="${FORGET_CLIENT_IDX:-0}"

LISSA_DEPTH="${LISSA_DEPTH:-1}"
LISSA_DAMPING="${LISSA_DAMPING:-0.01}"
UNLEARN_GRAD_SAMPLE_SIZE="${UNLEARN_GRAD_SAMPLE_SIZE:-8}"
UNLEARN_ETA="${UNLEARN_ETA:-0.01}"
HESSIAN_REF_PER_CLIENT="${HESSIAN_REF_PER_CLIENT:-2}"
UNLEARN_NUM_STEPS="${UNLEARN_NUM_STEPS:-5}"
UPDATE_NORM_CAP="${UPDATE_NORM_CAP:-0.10}"
GUARD_MARGIN="${GUARD_MARGIN:-0.10}"
MIA_SAMPLE_LIMIT="${MIA_SAMPLE_LIMIT:-0}"

mkdir -p "${ROOT}/logs"

case_done() {
  local seed="$1"
  local name="$2"
  [[ -d "${ROOT}/seed${seed}/${name}" ]] && find "${ROOT}/seed${seed}/${name}" -name final_results.json -print -quit | grep -q .
}

run_case() {
  local seed="$1"
  local name="$2"
  shift 2

  echo
  echo "========== seed${seed}/${name} =========="
  if case_done "${seed}" "${name}"; then
    echo "[SKIP] Existing final_results.json found for seed${seed}/${name}"
    return
  fi

  mkdir -p "${ROOT}/seed${seed}/logs"
  "${PYTHON_BIN}" main.py "$@" --log_root "${ROOT}/seed${seed}/${name}" 2>&1 | tee "${ROOT}/seed${seed}/logs/${name}.log"
}

summarize_seed() {
  local seed="$1"
  "${PYTHON_BIN}" scripts/summarize_experiments.py "${ROOT}/seed${seed}" --output "${ROOT}/seed${seed}/summary.csv"
}

table_seed() {
  local seed="$1"
  "${PYTHON_BIN}" scripts/make_paper_table.py "${ROOT}/seed${seed}/summary.csv" --output-prefix "${ROOT}/seed${seed}/paper_table"
}

plot_seed() {
  local seed="$1"
  "${PYTHON_BIN}" scripts/plot_experiments.py "${ROOT}/seed${seed}/summary.csv" --output-dir "${ROOT}/seed${seed}/figures" --formats png,pdf
}

run_mia_seed() {
  local seed="$1"
  "${PYTHON_BIN}" scripts/mia_loss_based.py "${ROOT}/seed${seed}" --checkpoint-label auto --sample-limit "${MIA_SAMPLE_LIMIT}"
}

read_guard() {
  local seed="$1"
  local margin="$2"
  SUMMARY_PATH="${ROOT}/seed${seed}/summary.csv" GUARD_MARGIN="${margin}" "${PYTHON_BIN}" - <<'PY'
import csv
import os
from pathlib import Path

summary_path = Path(os.environ["SUMMARY_PATH"])
margin = float(os.environ["GUARD_MARGIN"])

with summary_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

fedhds = next(row for row in rows if row.get("group") == "fedhds")
loss = float(fedhds["final_global_loss"])
print(f"{loss + margin:.6f}")
PY
}

summarize_all() {
  "${PYTHON_BIN}" scripts/summarize_experiments.py "${ROOT}" --output "${ROOT}/summary.csv"
  "${PYTHON_BIN}" scripts/make_paper_table.py "${ROOT}/summary.csv" --output-prefix "${ROOT}/paper_table"
}

plot_all() {
  "${PYTHON_BIN}" scripts/plot_experiments.py "${ROOT}/summary.csv" --output-dir "${ROOT}/figures" --formats png,pdf
}

run_seed() {
  local seed="$1"

  COMMON_ARGS=(
    --dataset "${DATASET}"
    --data_path "${DATA}"
    --model "${MODEL}"
    --num_clients "${NUM_CLIENTS}"
    -k "${CLIENT_FRAC}"
    --rounds "${ROUNDS}"
    --batch_or_epoch batch
    --local_step "${LOCAL_STEP}"
    --batch_size "${BATCH_SIZE}"
    --max_length "${MAX_LENGTH}"
    --data_sample "${DATA_SAMPLE}"
    --iid "${IID}"
    --device 0
    --seed "${seed}"
    --zeroshot
    --save
    --log
  )

  FEDHDS_ARGS=(
    --filtering
    --compound_dim 8
    --n_cluster 5
    --filtering_sample_limit 20
  )

  BASE_UNLEARN_ARGS=(
    --use_fedhds_unlearn
    --forget_client_idx "${FORGET_CLIENT_IDX}"
    --lissa_depth "${LISSA_DEPTH}"
    --lissa_damping "${LISSA_DAMPING}"
    --unlearn_grad_sample_size "${UNLEARN_GRAD_SAMPLE_SIZE}"
    --unlearn_eta "${UNLEARN_ETA}"
  )

  run_case "${seed}" fl \
    "${COMMON_ARGS[@]}"

  run_case "${seed}" fedhds \
    "${COMMON_ARGS[@]}" \
    "${FEDHDS_ARGS[@]}"

  run_case "${seed}" retrain_oracle \
    "${COMMON_ARGS[@]}" \
    "${FEDHDS_ARGS[@]}" \
    --retrain_exclude_client "${FORGET_CLIENT_IDX}"

  run_case "${seed}" forget_batch_hessian \
    "${COMMON_ARGS[@]}" \
    "${FEDHDS_ARGS[@]}" \
    "${BASE_UNLEARN_ARGS[@]}"

  summarize_seed "${seed}"
  local guard10
  guard10="$(read_guard "${seed}" "${GUARD_MARGIN}")"
  echo "[INFO] seed${seed} FedHDS global-loss guard +${GUARD_MARGIN} = ${guard10}"

  run_case "${seed}" retained_negative_guard \
    "${COMMON_ARGS[@]}" \
    "${FEDHDS_ARGS[@]}" \
    "${BASE_UNLEARN_ARGS[@]}" \
    --use_retained_hessian \
    --hessian_ref_per_client "${HESSIAN_REF_PER_CLIENT}" \
    --unlearn_num_steps "${UNLEARN_NUM_STEPS}" \
    --unlearn_max_update_norm "${UPDATE_NORM_CAP}" \
    --unlearn_update_sign negative \
    --unlearn_forget_loss_guard \
    --unlearn_forget_loss_tolerance 0.0 \
    --unlearn_global_loss_guard_max "${guard10}"

  run_case "${seed}" ga \
    "${COMMON_ARGS[@]}" \
    "${FEDHDS_ARGS[@]}" \
    "${BASE_UNLEARN_ARGS[@]}" \
    --unlearn_method ga

  run_case "${seed}" ga_guarded \
    "${COMMON_ARGS[@]}" \
    "${FEDHDS_ARGS[@]}" \
    "${BASE_UNLEARN_ARGS[@]}" \
    --unlearn_method ga_guarded \
    --unlearn_num_steps "${UNLEARN_NUM_STEPS}" \
    --unlearn_max_update_norm "${UPDATE_NORM_CAP}" \
    --unlearn_forget_loss_guard \
    --unlearn_forget_loss_tolerance 0.0 \
    --unlearn_global_loss_guard_max "${guard10}"

  run_mia_seed "${seed}"
  summarize_seed "${seed}"
  table_seed "${seed}"
  plot_seed "${seed}"
}

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] ROOT=${ROOT}"
echo "[INFO] MODEL=${MODEL}"
echo "[INFO] DATA=${DATA}"
echo "[INFO] IID=${IID}"
echo "[INFO] SEEDS=${SEEDS}"

for seed in ${SEEDS}; do
  run_seed "${seed}"
done

summarize_all
plot_all

tar -czf "${ARCHIVE}" \
  "${ROOT}" \
  main.py \
  server.py \
  cli_config.py \
  unlearning_methods.py \
  scripts/mia_loss_based.py \
  scripts/summarize_experiments.py \
  scripts/make_paper_table.py

echo "[DONE] Results are under ${ROOT}"
echo "[DONE] Archive written to ${ARCHIVE}"
