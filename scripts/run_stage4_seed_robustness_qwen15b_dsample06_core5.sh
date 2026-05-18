#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/FedHDS-zhaoge

export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

ROOT="formal_cloud_results/stage4_seed_robustness_qwen15b_dsample06_core5_20260504"
ARCHIVE="stage4_seed_robustness_qwen15b_dsample06_core5_20260504.tar.gz"
MODEL="/root/autodl-tmp/models/Qwen2.5-1.5B-ms"
DATA="/root/autodl-tmp/FedHDS-zhaoge/data/databricks-dolly-15k.jsonl"
SEEDS="${SEEDS:-43 44}"

mkdir -p "$ROOT/logs"

cat > "$ROOT/run_config.txt" <<EOF
EXP_LABEL=Stage 4 seed robustness check, large model and large data
MODEL=${MODEL}
MODEL_BASENAME=${MODEL##*/}
DATA=${DATA}
DATA_SAMPLE=0.6
ROUNDS=20
NUM_CLIENTS=20
CLIENT_FRAC=0.2
LOCAL_STEP=2
BATCH_SIZE=1
MAX_LENGTH=64
IID=0
FORGET_CLIENT_IDX=0
SEEDS=${SEEDS}
GROUPS=fl,fedhds,forget_batch_hessian,retained_auto_forget_guard_tol0_guard10,retained_negative_norm010_guard10
UNLEARN_GRAD_SAMPLE_SIZE=4
UNLEARN_ETA=0.01
HESSIAN_REF_PER_CLIENT=1
UNLEARN_NUM_STEPS=5
RETAINED_MAX_UPDATE_NORM=0.10
NEGATIVE_UPDATE_NORM=0.10
EOF

if [[ ! -e "$MODEL" ]]; then
  echo "[ERROR] MODEL path does not exist: $MODEL" >&2
  exit 2
fi

if [[ ! -f "$DATA" ]]; then
  echo "[ERROR] DATA file does not exist: $DATA" >&2
  exit 2
fi

case_done() {
  local seed="$1"
  local name="$2"
  [[ -d "$ROOT/seed${seed}/$name" ]] && find "$ROOT/seed${seed}/$name" -name final_results.json -print -quit | grep -q .
}

run_case() {
  local seed="$1"
  local name="$2"
  shift 2
  echo
  echo "========== seed${seed}/${name} =========="
  if case_done "$seed" "$name"; then
    echo "[SKIP] Existing final_results.json found for seed${seed}/${name}"
    return
  fi
  mkdir -p "$ROOT/seed${seed}/logs"
  "$PYTHON_BIN" main.py "$@" --log_root "$ROOT/seed${seed}/$name" 2>&1 | tee "$ROOT/seed${seed}/logs/${name}.log"
}

summarize_seed() {
  local seed="$1"
  "$PYTHON_BIN" scripts/summarize_experiments.py "$ROOT/seed${seed}" --output "$ROOT/seed${seed}/summary.csv"
  "$PYTHON_BIN" scripts/make_paper_table.py "$ROOT/seed${seed}/summary.csv" --output-prefix "$ROOT/seed${seed}/paper_table"
}

plot_seed() {
  local seed="$1"
  "$PYTHON_BIN" scripts/plot_experiments.py "$ROOT/seed${seed}/summary.csv" --output-dir "$ROOT/seed${seed}/figures" --formats png,pdf
}

summarize_all() {
  "$PYTHON_BIN" scripts/summarize_experiments.py "$ROOT" --output "$ROOT/summary.csv"
  "$PYTHON_BIN" scripts/make_paper_table.py "$ROOT/summary.csv" --output-prefix "$ROOT/paper_table"
}

plot_all() {
  "$PYTHON_BIN" scripts/plot_experiments.py "$ROOT/summary.csv" --output-dir "$ROOT/figures" --formats png,pdf
}

read_guard() {
  local seed="$1"
  local margin="$2"
  SUMMARY_PATH="$ROOT/seed${seed}/summary.csv" GUARD_MARGIN="$margin" "$PYTHON_BIN" - <<'PY'
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

run_seed() {
  local seed="$1"

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
    --data_sample 0.6
    --iid 0
    --device 0
    --seed "$seed"
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
    --unlearn_grad_sample_size 4
    --unlearn_eta 0.01
  )

  RETAINED_ARGS=(
    --use_retained_hessian
    --hessian_ref_per_client 1
    --unlearn_num_steps 5
  )

  run_case "$seed" fl \
    "${COMMON_ARGS[@]}" \
    --zeroshot

  run_case "$seed" fedhds \
    "${COMMON_ARGS[@]}" \
    "${FEDHDS_ARGS[@]}"

  run_case "$seed" forget_batch_hessian \
    "${COMMON_ARGS[@]}" \
    "${FEDHDS_ARGS[@]}" \
    "${BASE_UNLEARN_ARGS[@]}"

  summarize_seed "$seed"
  local guard10
  guard10="$(read_guard "$seed" 0.10)"
  echo "[INFO] seed${seed} FedHDS global-loss guard +0.10 = ${guard10}"

  run_case "$seed" retained_auto_forget_guard_tol0_guard10 \
    "${COMMON_ARGS[@]}" \
    "${FEDHDS_ARGS[@]}" \
    "${BASE_UNLEARN_ARGS[@]}" \
    "${RETAINED_ARGS[@]}" \
    --unlearn_max_update_norm 0.10 \
    --unlearn_update_sign auto \
    --unlearn_direction_check \
    --unlearn_forget_loss_guard \
    --unlearn_forget_loss_tolerance 0.0 \
    --unlearn_global_loss_guard_max "$guard10"

  run_case "$seed" retained_negative_norm010_guard10 \
    "${COMMON_ARGS[@]}" \
    "${FEDHDS_ARGS[@]}" \
    "${BASE_UNLEARN_ARGS[@]}" \
    "${RETAINED_ARGS[@]}" \
    --unlearn_max_update_norm 0.10 \
    --unlearn_update_sign negative \
    --unlearn_forget_loss_guard \
    --unlearn_forget_loss_tolerance 0.0 \
    --unlearn_global_loss_guard_max "$guard10"

  summarize_seed "$seed"
  plot_seed "$seed"
}

echo "[INFO] Stage 4 seed robustness check starts at $(date)"
echo "[INFO] ROOT=$ROOT"
echo "[INFO] MODEL=$MODEL"
echo "[INFO] CONFIG model=${MODEL##*/} data_sample=0.6 rounds=20 seeds=${SEEDS} groups=core5"

for seed in $SEEDS; do
  run_seed "$seed"
done

summarize_all
plot_all

tar -czf "$ARCHIVE" \
  "$ROOT" \
  main.py \
  server.py \
  client.py \
  scripts/run_stage4_seed_robustness_qwen15b_dsample06_core5.sh \
  scripts/summarize_experiments.py \
  scripts/make_paper_table.py \
  scripts/plot_experiments.py

echo "[DONE] Stage 4 seed robustness check finished at $(date)"
echo "[DONE] Archive: /root/autodl-tmp/FedHDS-zhaoge/${ARCHIVE}"
