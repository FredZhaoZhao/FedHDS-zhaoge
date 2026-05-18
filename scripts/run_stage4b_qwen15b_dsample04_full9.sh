#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/FedHDS-zhaoge

export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

ROOT="formal_cloud_results/stage4b_qwen15b_dsample04_full9_20260504"
ARCHIVE="stage4b_qwen15b_dsample04_full9_20260504.tar.gz"
MODEL="/root/autodl-tmp/models/Qwen2.5-1.5B-ms"

mkdir -p "$ROOT"

env \
  DATE_TAG=20260504 \
  ROOT="$ROOT" \
  ARCHIVE="$ARCHIVE" \
  MODEL="$MODEL" \
  DATA_SAMPLE=0.4 \
  ROUNDS=20 \
  RUN_FULL_ABLATION=1 \
  EXP_LABEL="Stage 4B large-model/small-data experiment (Qwen2.5-1.5B, data_sample=0.4, full-9)" \
  EXTRA_ARCHIVE_FILES="scripts/run_stage4b_qwen15b_dsample04_full9.sh" \
  bash scripts/run_stage4_large_model_data.sh
