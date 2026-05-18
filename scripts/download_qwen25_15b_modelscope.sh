#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp
mkdir -p models/Qwen2.5-1.5B-ms logs

/root/miniconda3/bin/python - <<'PY'
from modelscope import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-1.5B",
    local_dir="/root/autodl-tmp/models/Qwen2.5-1.5B-ms",
)
PY

test -f /root/autodl-tmp/models/Qwen2.5-1.5B-ms/config.json
echo "[DONE] Qwen2.5-1.5B downloaded to /root/autodl-tmp/models/Qwen2.5-1.5B-ms"
