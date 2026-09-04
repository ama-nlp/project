#!/usr/bin/env bash
# One-time setup, run on the Ada LOGIN node (it needs network access).
#
# Storage facts that drive this script (Ada user guide, "File Systems"):
#   /home         30 GB quota, visible on login AND compute nodes.
#   /share1       100 GB quota (confirmed). The user guide calls this
#                 "master node only"; we store models and datasets here per
#                 project convention. probe_ada.sh verifies visibility from
#                 a compute node — check it before trusting this.
#   /scratch      2 TB, node-local, purged after 7 days.
#   /ssd_scratch  960 GB, node-local, fast, purged after 7 days.
#
# Models   -> /share1/$USER/models   (HF_HOME)
# Datasets -> /share1/$USER/datasets (PROJECT_DATA_DIR)
set -euo pipefail

export HF_HOME="${HF_HOME:-/share1/$USER/models}"
export PROJECT_DATA_DIR="${PROJECT_DATA_DIR:-/share1/$USER/datasets}"
mkdir -p "$HF_HOME" "$PROJECT_DATA_DIR"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "=== quotas: /home 30 GB (venv ~10 GB), /share1 100 GB (models) ==="
quota -s 2>/dev/null || du -sh "$HOME" 2>/dev/null || true

# --- 1. Upstream files -----------------------------------------------------
# Not committed: ariahw/rl-rewardhacking has no LICENSE, so we fetch rather
# than redistribute. See scripts/fetch_upstream.sh.
bash "$REPO_DIR/scripts/fetch_upstream.sh"

# --- 2. Python environment -------------------------------------------------
export UV_PYTHON_PREFERENCE=managed
export UV_PYTHON_DOWNLOADS=automatic
uv sync --group gpu --group dev

# --- 3. Model weights ------------------------------------------------------
# With models on /share1 (100 GB) rather than /home, size is no longer the
# constraint: Qwen3-14B fp16 (~28 GB) fits comfortably. Start at 8B anyway and
# escalate only if the P3 pilot needs it — bigger models cost queue time, not
# just disk. Override with PROJECT_MODELS.
MODELS="${PROJECT_MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen3-8B}"

export HF_HUB_ENABLE_HF_TRANSFER=1
for repo in $MODELS; do
    echo "=== fetching $repo -> $HF_HOME ==="
    uv run --group gpu python -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])
"
done

echo
echo "=== disk used ==="
du -sh "$HF_HOME" "$REPO_DIR/.venv" 2>/dev/null || true
echo
echo "Add to ~/.bashrc on Ada:"
echo "  export HF_HOME=$HF_HOME"
echo "  export PROJECT_DATA_DIR=$PROJECT_DATA_DIR"
