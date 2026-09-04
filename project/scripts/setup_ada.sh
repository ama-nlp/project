#!/usr/bin/env bash
# One-time setup, run on the Ada LOGIN node (it needs network access).
#
# Storage facts that drive this script (Ada user guide, "File Systems"):
#   /home         30 GB quota, visible on login AND compute nodes.
#   /share1       100 GB, LOGIN NODE ONLY. Verified: on gnode063 the path does
#                 not exist at all. Useless for anything a job needs at runtime.
#                 Still fine for archiving finished results off the home quota.
#   /scratch      2 TB, node-local, purged after 7 days.
#   /ssd_scratch  960 GB, node-local, fast, purged after 7 days.
#
# So everything a job touches lives under $HOME:
#   models   -> $HOME/hf   (HF_HOME)
#   datasets -> the repo's data/, fetched by fetch_upstream.sh (3.7 MB)
# Budget: venv ~6 GB + Qwen3-8B fp16 ~16 GB + Qwen3-0.6B ~1.5 GB = ~24 GB of 30.
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/hf}"
mkdir -p "$HF_HOME"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "=== /home quota: 30 GB. venv ~6 GB + 8B ~16 GB + 0.6B ~1.5 GB = ~24 GB ==="
quota -s 2>/dev/null || du -sh "$HOME" 2>/dev/null || true

# --- 1. Upstream files -----------------------------------------------------
# Not committed: ariahw/rl-rewardhacking has no LICENSE, so we fetch rather
# than redistribute. See scripts/fetch_upstream.sh.
bash "$REPO_DIR/scripts/fetch_upstream.sh"

# --- 2. Python environment -------------------------------------------------
export UV_PYTHON_PREFERENCE=managed
export UV_PYTHON_DOWNLOADS=automatic

# The login node is shared and memory-capped: unthrottled, uv aborts with
# "memory allocation of N bytes failed" while unpacking torch's ~800 MB wheel
# and the nvidia-* libraries alongside it. Serialising keeps peak memory low.
# Running this under SLURM instead (slurm/setup.sbatch) avoids the cap entirely.
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"

# Keep the wheel cache off the 30 GB home quota when we have node-local disk.
# venv (~6 GB) + models (~18 GB) already comes to ~24 GB; a ~4 GB cache on top
# would overflow. On a compute node the cache is scratch and disposable.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -d /scratch ]; then
    export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/$USER/uv-cache}"
    mkdir -p "$UV_CACHE_DIR"
    echo "uv cache -> $UV_CACHE_DIR (node-local, off quota)"
fi

uv sync --group gpu --group dev

# On the login node the cache does land in $HOME; reclaim it before the model
# download needs the space.
if [ -z "${UV_CACHE_DIR:-}" ]; then
    uv cache prune >/dev/null 2>&1 || true
fi

# --- 3. Model weights ------------------------------------------------------
# Qwen3-8B fp16 (~16 GB) fits the 30 GB home quota alongside the venv.
# Qwen3-14B (~28 GB) does not. If P3 needs it, either ask hpc.admin for a quota
# increase, or stage to node-local /ssd_scratch at job start — compute nodes do
# have internet (verified HTTP 200), so a per-node download is possible.
# Override with PROJECT_MODELS.
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
echo "Add to ~/.bashrc on Ada:  export HF_HOME=$HF_HOME"
