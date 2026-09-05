#!/usr/bin/env bash
# One-time setup. Run it on a COMPUTE node:
#
#   sbatch slurm/setup.sbatch
#
# NOT on the login node. Two independent reasons, both observed:
#   1. Wheels. The login node is RHEL 7 (gcc 4.8.5, old glibc), so pip finds no
#      manylinux wheel for numpy and uv falls back to an sdist build, which
#      dies at "ERROR: Compiler cython cannot compile programs". Compute nodes
#      are u22 (Ubuntu 22.04) and the wheel simply downloads.
#   2. Memory. The login node is shared and capped; even when a build is not
#      needed, unpacking torch there aborts with "memory allocation of N bytes
#      failed".
# Compute nodes reach the internet (verified HTTP 200) and see the same $HOME,
# so nothing is lost by running it there.
#
# Storage facts that drive this script (Ada user guide, "File Systems"):
#   /home         30 GB quota, visible on login AND compute nodes.
#   /share1       100 GB, LOGIN NODE ONLY. Verified: on gnode063 the path does
#                 not exist at all. Not mounted is not unreachable -- a job
#                 scps from the login node at start-up (SubliminalMerge
#                 job_01, job_30; see slurm/stage_model.sh).
#   /scratch      2 TB, node-local, purged after 7 days.
#   /ssd_scratch  960 GB, node-local, fast, purged after 7 days.
#
# So the split is:
#   weights  -> /share1/$USER/models   (staged to /scratch at job start)
#   datasets -> the repo's data/, fetched by fetch_upstream.sh (3.7 MB)
#   code, venv, runs/ -> $HOME
# Weights are deliberately NOT under $HOME. venv ~6 GB + 8B ~16 GB + 0.6B
# ~1.5 GB is ~24 GB of a 30 GB quota, which leaves nothing for the traces P5
# writes, and a 14B (~28 GB) does not fit at all.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "=== /home quota: 30 GB: code, venv (~6 GB), runs/. No weights here. ==="
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
# The venv is ~6 GB and a ~4 GB cache on top of it is real pressure once runs/
# starts filling. On a compute node the cache is scratch and disposable.
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

# --- 3. Model weights: NOT here ---------------------------------------------
# Weights live in /share1/$USER/models and are fetched by a separate job:
#
#   sbatch slurm/download_model.sbatch
#
# That downloads on a compute node into node-local /scratch and scps up to
# /share1 (SubliminalMerge/download_model.sh). It is a job, not part of this
# script, because a 16 GB download has no business on the shared, memory-capped
# login node -- and because keeping /share1 out of here means this script runs
# on either kind of node.
echo
echo "=== venv ready ==="
du -sh "$REPO_DIR/.venv" 2>/dev/null || true
echo
echo "Next: sbatch slurm/download_model.sbatch   (weights -> /share1/$USER/models)"
echo "If ~/hf still exists it is a leftover model cache: rm -rf ~/hf"
