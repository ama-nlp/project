#!/bin/bash
# P0 gate on Ada, cheap version: Qwen3-0.6B over 8 problems.
# Proves transformers loads the model, the chat template emits <think>, the
# sandbox survives SLURM's cgroup limits, and traces validate. Minutes, one GPU.
#
#   sbatch slurm/smoke.sbatch
#
#SBATCH --job-name=project-smoke
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=00:45:00
#SBATCH --output=logs/smoke-%j.out
#
# NOTE: logs/ must already exist. SLURM opens --output before the script runs,
# so the mkdir below is too late; without the directory the job dies at launch
# with no log at all. logs/.gitkeep keeps it in the repo.
#SBATCH --error=logs/smoke-%j.err
#
# Why these numbers:
#   -c 10          Ada policy is at most 10 CPUs per GPU; jobs breaking the
#                  1:10 ratio may be killed without warning.
#   --mem=30G      MaxMemPerCPU is 3000 MB, so 10 CPUs caps at 30 GB.
#   no module load Ada has no cuda modules; torch ships its own runtime.

set -euo pipefail
mkdir -p logs

# Storage, settled by scripts/probe_ada.sh on gnode063:
#   /share1      NOT visible from compute nodes. Cannot hold weights.
#   /home        30 GB quota, visible everywhere. Qwen3-8B fp16 (~16 GB) plus
#                the venv (~6 GB without vLLM) fits with room to spare.
#   /ssd_scratch 869 GB node-local, and compute nodes DO have internet, so a
#                model too big for /home can be pulled straight to scratch at
#                job start. Set PROJECT_HF_SCRATCH=1 for that; it re-downloads
#                whenever the job lands on a new node.
if [ "${PROJECT_HF_SCRATCH:-0}" = "1" ]; then
    export HF_HOME="/ssd_scratch/$USER/hf"
    mkdir -p "$HF_HOME"
else
    export HF_HOME="${HF_HOME:-$HOME/hf}"
fi
# Compute nodes have network access, so a cache miss downloads rather than
# crashing. Pre-staging via setup_ada.sh just makes the first job faster.
export UV_PYTHON_PREFERENCE=managed
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-10}"  # sandbox workers = allocated cores
export PROJECT_RUNS_DIR="${PROJECT_RUNS_DIR:-runs}"
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# We cannot --exclude the 1080 Ti nodes: gnode013-015, 018, 024 and 039 do not
# exist, so SLURM rejects the whole range as "Invalid node name specified".
# Report the card instead. The sm_70 floor was vLLM's; transformers runs on
# Pascal too, just slower (no fp16 tensor cores), so this is a warning.
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "compute capability: $CAP"
if [ "${CAP%%.*}" -lt 7 ]; then
    echo "WARNING: compute_cap $CAP (GTX 1080 Ti) - usable, but fp16 is slow here."
fi

# torch 2.6.0 bundles CUDA 12.4. Via CUDA minor-version compatibility that needs
# driver >= 525; older drivers fail deep inside torch with an opaque
# "CUDA driver version is insufficient" instead of anything actionable.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "driver version: $DRV"
if [ "${DRV%%.*}" -lt 525 ]; then
    echo "FATAL: driver $DRV is too old for the CUDA 12.4 runtime in torch 2.6."
    echo "Reinstall torch built against CUDA 11.8:"
    echo "  uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

uv run --group gpu project generate \
    --backend hf \
    --model Qwen/Qwen3-0.6B \
    --dtype float16 \
    --micro_batch "${PROJECT_MICRO_BATCH:-4}" \
    --arm C \
    --n 8 \
    --max_tokens 4096 \
    --seed 0
