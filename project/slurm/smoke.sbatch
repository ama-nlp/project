#!/bin/bash
# P0 gate on Ada, cheap version: Qwen3-0.6B over 8 problems.
# Proves vLLM loads, the chat template emits <think>, the sandbox runs under
# SLURM's cgroup limits, and traces validate. Minutes, one GPU.
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
#SBATCH --error=logs/smoke-%j.err
#
# Why these numbers:
#   -c 10          Ada policy is at most 10 CPUs per GPU; jobs breaking the
#                  1:10 ratio may be killed without warning.
#   --mem=30G      MaxMemPerCPU is 3000 MB, so 10 CPUs caps at 30 GB.
#   no module load Ada has no cuda modules; vLLM ships its own runtime.

set -euo pipefail
mkdir -p logs

export HF_HOME="${HF_HOME:-/share1/$USER/models}"
export PROJECT_DATA_DIR="${PROJECT_DATA_DIR:-/share1/$USER/datasets}"
export HF_HUB_OFFLINE=1                       # weights pre-cached by setup_ada.sh
export UV_PYTHON_PREFERENCE=managed
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-10}"  # sandbox workers = allocated cores
export PROJECT_RUNS_DIR="${PROJECT_RUNS_DIR:-runs}"
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# Node naming on Ada is sparse (gnode013-015, 018, 024, 039 do not exist), so a
# --exclude range errors out with "Invalid node name specified". Guard at
# runtime instead: vLLM needs compute capability >= 7.0, and the 1080 Ti nodes
# are 6.1.
if [ ! -d "$HF_HOME" ]; then
    echo "FATAL: $HF_HOME not visible from $(hostname)."
    echo "The Ada user guide lists /share1 as master-node only. If that is true"
    echo "here, weights must live somewhere both nodes see (\$HOME)."
    exit 1
fi

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "compute capability: $CAP"
if [ "${CAP%%.*}" -lt 7 ]; then
    echo "FATAL: got compute_cap $CAP (GTX 1080 Ti). vLLM requires >= 7.0."
    echo "Resubmit; SLURM will usually land you on a 2080 Ti node."
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
