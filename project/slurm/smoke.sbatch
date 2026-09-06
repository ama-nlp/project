#!/bin/bash
# P0 smoke on Ada: every model in PROJECT_MODELS over 4 problems each.
# Proves transformers loads the model, the chat template emits <think>, the
# sandbox survives SLURM's cgroup limits, and traces validate.
#
#   sbatch slurm/smoke.sbatch                            # 0.6B then 8B
#   PROJECT_MODELS=Qwen/Qwen3-0.6B sbatch slurm/smoke.sbatch   # just the cheap one
#
#SBATCH --job-name=project-smoke
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=logs/smoke-%j.out
#
# NOTE: logs/ must already exist. SLURM opens --output before the script runs,
# so the mkdir below is too late; without the directory the job dies at launch
# with no log at all. logs/.gitkeep keeps it in the repo.
#SBATCH --error=logs/smoke-%j.err
#
# Why these numbers:
#   --gres=gpu:2   Qwen3-8B fp16 is ~16.4 GB and these cards are 11 GB, so it
#                  does NOT fit on one. device_map="auto" (backends.py:116)
#                  shards it across two. The 0.6B ignores the second card.
#   -c 10          Ada policy is at most 10 CPUs per GPU; jobs breaking the
#                  1:10 ratio may be killed without warning. Two GPUs would
#                  allow 20, but the sandbox does not need them.
#   --mem=30G      MaxMemPerCPU is 3000 MB, so 10 CPUs caps at 30 GB.
#   --time=6h      The 0.6B alone took 29 min on a 1080 Ti (fp16 is crippled on
#                  sm_61). The 8B on the same card is the tail risk here.
#   no module load Ada has no cuda modules; torch ships its own runtime.

set -euo pipefail
mkdir -p logs

MODELS="${PROJECT_MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen3-8B}"

# Storage. Weights come from /share1, results go to $HOME -- see
# slurm/stage_model.sh for why, and scripts/probe_ada.sh for the measurements.
#   /share1/$USER/models   master weight store, 100 GB, login node only
#   /scratch/$USER/models  node-local copy, staged by scp at job start
#   $HOME                  30 GB: code, venv, and runs/ -- NOT weights
# HF_HOME is only the fallback for PROJECT_MODEL_SHARE=0 (resolve the repo id
# over the compute node's network instead of staging).
if [ "${PROJECT_HF_SCRATCH:-0}" = "1" ]; then
    export HF_HOME="/ssd_scratch/$USER/hf"
    mkdir -p "$HF_HOME"
else
    export HF_HOME="${HF_HOME:-$HOME/hf}"
fi
source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"   # staging happens per model, below
export UV_PYTHON_PREFERENCE=managed
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-10}"  # sandbox workers = allocated cores
export PROJECT_RUNS_DIR="${PROJECT_RUNS_DIR:-runs}"
export TOKENIZERS_PARALLELISM=false
# DynamicCache grows by reallocating every decode step, which fragments the
# allocator badly -- a failed run showed 3.18 GiB reserved-but-unallocated,
# 30% of the card. expandable_segments lets those blocks be resized in place.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

# KV cache is micro_batch x (prompt + max_tokens) x bytes/token, and
# bytes/token differs per model: 0.109 MiB for the 0.6B (2 x 28 layers x 8
# kv_heads x 128 head_dim x 2 bytes), 0.141 MiB for the 8B. The two knobs
# multiply -- 4 x 16384 was 7.2 GiB on an 11 GB card and OOM'd -- and the 8B
# also has ~16.4 GB of weights eating the budget first. So the batch is set per
# model, and _preflight() in backends.py refuses anything that will not fit
# rather than discovering it twenty minutes in.
FAILED=()
for REPO in $MODELS; do
    NAME=$(basename "$REPO")
    case "$NAME" in
        *-8B|*-14B) MB=2 ;;   # weights dominate; leave room for the cache
        *)          MB=4 ;;
    esac
    MB="${PROJECT_MICRO_BATCH:-$MB}"

    echo
    echo "=============================================================="
    echo "smoke: $REPO  (micro_batch $MB)"
    echo "=============================================================="

    MODEL=$(stage_model "$REPO") || { FAILED+=("$NAME: staging"); continue; }
    echo "model: $MODEL"

    # An explicit run_id keeps the two models' traces apart under runs/ --
    # the default is a bare timestamp, which would be guesswork to attribute.
    uv run --group gpu project generate \
        --backend hf \
        --model "$MODEL" \
        --dtype float16 \
        --micro_batch "$MB" \
        --arm C \
        --n 4 \
        --max_tokens "${PROJECT_MAX_TOKENS:-4096}" \
        --run_id "smoke-${SLURM_JOB_ID}-${NAME}" \
        --enable_thinking "${PROJECT_ENABLE_THINKING:-True}" \
        --seed 0 \
        || { FAILED+=("$NAME: generate"); continue; }

    echo "--- $NAME OK -> runs/smoke-${SLURM_JOB_ID}-${NAME}/"
done

echo
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "=== FAILED: ${FAILED[*]} ==="
    exit 1
fi
echo "=== all models OK ==="
