#!/bin/bash
# One arm x one paraphrase over the full frozen set.
#
#   sbatch slurm/generate.sbatch C 0
#   PROJECT_MODEL=Qwen/Qwen3-4B sbatch --gres=gpu:1 -c 10 --mem=30G \
#       slurm/generate.sbatch C 0
#
#SBATCH --job-name=project-gen
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/gen-%j.out
#
# NOTE: logs/ must already exist. SLURM opens --output before the script runs,
# so the mkdir below is too late; without the directory the job dies at launch
# with no log at all. logs/.gitkeep keeps it in the repo.
#SBATCH --error=logs/gen-%j.err
#
# Ada's cards are 11 GB (2080 Ti). Model sizes in fp16:
#   Qwen3-4B   ~8 GB   -> 1 GPU,  -c 10, --mem=30G
#   Qwen3-8B   ~16 GB  -> 2 GPUs, -c 20, --mem=60G   <- this default
#   Qwen3-14B  ~28 GB  -> 4 GPUs, -c 40, --mem=120G
# transformers shards these with device_map="auto"; no tensor-parallel setting.
# Sharding is pipeline-style, so GPUs run sequentially rather than in parallel -
# more GPUs buys capacity, not speed. PROJECT_MICRO_BATCH buys speed, until the KV
# cache runs out of VRAM (~1.2 GB per sequence for 8B at 8k, ~2.4 GB at 16k -
# which is why micro_batch drops to 2 now that max_tokens is 16384).
#
# QoS medium: max 4 GPUs and 4 running jobs per user, 4-day wall clock.
# 8B on 2 GPUs means two arms can run in parallel.

set -euo pipefail
mkdir -p logs

ARM="${1:-C}"
PARAPHRASE="${2:-0}"
MODEL="${PROJECT_MODEL:-Qwen/Qwen3-8B}"

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
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-10}"
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

# One run_id per arm x paraphrase keeps cells in separate files inside one run
# directory, which is what the P5 analysis expects.
uv run --group gpu project generate \
    --backend hf \
    --model "$MODEL" \
    --dtype float16 \
    --micro_batch "${PROJECT_MICRO_BATCH:-2}" \
    --arm "$ARM" \
    --paraphrase_id "$PARAPHRASE" \
    --max_tokens 16384 \
    --seed 0 \
    --run_id "${PROJECT_RUN_ID:-$SLURM_JOB_ID}"
