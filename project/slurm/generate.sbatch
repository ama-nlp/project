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
#SBATCH --error=logs/gen-%j.err
#
# Ada's cards are 11 GB (2080 Ti). Model sizes in fp16:
#   Qwen3-4B   ~8 GB   -> 1 GPU,  -c 10, --mem=30G
#   Qwen3-8B   ~16 GB  -> 2 GPUs, -c 20, --mem=60G   <- this default
#   Qwen3-14B  ~28 GB  -> 4 GPUs, -c 40, --mem=120G
# transformers shards these with device_map="auto"; no tensor-parallel setting.
# Sharding is pipeline-style, so GPUs run sequentially rather than in parallel -
# more GPUs buys capacity, not speed. PROJECT_MICRO_BATCH buys speed, until the KV
# cache runs out of VRAM (~1.2 GB per sequence for 8B at 8k context).
#
# QoS medium: max 4 GPUs and 4 running jobs per user, 4-day wall clock.
# 8B on 2 GPUs means two arms can run in parallel.

set -euo pipefail
mkdir -p logs

ARM="${1:-C}"
PARAPHRASE="${2:-0}"
MODEL="${PROJECT_MODEL:-Qwen/Qwen3-8B}"

export HF_HOME="${HF_HOME:-/share1/$USER/models}"
export PROJECT_DATA_DIR="${PROJECT_DATA_DIR:-/share1/$USER/datasets}"
export HF_HUB_OFFLINE=1
export UV_PYTHON_PREFERENCE=managed
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-10}"
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

# One run_id per arm x paraphrase keeps cells in separate files inside one run
# directory, which is what the P5 analysis expects.
uv run --group gpu project generate \
    --backend hf \
    --model "$MODEL" \
    --dtype float16 \
    --micro_batch "${PROJECT_MICRO_BATCH:-4}" \
    --arm "$ARM" \
    --paraphrase_id "$PARAPHRASE" \
    --max_tokens 8192 \
    --seed 0 \
    --run_id "${PROJECT_RUN_ID:-$SLURM_JOB_ID}"
