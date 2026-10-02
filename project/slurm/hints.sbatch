#!/usr/bin/env bash
# Hint environment pilot (project.hints) on OLMo-3-7B-Think.
#
#   PROJECT_MODEL_REVISION=d97e442d7cc678210054dbcc9b440894d62c89a4 \
#     sbatch --nodelist=gnode061 slurm/hints.sbatch
#
# Extra CLI flags pass through: sbatch slurm/hints.sbatch --n 30 --arms C
#SBATCH --job-name=hints
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/hints-%j.out
#SBATCH --error=logs/hints-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if [ "${DRV%%.*}" -lt 580 ]; then
    echo "FATAL: vLLM 0.24 requires driver >= 580; this node has $DRV."
    exit 1
fi

VLLM_VENV="${PROJECT_VLLM_VENV:-$HOME/vllm-env}"
source "$SLURM_SUBMIT_DIR/slurm/vllm_env.sh"
vllm_env "$VLLM_VENV"
VIRTUAL_ENV="$VLLM_VENV" uv pip install --quiet -e .

MODEL_ID="${PROJECT_MODEL:-allenai/Olmo-3-7B-Think}"
MODEL_REVISION="${PROJECT_MODEL_REVISION:?PROJECT_MODEL_REVISION is required}"
source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MODEL=$(stage_model "$MODEL_ID")

export TOKENIZERS_PARALLELISM=false
export PROJECT_MAX_NUM_SEQS="${PROJECT_MAX_NUM_SEQS:-32}"
RUN_ID="${PROJECT_RUN_ID:-hints-${SLURM_JOB_ID}}"

"$VLLM_VENV/bin/project" hints \
    --model "$MODEL" \
    --model_id "$MODEL_ID" \
    --model_revision "$MODEL_REVISION" \
    --run_id "$RUN_ID" \
    "$@"

"$VLLM_VENV/bin/project" hints-summary "runs/$RUN_ID/hints.jsonl"
