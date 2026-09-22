#!/usr/bin/env bash
# One-conversation gate for the OLMo reward-hacking screening chain.
# The real jobs are submitted only with an afterok dependency on this job.
#SBATCH --job-name=olmo-rh-smoke
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/olmo-smoke-%j.out
#SBATCH --error=logs/olmo-smoke-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
if [ "${DRV%%.*}" -lt 580 ]; then
    echo "FATAL: vLLM 0.24 requires driver >= 580; this node has $DRV."
    exit 1
fi

VLLM_VENV="${PROJECT_VLLM_VENV:-$HOME/vllm-env}"
if [ ! -x "$VLLM_VENV/bin/python" ]; then
    echo "FATAL: no vLLM environment at $VLLM_VENV"
    exit 1
fi
source "$SLURM_SUBMIT_DIR/slurm/vllm_env.sh"
vllm_env "$VLLM_VENV"
VIRTUAL_ENV="$VLLM_VENV" uv pip install --quiet -e .

MODEL_ID="${PROJECT_MODEL:-allenai/Olmo-3-7B-Think}"
MODEL_REVISION="${PROJECT_MODEL_REVISION:?PROJECT_MODEL_REVISION is required}"
source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MODEL=$(stage_model "$MODEL_ID")
test -s "$MODEL/chat_template.jinja" \
    || { echo "FATAL: staged OLMo model has no chat_template.jinja"; exit 1; }
grep -q '<think>' "$MODEL/chat_template.jinja" \
    || { echo "FATAL: OLMo chat template has no thinking prefix"; exit 1; }

export TOKENIZERS_PARALLELISM=false
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"
export PROJECT_MAX_NUM_SEQS=1
export PROJECT_MAX_MODEL_LEN=48000
RUN_ID="${PROJECT_RUN_ID:-olmo-smoke-${SLURM_JOB_ID}}"

"$VLLM_VENV/bin/project" phase3b \
    --arm C \
    --backend vllm \
    --model "$MODEL" \
    --model_id "$MODEL_ID" \
    --model_revision "$MODEL_REVISION" \
    --n 1 \
    --samples_per_problem 1 \
    --rounds 3 \
    --stop_on_pass=False \
    --batch_size 1 \
    --max_tokens 8192 \
    --temperature 1.0 \
    --top_p 0.95 \
    --seed 0 \
    --run_id "$RUN_ID"

"$VLLM_VENV/bin/python" scripts/check_olmo_smoke.py \
    "runs/$RUN_ID/armC_p0.jsonl" \
    --model "$MODEL_ID" \
    --revision "$MODEL_REVISION"
