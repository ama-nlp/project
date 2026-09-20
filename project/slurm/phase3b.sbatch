#!/usr/bin/env bash
# Phase 3b: repaired evaluator, neutral A/B/C frames, and the same
# stop-on-PASS Round 1 -> Round 2 -> Round 3 protocol as Phase 3.
#
#   sbatch --nodelist=gnode061 slurm/phase3b.sbatch
#   sbatch --array=0 --nodelist=gnode061 slurm/phase3b.sbatch  # Arm A only
#
# Defaults: Qwen3-8B, 20 problems, 4 samples/problem, T=1.0, max 3 rounds.
# Override with PROJECT_P3B_N, PROJECT_P3B_K, PROJECT_MAX_TOKENS,
# PROJECT_MODEL, or PROJECT_RUNS_DIR.
#SBATCH --job-name=p3b-pilot
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/p3b-%A_%a.out
#SBATCH --error=logs/p3b-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"

ARMS=(A B C)
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$INDEX" -lt 0 ] || [ "$INDEX" -ge "${#ARMS[@]}" ]; then
    echo "FATAL: array index $INDEX does not map to an A/B/C arm"
    exit 2
fi
ARM="${ARMS[$INDEX]}"

nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if [ "${DRV%%.*}" -lt 580 ]; then
    echo "FATAL: vLLM 0.24 requires driver >= 580; this node has $DRV."
    echo "Resubmit on a known-good node such as gnode061, 062, 070, or 087."
    exit 1
fi

VLLM_VENV="${PROJECT_VLLM_VENV:-$HOME/vllm-env}"
if [ ! -x "$VLLM_VENV/bin/python" ]; then
    echo "FATAL: no vLLM environment at $VLLM_VENV. Run slurm/probe_vllm.sbatch first."
    exit 1
fi
source "$SLURM_SUBMIT_DIR/slurm/vllm_env.sh"
vllm_env "$VLLM_VENV"
VIRTUAL_ENV="$VLLM_VENV" uv pip install --quiet -e .

source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MODEL_ID="${PROJECT_MODEL:-Qwen/Qwen3-8B}"
MODEL=$(stage_model "$MODEL_ID")
MODEL_META_ARGS=(--model_id "${PROJECT_MODEL_ID:-$MODEL_ID}")
if [ -n "${PROJECT_MODEL_REVISION:-}" ]; then
    MODEL_META_ARGS+=(--model_revision "$PROJECT_MODEL_REVISION")
fi

export TOKENIZERS_PARALLELISM=false
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-40}"
RUN_STEM="${PROJECT_RUN_ID:-p3b-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
RUN_ID="${RUN_STEM}-${ARM}"

echo "Phase 3b Arm $ARM; model $MODEL; run $RUN_ID"
"$VLLM_VENV/bin/project" phase3b \
    --arm "$ARM" \
    --backend vllm \
    --model "$MODEL" \
    "${MODEL_META_ARGS[@]}" \
    --n "${PROJECT_P3B_N:-20}" \
    --samples_per_problem "${PROJECT_P3B_K:-4}" \
    --rounds 3 \
    --batch_size "${PROJECT_BATCH_SIZE:-16}" \
    --max_tokens "${PROJECT_MAX_TOKENS:-16384}" \
    --temperature "${PROJECT_TEMPERATURE:-1.0}" \
    --top_p "${PROJECT_TOP_P:-0.95}" \
    --seed "${PROJECT_SEED:-0}" \
    --run_id "$RUN_ID"
