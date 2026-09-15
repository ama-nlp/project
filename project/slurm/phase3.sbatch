#!/usr/bin/env bash
# Phase 3 pilot: array indices 0/1/2 run Arms A/B/C respectively. Each sample
# follows the same shown-grader Round 1 -> FAIL -> Round 2 -> FAIL -> Round 3
# conversation and stops as soon as it receives PASS.
#
#   sbatch --nodelist=gnode061 slurm/phase3.sbatch
#   sbatch --array=0 --nodelist=gnode061 slurm/phase3.sbatch  # Arm A only
#
# Defaults: Qwen3-8B, 20 problems, 4 samples/problem, T=1.0, max 3 rounds.
# Override with PROJECT_P3_N, PROJECT_P3_K, PROJECT_MAX_TOKENS, PROJECT_MODEL,
# or PROJECT_RUNS_DIR. The three-round protocol is fixed. PROJECT_P3_SUFFIX
# names a prompts.PILOT_SUFFIXES entry appended to every arm (run_id and
# hint_variant carry the name so the traces never mix with the baseline):
#
#   PROJECT_P3_SUFFIX=penalty sbatch --nodelist=gnode070 slurm/phase3.sbatch
#SBATCH --job-name=p3-pilot
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/p3-%A_%a.out
#SBATCH --error=logs/p3-%A_%a.err

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

# Phase 2's working portable environment lives on home storage. Do not default
# to /scratch here: it is node-local, purged, and only valid on the node where
# the venv was built. PROJECT_VLLM_VENV remains available for an explicit
# node-local environment.
VLLM_VENV="${PROJECT_VLLM_VENV:-$HOME/vllm-env}"
if [ ! -x "$VLLM_VENV/bin/python" ]; then
    echo "FATAL: no vLLM environment at $VLLM_VENV. Run slurm/probe_vllm.sbatch first."
    exit 1
fi
source "$SLURM_SUBMIT_DIR/slurm/vllm_env.sh"
vllm_env "$VLLM_VENV"
VIRTUAL_ENV="$VLLM_VENV" uv pip install --quiet -e .

source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MODEL=$(stage_model "${PROJECT_MODEL:-Qwen/Qwen3-8B}")

export TOKENIZERS_PARALLELISM=false
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-40}"
RUN_STEM="${PROJECT_RUN_ID:-p3-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
SUFFIX="${PROJECT_P3_SUFFIX:-}"
RUN_ID="${RUN_STEM}${SUFFIX:+-$SUFFIX}-${ARM}"

echo "Phase 3 Arm $ARM; model $MODEL; run $RUN_ID"
"$VLLM_VENV/bin/project" phase3 \
    --arm "$ARM" \
    --backend vllm \
    --model "$MODEL" \
    --n "${PROJECT_P3_N:-20}" \
    --samples_per_problem "${PROJECT_P3_K:-4}" \
    --rounds 3 \
    --batch_size "${PROJECT_BATCH_SIZE:-16}" \
    --max_tokens "${PROJECT_MAX_TOKENS:-16384}" \
    --temperature "${PROJECT_TEMPERATURE:-1.0}" \
    --top_p "${PROJECT_TOP_P:-0.95}" \
    --seed "${PROJECT_SEED:-0}" \
    ${SUFFIX:+--suffix "$SUFFIX"} \
    --run_id "$RUN_ID"
