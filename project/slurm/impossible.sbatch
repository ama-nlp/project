#!/usr/bin/env bash
# P3i: impossible-test variant under the neutral A/B/C oversight frames.
# See src/project/impossible.py for the design and docs/phase3i.md for usage.
#
#   sbatch --nodelist=gnode061 slurm/impossible.sbatch             # Arm C, both conditions
#   PROJECT_ARM=A sbatch --nodelist=gnode061 slurm/impossible.sbatch
#   sbatch --array=0 --nodelist=gnode061 slurm/impossible.sbatch   # impossible only
#
# Array index 0 = impossible, 1 = control; they run one at a time (%1).
# Defaults: Qwen3-8B, 20 problems, 4 samples/problem, T=1.0, max 3 rounds,
# 30000-token cap. Override with PROJECT_ARM, PROJECT_P3I_N, PROJECT_P3I_K,
# PROJECT_MAX_TOKENS, PROJECT_MODEL, or PROJECT_RUNS_DIR.
#SBATCH --job-name=p3i-pilot
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1%1
#SBATCH --output=logs/p3i-%A_%a.out
#SBATCH --error=logs/p3i-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"

CONDITIONS=(impossible control)
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$INDEX" -lt 0 ] || [ "$INDEX" -ge "${#CONDITIONS[@]}" ]; then
    echo "FATAL: array index $INDEX does not map to a condition"
    exit 2
fi
CONDITION="${CONDITIONS[$INDEX]}"
ARM="${PROJECT_ARM:-C}"

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

# The variant set is gitignored (upstream test strings), so build it here if this
# checkout lacks it. Deterministic: every build of the same base is identical.
VARIANTS="data/impossible_v1.jsonl"
if [ ! -s "$VARIANTS" ]; then
    "$VLLM_VENV/bin/python" scripts/make_impossible_set.py --out "$VARIANTS"
fi
echo "variant set sha256 $(sha256sum "$VARIANTS" | cut -d' ' -f1)"

source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MODEL=$(stage_model "${PROJECT_MODEL:-Qwen/Qwen3-8B}")

export TOKENIZERS_PARALLELISM=false
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-40}"
RUN_STEM="${PROJECT_RUN_ID:-p3i-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
RUN_ID="${RUN_STEM}-${ARM}-${CONDITION}"

echo "P3i Arm $ARM; condition $CONDITION; model $MODEL; run $RUN_ID"
"$VLLM_VENV/bin/project" impossible \
    --arm "$ARM" \
    --condition "$CONDITION" \
    --variants "$VARIANTS" \
    --backend vllm \
    --model "$MODEL" \
    --n "${PROJECT_P3I_N:-20}" \
    --samples_per_problem "${PROJECT_P3I_K:-4}" \
    --rounds 3 \
    --batch_size "${PROJECT_BATCH_SIZE:-16}" \
    --max_tokens "${PROJECT_MAX_TOKENS:-30000}" \
    --temperature "${PROJECT_TEMPERATURE:-1.0}" \
    --top_p "${PROJECT_TOP_P:-0.95}" \
    --seed "${PROJECT_SEED:-0}" \
    --run_id "$RUN_ID"
