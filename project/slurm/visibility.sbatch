#!/usr/bin/env bash
# Bounded Arm-C visibility diagnostic.
# Array cells: 0 correct/hidden, 1 correct/visible,
#              2 conflicting/hidden, 3 conflicting/visible.
# The neutral primary diagnostic runs all four cells. The strategy-salience
# positive-source diagnostic should run only cell 3 and must not be pooled.
#SBATCH --job-name=visibility-c
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/visibility-%A_%a.out
#SBATCH --error=logs/visibility-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"

SUITES=(correct correct conflicting conflicting)
VISIBILITY=(false true false true)
CELL_NAMES=(correct-hidden correct-visible conflicting-hidden conflicting-visible)
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$INDEX" -lt 0 ] || [ "$INDEX" -ge "${#SUITES[@]}" ]; then
    echo "FATAL: array index $INDEX does not map to a visibility cell"
    exit 2
fi

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

DATASET="data/leetcode_test_medhard.jsonl"
VARIANTS="data/impossible_v1.jsonl"
if [ ! -s "$DATASET" ]; then
    echo "FATAL: missing $DATASET; run scripts/fetch_upstream.sh on the login node"
    exit 1
fi
if [ ! -s "$VARIANTS" ]; then
    "$VLLM_VENV/bin/python" scripts/make_impossible_set.py --out "$VARIANTS"
fi

source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MODEL=$(stage_model "${PROJECT_MODEL:-Qwen/Qwen3-8B}")
export TOKENIZERS_PARALLELISM=false
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-40}"

PROMPT_VARIANT="${PROJECT_VISIBILITY_PROMPT:-neutral}"
RUN_STEM="${PROJECT_RUN_ID:-visibility-${PROMPT_VARIANT}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
RUN_ID="${RUN_STEM}-${CELL_NAMES[$INDEX]}"

echo "visibility diagnostic cell ${CELL_NAMES[$INDEX]}; prompt $PROMPT_VARIANT; model $MODEL"
"$VLLM_VENV/bin/project" visibility \
    --suite_condition "${SUITES[$INDEX]}" \
    --tests_visible "${VISIBILITY[$INDEX]}" \
    --prompt_variant "$PROMPT_VARIANT" \
    --variants "$VARIANTS" \
    --dataset "$DATASET" \
    --backend vllm \
    --model "$MODEL" \
    --samples_per_problem 4 \
    --rounds 3 \
    --batch_size "${PROJECT_BATCH_SIZE:-16}" \
    --max_tokens "${PROJECT_MAX_TOKENS:-30000}" \
    --temperature "${PROJECT_TEMPERATURE:-1.0}" \
    --top_p "${PROJECT_TOP_P:-0.95}" \
    --seed "${PROJECT_SEED:-0}" \
    --run_id "$RUN_ID"
