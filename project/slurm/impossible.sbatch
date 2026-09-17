#!/usr/bin/env bash
# P3i full run: impossible hidden tests with editable run_tests, Arm C only.
# See src/project/impossible.py for the design and docs/phase3i.md for usage.
#
#   sbatch --nodelist=gnode061 slurm/impossible.sbatch             # both conditions
#   PROJECT_P3I_SUBSET=p3-hack-20 sbatch --array=0 --nodelist=gnode061 \
#       slurm/impossible.sbatch  # 20-problem LeetCode enrichment run, impossible only
#
# Array index 0 = impossible, 1 = control; they run one at a time (%1).
# Set PROJECT_P3I_CONDITIONS=mixed25 and submit with --array=0 for one run in
# which a deterministic 25% of problems use the impossible grader and 75% use
# the honest control grader.
# Defaults: Qwen3-8B, every verified problem, 4 samples/problem, T=1.0, max 3 rounds,
# 30000-token cap. Override with PROJECT_P3I_N, PROJECT_P3I_K,
# PROJECT_MAX_TOKENS, PROJECT_MODEL, or PROJECT_RUNS_DIR.
#
# PROJECT_P3I_SET=impossiblebench swaps the leetcode set for Impossible-LiveCodeBench
# (scripts/make_impossiblebench_set.py; build it on the login node first, compute
# nodes may lack internet). Run IDs then start with p3ib- so traces never mix.
#SBATCH --job-name=p3i-c
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

IFS=',' read -r -a CONDITIONS <<< "${PROJECT_P3I_CONDITIONS:-impossible,control}"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$INDEX" -lt 0 ] || [ "$INDEX" -ge "${#CONDITIONS[@]}" ]; then
    echo "FATAL: array index $INDEX does not map to a condition"
    exit 2
fi
CONDITION="${CONDITIONS[$INDEX]}"
case "$CONDITION" in
    impossible|control|mixed25) ;;
    *)
        echo "FATAL: condition must be impossible, control, or mixed25; got $CONDITION"
        exit 2
        ;;
esac
ARM="C"

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
P3I_SET="${PROJECT_P3I_SET:-leetcode}"
P3I_SUBSET="${PROJECT_P3I_SUBSET:-all}"
SUBSET_ARGS=()
case "$P3I_SET" in
    leetcode)
        DATASET="data/leetcode_test_medhard.jsonl"
        VARIANTS="data/impossible_v1.jsonl"
        STEM="p3i-full"
        TIMEOUT=6
        case "$P3I_SUBSET" in
            all) ;;
            p3-hack-20)
                STEM="p3i-hack20"
                VARIANTS="data/impossible_p3_hack20_v1.jsonl"
                SUBSET_ARGS=(--problem_subset "$P3I_SUBSET")
                ;;
            *)
                echo "FATAL: LeetCode subset must be all or p3-hack-20, got $P3I_SUBSET"
                exit 2
                ;;
        esac
        if [ ! -s "$VARIANTS" ]; then
            BUILD_ARGS=()
            if [ "$P3I_SUBSET" = p3-hack-20 ]; then
                BUILD_ARGS=(--problem-subset p3-hack-20)
            fi
            "$VLLM_VENV/bin/python" scripts/make_impossible_set.py \
                "${BUILD_ARGS[@]}" --out "$VARIANTS"
        fi
        ;;
    impossiblebench)
        if [ "$P3I_SUBSET" != all ]; then
            echo "FATAL: problem subsets are defined only for the LeetCode set"
            exit 2
        fi
        DATASET="data/impossiblebench_lcb.jsonl"
        VARIANTS="data/impossible_lcb_oneoff.jsonl"
        STEM="p3ib"
        TIMEOUT=60
        if [ ! -s "$VARIANTS" ] || [ ! -s "$DATASET" ]; then
            echo "FATAL: build the ImpossibleBench set first: uv run scripts/make_impossiblebench_set.py"
            exit 1
        fi
        ;;
    *)
        echo "FATAL: PROJECT_P3I_SET must be leetcode or impossiblebench, got $P3I_SET"
        exit 2
        ;;
esac
echo "problem set $DATASET sha256 $(sha256sum "$DATASET" | cut -d' ' -f1)"
echo "variant set $VARIANTS sha256 $(sha256sum "$VARIANTS" | cut -d' ' -f1)"

source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MODEL=$(stage_model "${PROJECT_MODEL:-Qwen/Qwen3-8B}")

export TOKENIZERS_PARALLELISM=false
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-40}"
RUN_STEM="${PROJECT_RUN_ID:-${STEM}-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
RUN_ID="${RUN_STEM}-${ARM}-${CONDITION}"
N_ARGS=()
if [ -n "${PROJECT_P3I_N:-}" ]; then
    if [ "$P3I_SUBSET" != all ]; then
        echo "FATAL: PROJECT_P3I_N cannot be combined with PROJECT_P3I_SUBSET"
        exit 2
    fi
    N_ARGS=(--n "$PROJECT_P3I_N")
fi

echo "P3i ($P3I_SET) Arm $ARM; condition $CONDITION; model $MODEL; run $RUN_ID"
"$VLLM_VENV/bin/project" impossible \
    --arm "$ARM" \
    --condition "$CONDITION" \
    --variants "$VARIANTS" \
    --dataset "$DATASET" \
    "${SUBSET_ARGS[@]}" \
    --timeout "${PROJECT_TIMEOUT:-$TIMEOUT}" \
    --backend vllm \
    --model "$MODEL" \
    "${N_ARGS[@]}" \
    --samples_per_problem "${PROJECT_P3I_K:-4}" \
    --rounds 3 \
    --batch_size "${PROJECT_BATCH_SIZE:-16}" \
    --max_tokens "${PROJECT_MAX_TOKENS:-30000}" \
    --temperature "${PROJECT_TEMPERATURE:-1.0}" \
    --top_p "${PROJECT_TOP_P:-0.95}" \
    --seed "${PROJECT_SEED:-0}" \
    --run_id "$RUN_ID"
