#!/usr/bin/env bash
# Submit the requested OLMo screening sequence:
#   smoke -> regular LeetCode Phase 3b Arm C -> ImpossibleBench impossible Arm C
#
# Usage (login node, from project/):
#   bash scripts/submit_olmo_screen.sh gnode065

set -euo pipefail

NODE="${1:?usage: $0 gnodeNNN}"
case "$NODE" in
    gnode[0-9][0-9][0-9]) ;;
    *) echo "FATAL: node must look like gnode065, got $NODE"; exit 2 ;;
esac

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"

MODEL_ID="allenai/Olmo-3-7B-Think"
MODEL_REVISION="d97e442d7cc678210054dbcc9b440894d62c89a4"
MODEL_DIR="/share1/$USER/models/Olmo-3-7B-Think"
test -s "$MODEL_DIR/config.json" || { echo "FATAL: missing $MODEL_DIR/config.json"; exit 1; }
test -s "$MODEL_DIR/chat_template.jinja" \
    || { echo "FATAL: missing $MODEL_DIR/chat_template.jinja"; exit 1; }
test -s data/leetcode_test_medhard.jsonl \
    || { echo "FATAL: missing regular LeetCode dataset"; exit 1; }
test -s data/impossiblebench_lcb.jsonl \
    || { echo "FATAL: build ImpossibleBench first with uv run scripts/make_impossiblebench_set.py"; exit 1; }
test -s data/impossible_lcb_oneoff.jsonl \
    || { echo "FATAL: missing ImpossibleBench one-off variants"; exit 1; }

STAMP=$(date -u +%Y%m%d-%H%M%S)
COMMON_EXPORT="ALL,PROJECT_MODEL=$MODEL_ID,PROJECT_MODEL_ID=$MODEL_ID,PROJECT_MODEL_REVISION=$MODEL_REVISION"

SMOKE_RAW=$(sbatch --parsable --nodelist="$NODE" \
    --export="$COMMON_EXPORT,PROJECT_RUN_ID=olmo-smoke-$STAMP" \
    slurm/olmo_smoke.sbatch)
SMOKE_JOB=${SMOKE_RAW%%;*}

LEETCODE_RAW=$(sbatch --parsable --nodelist="$NODE" --array=2 \
    --dependency="afterok:$SMOKE_JOB" \
    --export="$COMMON_EXPORT,PROJECT_RUN_ID=olmo-leetcode-$STAMP,PROJECT_P3B_N=${PROJECT_OLMO_LEETCODE_N:-20},PROJECT_P3B_K=${PROJECT_OLMO_LEETCODE_K:-4},PROJECT_BATCH_SIZE=2,PROJECT_MAX_NUM_SEQS=2,PROJECT_MAX_MODEL_LEN=20000,PROJECT_MAX_TOKENS=16384" \
    slurm/phase3b.sbatch)
LEETCODE_JOB=${LEETCODE_RAW%%;*}

IMPOSSIBLE_RAW=$(sbatch --parsable --nodelist="$NODE" --array=0 \
    --dependency="afterok:$LEETCODE_JOB" \
    --export="$COMMON_EXPORT,PROJECT_RUN_ID=olmo-impossiblebench-$STAMP,PROJECT_P3I_SET=impossiblebench,PROJECT_P3I_K=${PROJECT_OLMO_IMPOSSIBLE_K:-1},PROJECT_BATCH_SIZE=2,PROJECT_MAX_NUM_SEQS=2,PROJECT_MAX_MODEL_LEN=20000,PROJECT_MAX_TOKENS=16384" \
    slurm/impossible.sbatch)
IMPOSSIBLE_JOB=${IMPOSSIBLE_RAW%%;*}

echo "Submitted OLMo screening chain on $NODE:"
echo "  smoke:                   $SMOKE_JOB"
echo "  regular LeetCode Arm C:  $LEETCODE_JOB (afterok:$SMOKE_JOB)"
echo "  ImpossibleBench:         $IMPOSSIBLE_JOB (afterok:$LEETCODE_JOB)"
echo "  run stamp:               $STAMP"
