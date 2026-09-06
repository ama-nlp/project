#!/bin/bash
# P0 steps 4+5: gate check and hand-read pack for one arm's traces.
#
# Chain it to the generate job so the analysis is ready when that finishes:
#
#   sbatch --dependency=afterok:<GENJOBID> slurm/gate.sbatch <GENJOBID>
#
# afterok means it only runs if generate exits 0; a crashed run leaves no
# misleading report behind. Use afterany if you want a report even then.
#
# No --gres: this is pure CPU. download_model.sbatch already establishes that
# u22 accepts a GPU-less job, so this does not spend one of the four QoS slots.
#
#SBATCH --job-name=project-gate
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH -n 4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gate-%j.out
#
# logs/ must exist before SLURM opens these; see generate.sbatch.
#SBATCH --error=logs/gate-%j.err

set -euo pipefail
mkdir -p logs

RUN_ID="${1:?usage: sbatch slurm/gate.sbatch <RUN_ID> [ARM] [PARAPHRASE]}"
ARM="${2:-C}"
PARAPHRASE="${3:-0}"

cd "$SLURM_SUBMIT_DIR"
TRACES="runs/${RUN_ID}/arm${ARM}_p${PARAPHRASE}.jsonl"
[ -s "$TRACES" ] || { echo "FATAL: no traces at $TRACES"; exit 1; }

OUT="runs/${RUN_ID}"
echo "traces: $TRACES  ($(wc -l < "$TRACES") records)"
echo

# The gate itself. check_gate.py exits non-zero if any record is missing a
# cot, a program or a verdict, so this job's state is the gate's verdict --
# sacct -j <thisjob> tells you PASS/FAIL without reading anything.
set +e
uv run scripts/check_gate.py "$TRACES" | tee "$OUT/gate_arm${ARM}_p${PARAPHRASE}.txt"
GATE=${PIPESTATUS[0]}
set -e

# Step 5's pack, written regardless: if the gate failed, the sampled traces are
# exactly what you need to see why.
uv run scripts/check_gate.py "$TRACES" --sample 20 \
    > "$OUT/read_me_arm${ARM}_p${PARAPHRASE}.txt"

echo
echo "wrote $OUT/gate_arm${ARM}_p${PARAPHRASE}.txt"
echo "wrote $OUT/read_me_arm${ARM}_p${PARAPHRASE}.txt  <- read this by hand"
echo
if [ "$GATE" -ne 0 ]; then
    echo "=== P0 GATE FAILED - see the report above ==="
    exit 1
fi
echo "=== P0 gate passed. Step 5 is the hand-read; it is not automatable. ==="
