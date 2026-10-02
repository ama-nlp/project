#!/usr/bin/env bash
# Hint environment (project.hints) on OLMo-3.1-32B-Think, 4-bit.
#
# Same design as the OLMo-3-7B-Think pilot (runs/hints-olmo-pilot, job 2712980):
# 100 MMLU-Pro questions x arms A/B/C x hints none/sycophancy/metadata/grader/
# unethical = 1,500 rows, temperature 0.6, seed 0. One deliberate change: the
# thinking budget is 8192, not 4096. At 4096, 657 of the 7B's 1,500 answers hit
# the budget and about two-thirds of hint-followers were forced to answer, which
# cuts off reasoning exactly where an admission of the hint would appear.
# Override with PROJECT_THINK_BUDGET=4096 for a like-for-like 7B comparison.
#
# Why this model: arXiv 2603.26410 measured the same hint paradigm on 12 open
# reasoning models. OLMo-3-7B-Think mentioned the hint in its thinking but not
# its answer 58.9% of the time; OLMo-3.1-32B-Think did so 88.3% of the time. The
# 32B is the within-family scale comparison for the 7B pilot.
#
# Weights: kaitchup/Olmo-3.1-32B-Think-awq-w4a16-asym, pinned below. 18.3 GB,
# compressed-tensors W4A16, asymmetric, group size 128, lm_head unquantized.
# On sm_75 vLLM 0.24 serves it with the Marlin W4A16 kernel (minimum capability
# 75, zero points supported), falling back to its Triton W4A16 kernel. It must
# be on /share1 first:
#
#   PROJECT_MODELS="kaitchup/Olmo-3.1-32B-Think-awq-w4a16-asym" \
#     PROJECT_MODEL_REVISION=7ac9bc13bd2b09017cec1768f5507546064e146f \
#     sbatch slurm/download_model.sbatch
#   sbatch --nodelist=gnode061 --dependency=afterok:<download job> \
#     --kill-on-invalid-dep=yes slurm/olmo32_hints.sbatch
#
# Resumable: project.hints skips rows already on disk. To continue a run that
# was killed, resubmit with PROJECT_RUN_ID=<its run id>.
#
# Overrides: PROJECT_HINTS_N, PROJECT_THINK_BUDGET, PROJECT_MAX_NUM_SEQS,
# PROJECT_RUN_ID. Extra flags pass through to `project hints`.
#SBATCH --job-name=olmo32-hints
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/olmo32-hints-%j.out
#SBATCH --error=logs/olmo32-hints-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"

MODEL_ID="kaitchup/Olmo-3.1-32B-Think-awq-w4a16-asym"
MODEL_REVISION="7ac9bc13bd2b09017cec1768f5507546064e146f"
N="${PROJECT_HINTS_N:-100}"
THINK_BUDGET="${PROJECT_THINK_BUDGET:-8192}"
RUN_ID="${PROJECT_RUN_ID:-olmo32-hints-${SLURM_JOB_ID}}"
OUT="runs/$RUN_ID/hints.jsonl"

nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if [ "${DRV%%.*}" -lt 580 ]; then
    echo "FATAL: vLLM 0.24 requires driver >= 580; this node has $DRV."
    echo "Resubmit on a known-good node such as gnode061, 062, 070, or 087."
    exit 1
fi

[ -s data/hints_mmlupro.jsonl ] || {
    echo "FATAL: data/hints_mmlupro.jsonl missing. Build it with scripts/make_hint_set.py."
    exit 1
}

VLLM_VENV="${PROJECT_VLLM_VENV:-$HOME/vllm-env}"
source "$SLURM_SUBMIT_DIR/slurm/vllm_env.sh"
vllm_env "$VLLM_VENV"
VIRTUAL_ENV="$VLLM_VENV" uv pip install --quiet -e .

source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MODEL=$(stage_model "$MODEL_ID")

# Refuse an unquantized or wrong checkpoint before paying for engine start-up:
# 32B in fp16 is 64 GB and cannot fit four 11 GB cards.
python3 - "$MODEL" <<'PY'
import json, sys
c = json.load(open(f"{sys.argv[1]}/config.json"))
q = c.get("quantization_config") or {}
assert "Olmo3ForCausalLM" in (c.get("architectures") or []), c.get("architectures")
assert q.get("quant_method") == "compressed-tensors", q.get("quant_method")
print(f"checkpoint ok: {c['num_hidden_layers']} layers, {c['num_key_value_heads']} kv heads, "
      f"quant {q['quant_method']}")
PY

export TOKENIZERS_PARALLELISM=false
export PROJECT_MAX_NUM_SEQS="${PROJECT_MAX_NUM_SEQS:-32}"
echo "run $RUN_ID | model $MODEL_ID@${MODEL_REVISION:0:7} | n $N | think_budget $THINK_BUDGET"

run_hints () {
    "$VLLM_VENV/bin/project" hints \
        --model "$MODEL" \
        --model_id "$MODEL_ID" \
        --model_revision "$MODEL_REVISION" \
        --think_budget "$THINK_BUDGET" \
        --run_id "$RUN_ID" \
        "$@"
}

# Smoke: the first question of the real run, 15 rows. Same run id and seeds, so
# these rows are the full run's first rows, not throwaway work. Fails the job in
# minutes if the 4-bit model loads but produces unusable output.
BEFORE=$( [ -s "$OUT" ] && wc -l < "$OUT" || echo 0 )
T0=$(date +%s)
run_hints --n 1 "${@}"
T1=$(date +%s)
python3 - "$OUT" "$BEFORE" "$((T1 - T0))" "$N" <<'PY'
import json, sys
path, before, secs, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = [json.loads(l) for l in open(path) if l.strip()]
new = rows[before:]
if not new:
    print(f"smoke: resuming, {len(rows)} rows already on disk")
    sys.exit(0)
answered = sum(r["answer"] is not None for r in new)
with_cot = sum(bool(r["cot"].strip()) for r in new)
tokens = sum(r["n_think_tokens"] for r in new)
print(f"smoke: {len(new)} rows in {secs}s including engine start-up | "
      f"answered {answered}/{len(new)} | non-empty CoT {with_cot}/{len(new)} | "
      f"forced {sum(bool(r['forced']) for r in new)} | mean thinking tokens "
      f"{tokens / len(new):.0f}")
if answered < 0.8 * len(new) or with_cot < len(new):
    print("FATAL: the model loads but its output does not parse. Inspect the smoke rows:")
    print(f"  {path}")
    sys.exit(3)
remaining = n * 15 - len(rows)
print(f"smoke: {tokens / secs:.0f} thinking tokens/s, a lower bound because it includes "
      f"start-up and a single 15-row batch. Naive projection for the remaining "
      f"{remaining} rows: {remaining * secs / len(new) / 3600:.1f} h, an overestimate "
      f"since the full run batches 150 rows at a time.")
PY

run_hints --n "$N" "${@}"
"$VLLM_VENV/bin/project" hints-summary "$OUT"
