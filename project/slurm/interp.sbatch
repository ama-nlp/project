#!/usr/bin/env bash
# Phase 2 interpretability: activations of the 4-bit AISI OLMo-32B on stored
# traces, then linear probes (docs/results/interp-live.md).
#
#   1. export   the listed AISI eval runs to JSONL (scripts/interp_export.py,
#               in ~/aisi-eval-env, which has inspect_ai), unless
#               PROJECT_INTERP_TRACES points at an existing JSONL
#   2. check    teacher-force 8 traces and require a sane NLL on the sampled
#               completions: the first time this checkpoint runs outside vLLM
#   3. extract  residual stream at p0 / c16 / c64 / think_end / pre_hack for
#               every trace (scripts/interp_extract.py)
#   4. probe    problem-grouped CV probes per position x layer
#               (scripts/interp_probe.py)
#
# PROJECT_INTERP_STEP=opening runs I7 instead (scripts/interp_opening.py): the
# model's own probability of opening with the hack vs <thinking>, per prompt,
# with a logit lens. Prompt-only forward passes.
# PROJECT_INTERP_STEP=ablate|patch runs I3a / I3b (scripts/interp_patch.py).
#
#   sbatch --nodelist=gnode062 --export=ALL,PROJECT_AISI_NAME=olmo3.1-32b-rh-kl0.02-s2-ckpt390-w4a16,\
#     PROJECT_INTERP_RUNS="C022=runs/aisi-rh-eval-2717029/main A022=runs/aisi-rh-eval-2717039/arm-A ..." \
#     slurm/interp.sbatch
#
# Python: vllm-env already has torch 2.11 (cu128), transformers 5.17 and
# compressed-tensors 0.17, so it is used read-only; anything it lacks
# (accelerate, for device_map) goes into a small --target overlay at
# ~/interp-extra rather than into vllm-env. The model loads with
# device_map="auto" across the four cards; compressed-tensors dequantises the
# 4-bit weights on the fly, so ~17 GB of weights fit in 4 x 11 GB.
#SBATCH --job-name=interp
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=2080ti
#SBATCH --cpus-per-task=38
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=logs/interp-%j.out
#SBATCH --error=logs/interp-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
NAME="${PROJECT_AISI_NAME:?set PROJECT_AISI_NAME to a model dir under /share1/<user>/models}"
OUT="runs/interp-$SLURM_JOB_ID"
mkdir -p "$OUT" logs
SCRATCH="${PROJECT_SCRATCH:-/scratch/$USER}"
echo "job $SLURM_JOB_ID on $(hostname), model $NAME, out $OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
# torch 2.11's cu128 build has no Pascal (sm_61) kernels: job 10716 on a
# GTX 1080 Ti node failed with "no kernel image". Hence --constraint=2080ti.
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [ "${CAP%%.*}" -lt 7 ]; then
    echo "FATAL: compute capability $CAP; torch here needs >= 7.0 (use a 2080ti node)"; exit 1
fi

# --- traces ------------------------------------------------------------------
TRACES="${PROJECT_INTERP_TRACES:-$OUT/traces.jsonl}"
if [ -z "${PROJECT_INTERP_TRACES:-}" ]; then
    AISI_REPO="${PROJECT_AISI_REPO:-$HOME/aisi-rh}"
    # shellcheck disable=SC2086  # word-split LABEL=dir pairs
    PYTHONPATH="$AISI_REPO/rl-envs/src:$SLURM_SUBMIT_DIR/src" \
        "$HOME/aisi-eval-env/bin/python" scripts/interp_export.py --out "$TRACES" \
        ${PROJECT_INTERP_RUNS:?set PROJECT_INTERP_RUNS or PROJECT_INTERP_TRACES}
fi

# --- python ------------------------------------------------------------------
VLLM_VENV="${PROJECT_VLLM_VENV:-$HOME/vllm-env}"
source "$SLURM_SUBMIT_DIR/slurm/vllm_env.sh"
vllm_env "$VLLM_VENV"
EXTRA="$HOME/interp-extra"
export PYTHONPATH="$EXTRA${PYTHONPATH:+:$PYTHONPATH}"
PY="$VLLM_VENV/bin/python"
export PATH="$HOME/.local/bin:$PATH"  # uv lives here; batch jobs do not get it on PATH
if ! "$PY" -c "import accelerate" 2>/dev/null; then
    export UV_LINK_MODE=copy
    uv pip install --python "$PY" --target "$EXTRA" --no-deps "accelerate>=1.0"
fi
"$PY" -c "
import torch, transformers, compressed_tensors, accelerate
print('torch', torch.__version__, '| transformers', transformers.__version__,
      '| compressed-tensors', compressed_tensors.__version__, '| accelerate', accelerate.__version__,
      '| cuda devices', torch.cuda.device_count())"

# --- model -------------------------------------------------------------------
source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MODEL=$(stage_model "$NAME")

# --- I3a / I3b: prompt ablations and activation patching ----------------------
case "${PROJECT_INTERP_STEP:-extract}" in
    ablate)
        "$PY" scripts/interp_patch.py --mode ablate --model "$MODEL" --traces "$TRACES" \
            --out "$OUT" ${PROJECT_INTERP_CHUNK:+--chunk "$PROJECT_INTERP_CHUNK"}
        echo "results: $OUT"; exit 0 ;;
    patch)
        "$PY" scripts/interp_patch.py --mode patch --model "$MODEL" --traces "$TRACES" \
            --out "$OUT" --ablate "${PROJECT_INTERP_ABLATE:?set PROJECT_INTERP_ABLATE}" \
            ${PROJECT_INTERP_PAIRS:+--pairs "$PROJECT_INTERP_PAIRS"} \
            ${PROJECT_INTERP_CHUNK:+--chunk "$PROJECT_INTERP_CHUNK"}
        echo "results: $OUT"; exit 0 ;;
esac

# --- I7: first-token opening probabilities and logit lens ---------------------
if [ "${PROJECT_INTERP_STEP:-extract}" = "opening" ]; then
    T0=$(date +%s)
    "$PY" scripts/interp_opening.py --model "$MODEL" --traces "$TRACES" --out "$OUT" \
        ${PROJECT_INTERP_LAYERS:+--layers "$PROJECT_INTERP_LAYERS"} \
        ${PROJECT_INTERP_CHUNK:+--chunk "$PROJECT_INTERP_CHUNK"}
    echo "opening: $(( $(date +%s) - T0 )) s; results: $OUT"
    exit 0
fi

# --- check, extract, probe ----------------------------------------------------
"$PY" scripts/interp_extract.py --model "$MODEL" --traces "$TRACES" --out "$OUT/check" \
    --limit 8 --check-only \
    || { echo "FATAL: load check failed; see $OUT/check/load_check.json"; exit 1; }
T0=$(date +%s)
"$PY" scripts/interp_extract.py --model "$MODEL" --traces "$TRACES" --out "$OUT" \
    ${PROJECT_INTERP_LAYERS:+--layers "$PROJECT_INTERP_LAYERS"}
echo "extract: $(( $(date +%s) - T0 )) s"
"$PY" scripts/interp_probe.py --acts "$OUT" --label hack_attempt --json "$OUT/probe_hack_attempt.json"
"$PY" scripts/interp_probe.py --acts "$OUT" --label mention --json "$OUT/probe_mention.json" || true
echo "results: $OUT ($(du -sh "$OUT" | cut -f1))"
