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
# PROJECT_INTERP_STEP=refusal runs I3c-refusal (scripts/interp_refusal.py).
# PROJECT_INTERP_STEP=steer with PROJECT_INTERP_KIND=watched|emotion runs I3c-watched /
# I3c-emotion (scripts/interp_steer.py). PROJECT_INTERP_STEP=distress runs I3d
# (scripts/interp_distress.py). PROJECT_INTERP_STEP=jlens runs I10, the targeted
# Jacobian lens (scripts/interp_jlens.py). PROJECT_INTERP_STEP=lora_check runs
# the E0 check of an unmerged adapter on the base (scripts/interp_lora.py).
# PROJECT_INTERP_STEP=couple runs E3 / E4 / E6 (scripts/interp_couple.py).
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
# gnode069 (job 16330) exposed only 3 of its 4 cards ("Failed to get device
# handle for GPU 1"); stop at once rather than load onto a degraded node.
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU")
if [ "$NGPU" -lt 4 ]; then
    echo "FATAL: only $NGPU GPUs visible on $(hostname); resubmit with --exclude=$(hostname)"; exit 1
fi
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

# --- E0: unmerged adapter on the 4-bit base vs the merged model ---------------
# PROJECT_INTERP_STEP=lora_check, with PROJECT_AISI_NAME the merged model,
# PROJECT_E_BASE the base-only build (aisi_rh_prepare.sbatch with
# PROJECT_AISI_BASE_ONLY=1) and PROJECT_E_ADAPTER / PROJECT_E_CHECKPOINT the
# matching AISI adapter (default: kl0.0 checkpoint-400).
if [ "${PROJECT_INTERP_STEP:-extract}" = "lora_check" ]; then
    BASE_MODEL=$(stage_model "${PROJECT_E_BASE:-olmo3.1-32b-instruct-sft-w4a16}")
    E_ADAPTER="${PROJECT_E_ADAPTER:-ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2}"
    E_CKPT="${PROJECT_E_CHECKPOINT:-checkpoint-400}"
    ADIR="$SCRATCH/aisi-rh-work/adapter/$(basename "$E_ADAPTER")"
    "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('$E_ADAPTER', local_dir='$ADIR', allow_patterns=['$E_CKPT/*'])"
    RC=0
    "$PY" scripts/interp_lora.py --check --merged "$MODEL" --base "$BASE_MODEL" \
        --adapter "$ADIR/$E_CKPT" --traces "$TRACES" --out "$OUT" \
        ${PROJECT_INTERP_LIMIT:+--limit "$PROJECT_INTERP_LIMIT"} \
        ${PROJECT_INTERP_CHUNK:+--chunk "$PROJECT_INTERP_CHUNK"} || RC=$?
    # PROJECT_HOLD_FOR=<file>: keep this job (and so ssh access to this node,
    # which pam_slurm_adopt grants only to users with a job here) alive until
    # another job has copied a model off this node's /scratch and touched the
    # file, or PROJECT_HOLD_MAX seconds pass. Used because the base build lives
    # only on gnode071, whose driver (570) is too old for vLLM 0.24 (job 25716).
    if [ -n "${PROJECT_HOLD_FOR:-}" ]; then
        END=$(( $(date +%s) + ${PROJECT_HOLD_MAX:-10800} ))
        echo "holding $(hostname) until $PROJECT_HOLD_FOR exists"
        while [ ! -e "$PROJECT_HOLD_FOR" ] && [ "$(date +%s)" -lt "$END" ]; do sleep 30; done
        [ -e "$PROJECT_HOLD_FOR" ] && echo "released: $PROJECT_HOLD_FOR" || echo "hold timed out"
    fi
    echo "results: $OUT"; exit "$RC"
fi

# --- E3 / E4 / E6: plan -> code coupling on the base + unmerged adapters -------
# PROJECT_INTERP_STEP=couple, PROJECT_INTERP_KIND=gate|write|curve,
# PROJECT_AISI_NAME the base build, traces exported from an E1 job's
# prefill-plan-eq / prefill-plan-exit runs (labels plan-eq, plan-exit).
# gate / write use kl0.0 checkpoint-400. curve uses "base" plus every
# PROJECT_E_CHECKPOINTS entry of the kl0.0 repo (e.g. "50 100 120 400"), and
# kl0.02 checkpoint-390 as a second endpoint. PROJECT_E_SCANS is passed to
# --e1-scans for the metric's validation against E1's sampled outputs.
if [ "${PROJECT_INTERP_STEP:-extract}" = "couple" ]; then
    KIND="${PROJECT_INTERP_KIND:?set PROJECT_INTERP_KIND=gate|write|curve}"
    K00=ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2
    K002=ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.02-seed2
    AROOT="$SCRATCH/aisi-rh-work/adapter"
    fetch () {  # repo checkpoint -> prints the local dir
        "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('$1', local_dir='$AROOT/$(basename "$1")', allow_patterns=['$2/*'])" >&2
        echo "$AROOT/$(basename "$1")/$2"
    }
    ARGS=()
    if [ "$KIND" = "curve" ]; then
        CK="base"
        for c in ${PROJECT_E_CHECKPOINTS:?set PROJECT_E_CHECKPOINTS}; do
            CK="$CK,k0_$c=$(fetch "$K00" "checkpoint-$c")"
        done
        CK="$CK,kl002=$(fetch "$K002" checkpoint-390)"
        ARGS+=(--checkpoints "$CK")
    else
        ARGS+=(--adapter "$(fetch "$K00" checkpoint-400)")
    fi
    "$PY" scripts/interp_couple.py --mode "$KIND" --model "$MODEL" --traces "$TRACES" \
        --out "$OUT" "${ARGS[@]}" ${PROJECT_E_SCANS:+--e1-scans "$PROJECT_E_SCANS"} \
        ${PROJECT_INTERP_N:+--n "$PROJECT_INTERP_N"} \
        ${PROJECT_INTERP_CHUNK:+--chunk "$PROJECT_INTERP_CHUNK"}
    echo "results: $OUT"; exit 0
fi

# --- I3a / I3b / I3c-refusal: prompt ablations, patching, refusal direction ----
case "${PROJECT_INTERP_STEP:-extract}" in
    ablate)
        "$PY" scripts/interp_patch.py --mode ablate --model "$MODEL" --traces "$TRACES" \
            --out "$OUT" ${PROJECT_INTERP_CHUNK:+--chunk "$PROJECT_INTERP_CHUNK"} \
            ${PROJECT_INTERP_RUN:+--run "$PROJECT_INTERP_RUN"} \
            ${PROJECT_INTERP_CONDITIONS:+--conditions "$PROJECT_INTERP_CONDITIONS"}
        echo "results: $OUT"; exit 0 ;;
    patch)
        "$PY" scripts/interp_patch.py --mode patch --model "$MODEL" --traces "$TRACES" \
            --out "$OUT" --ablate "${PROJECT_INTERP_ABLATE:?set PROJECT_INTERP_ABLATE}" \
            ${PROJECT_INTERP_PAIRS:+--pairs "$PROJECT_INTERP_PAIRS"} \
            ${PROJECT_INTERP_CORRUPTION:+--corruption "$PROJECT_INTERP_CORRUPTION"} \
            ${PROJECT_INTERP_RUN:+--run "$PROJECT_INTERP_RUN"} \
            ${PROJECT_INTERP_CHUNK:+--chunk "$PROJECT_INTERP_CHUNK"}
        echo "results: $OUT"; exit 0 ;;
    refusal)
        "$PY" scripts/interp_refusal.py --model "$MODEL" --traces "$TRACES" \
            --prompts data/refusal_prompts.json --out "$OUT" \
            ${PROJECT_INTERP_CHUNK:+--chunk "$PROJECT_INTERP_CHUNK"}
        echo "results: $OUT"; exit 0 ;;
    distress)
        "$PY" scripts/interp_distress.py --model "$MODEL" --traces "$TRACES" --out "$OUT" \
            --old-directions runs/interp-16980/directions.pt \
            ${PROJECT_INTERP_CHUNK:+--chunk "$PROJECT_INTERP_CHUNK"}
        echo "results: $OUT"; exit 0 ;;
    jlens)
        "$PY" scripts/interp_jlens.py --model "$MODEL" --traces "$TRACES" --out "$OUT" \
            ${PROJECT_INTERP_CHUNK:+--chunk "$PROJECT_INTERP_CHUNK"} \
            ${PROJECT_INTERP_CALIB_SAMPLES:+--calib-samples "$PROJECT_INTERP_CALIB_SAMPLES"}
        echo "results: $OUT"; exit 0 ;;
    steer)
        "$PY" scripts/interp_steer.py --kind "${PROJECT_INTERP_KIND:?set PROJECT_INTERP_KIND=watched|emotion}" \
            --model "$MODEL" --traces "$TRACES" --out "$OUT" \
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
