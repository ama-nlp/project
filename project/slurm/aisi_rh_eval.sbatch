#!/usr/bin/env bash
# AISI's CodeContests reward-hacking eval on our 4-bit AISI OLMo-3.1-32B.
#
# Goal: check that the merged + W4A16 model (scripts/prepare_aisi_rh_model.py)
# still reward-hacks at a rate comparable to what AISI report for
# reward-hacking-olmo3.1-32b-kl0.02-seed2 before we build the A/B/C framings on
# top of it. Uses AISI's own task and scorer unchanged (repo pinned below):
# "dont_hack" system prompt with "sutl" hints, all three hacks enabled, local
# sandbox, pytest with a 10 s timeout, static hack detection.
#
#   sbatch --nodelist=gnode061 slurm/aisi_rh_eval.sbatch
#
# One-time setup on the login node (git works there; uv and pip builds do not):
#   git clone https://github.com/UKGovernmentBEIS/reward-hacking-misalignment ~/aisi-rh
#   (cd ~/aisi-rh && git checkout 169c3c7)
# The first run builds ~/aisi-eval-env; later runs reuse it.
#
# Overrides: PROJECT_AISI_N (samples, default 200), PROJECT_AISI_TEMP (1.0),
# PROJECT_AISI_VARIANT (dont_hack), PROJECT_MAX_MODEL_LEN (8192, as AISI serve).
#
# Monitoring arms: PROJECT_AISI_ARMS="A B" runs scripts/aisi_rh_arms.py once
# per arm on the same first N problems, instead of the plain eval. Arm C is the
# plain eval itself. PROJECT_AISI_GATE=<plain-eval run dir> makes the job refuse
# to start vLLM unless that run reward-hacked at least PROJECT_AISI_MIN_HACKS
# times (default 10 of 200): framings on a model that does not hack are moot.
# PROJECT_AISI_MIN_MENTION=<fraction> additionally requires that share of its
# hacked outputs to name the hack in <thinking> (AISI's keyword label, see
# scripts/aisi_rh_cot_scan.py): suppression cannot be measured without it.
#
# Another model: PROJECT_AISI_NAME=<dir under /share1/$USER/models>, e.g. the
# kl0.0 build olmo3.1-32b-rh-kl0.0-s2-ckpt400-w4a16 (slurm/aisi_rh_prepare.sbatch).
#
#   sbatch --nodelist=gnode048 --dependency=afterok:<plain job> \
#     --export=ALL,PROJECT_AISI_ARMS="A B",PROJECT_AISI_GATE=runs/aisi-rh-eval-<plain job> \
#     slurm/aisi_rh_eval.sbatch
#SBATCH --job-name=aisi-rh-eval
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=38
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/aisi-eval-%j.out
#SBATCH --error=logs/aisi-eval-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"

NAME="${PROJECT_AISI_NAME:-olmo3.1-32b-rh-kl0.02-s2-ckpt390-w4a16}"
N="${PROJECT_AISI_N:-200}"
TEMP="${PROJECT_AISI_TEMP:-1.0}"
VARIANT="${PROJECT_AISI_VARIANT:-dont_hack}"
MAX_LEN="${PROJECT_MAX_MODEL_LEN:-8192}"
AISI_REPO="${PROJECT_AISI_REPO:-$HOME/aisi-rh}"
AISI_COMMIT=169c3c76a02e51092b4023a8c7baba38f41e2800
SCRATCH="/scratch/$USER"
RUN_ID="${PROJECT_RUN_ID:-aisi-rh-eval-${SLURM_JOB_ID}}"
OUT="$PWD/runs/$RUN_ID"
PORT=8000
ARMS="${PROJECT_AISI_ARMS:-}"
mkdir -p "$OUT"


nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if [ "${DRV%%.*}" -lt 580 ]; then
    echo "FATAL: vLLM 0.24 requires driver >= 580; this node has $DRV."
    echo "Resubmit on a known-good node such as gnode061, 062, 070, or 087."
    exit 1
fi

[ "$(git -C "$AISI_REPO" rev-parse HEAD)" = "$AISI_COMMIT" ] \
    || { echo "FATAL: $AISI_REPO is not at $AISI_COMMIT"; exit 1; }

# --- disk space ---------------------------------------------------------------
# $HOME is a 30 GB quota already ~20 GB full (vllm-env alone is 12 GB), so only
# the eval venv (hardlinked from the uv cache) and the small result files go
# there. The model copy, the CodeContests download and sandbox temp files all
# go to node-local /scratch.
home_free_gb () {
    quota -w 2>/dev/null | awk '/home/ {print int(($3 - $2) / 1048576); exit}'
}
HOME_FREE=$(home_free_gb || true)
SCRATCH_FREE=$(df -BG --output=avail /scratch 2>/dev/null | tail -1 | tr -dc 0-9 || echo 0)
echo "free: \$HOME ${HOME_FREE:-unknown} GB of quota | /scratch ${SCRATCH_FREE} GB"
if [ -n "${HOME_FREE:-}" ] && [ "$HOME_FREE" -lt 3 ]; then
    echo "FATAL: under 3 GB left in the \$HOME quota; free some before running."
    exit 1
fi
if [ "${SCRATCH_FREE:-0}" -lt 40 ]; then
    echo "FATAL: need ~40 GB on /scratch (17 GB model + ~5 GB dataset + slack)."
    exit 1
fi
TMPDIR="$SCRATCH/tmp-$SLURM_JOB_ID"
cleanup () {
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null || true
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

# --- eval environment --------------------------------------------------------
# Built the way vllm-env was (probe_vllm.sbatch): uv on the compute node, with
# the shared uv cache in $HOME. Most of inspect-ai's dependencies (numpy,
# pydantic, httpx, openai, rich, fsspec, huggingface-hub) are already in that
# cache, so only ~100 MB is new (datasets, pyarrow, pandas, inspect-ai, pytest).
# A separate venv, so vllm-env is never touched. Hardlinks: the cache and the
# venv share the $HOME filesystem, so the venv costs almost no extra quota.
export UV_LINK_MODE=hardlink UV_PYTHON_PREFERENCE=managed
EVAL_ENV="$HOME/aisi-eval-env"
if ! "$EVAL_ENV/bin/python" -c "import inspect_ai, datasets, openai, pytest" 2>/dev/null; then
    rm -rf "$EVAL_ENV"
    uv venv -q --python 3.12 "$EVAL_ENV"
    T_ENV=$(date +%s)
    VIRTUAL_ENV="$EVAL_ENV" uv pip install \
        "inspect-ai>=0.3.0" "datasets>=2.0.0" huggingface_hub platformdirs pytest openai
    echo "eval env built in $(( $(date +%s) - T_ENV )) s; \$HOME free now $(home_free_gb || echo '?') GB"
fi
# rh_envs is used from source rather than installed, so nothing needs building.
export PYTHONPATH="$AISI_REPO/rl-envs/src:$SLURM_SUBMIT_DIR/src"  # rh_envs; project.prompts (stdlib only)
"$EVAL_ENV/bin/python" -c "
import inspect_ai, datasets, openai, pytest, rh_envs.codecontests_rh
from importlib.metadata import version
print('eval env:', ', '.join(f'{p} {version(p)}' for p in ('inspect-ai','datasets','openai','pytest')))"

# --- gate on the plain run, before paying for vLLM start-up ------------------
if [ -n "${PROJECT_AISI_GATE:-}" ]; then
    GATE_ARGS=(--gate-min-hacks "${PROJECT_AISI_MIN_HACKS:-10}")
    [ -n "${PROJECT_AISI_MIN_MENTION:-}" ] && GATE_ARGS+=(--gate-min-mention "$PROJECT_AISI_MIN_MENTION")
    "$EVAL_ENV/bin/python" scripts/aisi_rh_cot_scan.py "C=$PROJECT_AISI_GATE/main" "${GATE_ARGS[@]}" \
        || { echo "GATE: plain run $PROJECT_AISI_GATE does not qualify; stopping before vLLM"; exit 1; }
fi

# --- model ------------------------------------------------------------------
VLLM_VENV="${PROJECT_VLLM_VENV:-$HOME/vllm-env}"
source "$SLURM_SUBMIT_DIR/slurm/vllm_env.sh"
vllm_env "$VLLM_VENV"
source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MODEL=$(stage_model "$NAME")

# The saved config is in the flat pre-Transformers-5 format, but Transformers 5
# may re-nest it on load, which vLLM 0.24 cannot read. Decide at runtime, with
# the same rule as backends._flatten_olmo3_rope_parameters.
OVERRIDES=$("$VLLM_VENV/bin/python" - "$MODEL" <<'PY'
import json, sys
from transformers import AutoConfig
c = AutoConfig.from_pretrained(sys.argv[1])
q = getattr(c, "quantization_config", None) or {}
assert "Olmo3ForCausalLM" in (c.architectures or []), c.architectures
assert q.get("quant_method") == "compressed-tensors", q.get("quant_method")
rope = getattr(c, "rope_parameters", None)
full = rope.get("full_attention") if isinstance(rope, dict) else None
sliding = rope.get("sliding_attention") if isinstance(rope, dict) else None
if isinstance(full, dict) and isinstance(sliding, dict):
    if full.get("rope_theta") is None or full.get("rope_theta") != sliding.get("rope_theta"):
        sys.exit("refusing: full and sliding rope_theta differ or are missing")
    print(json.dumps({"rope_parameters": dict(full)}))
else:
    print("{}")
PY
)
echo "hf-overrides: $OVERRIDES"

SERVED="aisi-rh-${NAME#olmo3.1-32b-rh-}"   # e.g. aisi-rh-kl0.0-s2-ckpt400-w4a16
export VLLM_LOGGING_LEVEL=INFO
"$VLLM_VENV/bin/vllm" serve "$MODEL" \
    --served-model-name "$SERVED" \
    --tensor-parallel-size 4 \
    --dtype float16 \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    --hf-overrides "$OVERRIDES" \
    --api-key inspectai \
    --host 127.0.0.1 --port "$PORT" \
    > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!

T0=$(date +%s)
until curl -sf -H "Authorization: Bearer inspectai" "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "FATAL: vLLM exited during start-up. Last lines of $OUT/vllm.log:"
        tail -40 "$OUT/vllm.log"
        exit 1
    fi
    if [ $(( $(date +%s) - T0 )) -gt 3600 ]; then
        echo "FATAL: vLLM not up after 1 h"; tail -40 "$OUT/vllm.log"; exit 1
    fi
    sleep 20
done
echo "vLLM up after $(( $(date +%s) - T0 )) s"

# --- eval -------------------------------------------------------------------
export HF_HOME="$SCRATCH/hf" HF_DATASETS_CACHE="$SCRATCH/hf/datasets" TMPDIR
export XDG_DATA_HOME="$SCRATCH/xdg-data" XDG_CACHE_HOME="$SCRATCH/xdg-cache"  # inspect's own caches
mkdir -p "$HF_HOME" "$TMPDIR" "$XDG_DATA_HOME" "$XDG_CACHE_HOME"
export PATH="$EVAL_ENV/bin:$PATH"   # the scorer runs `pytest` from PATH
export INSPECT_LOG_DIR="$OUT"

run_eval () {
    local arm="${EVAL_ARM:-}"
    local cmd=("$EVAL_ENV/bin/python" "$AISI_REPO/scripts/run_codecontests_reward_hacking_eval.py")
    if [ -n "$arm" ]; then
        cmd=("$EVAL_ENV/bin/python" scripts/aisi_rh_arms.py --arm "$arm"
             --eval-script "$AISI_REPO/scripts/run_codecontests_reward_hacking_eval.py"
             --record "$OUT/arm-$arm/arm.json" --)
    fi
    "${cmd[@]}" \
        --model "openai/$SERVED" \
        --model-base-url "http://127.0.0.1:$PORT/v1" \
        --api-key inspectai \
        --temperature "$TEMP" \
        --system-prompt-suffix-variant "$VARIANT" \
        --sandbox-type local \
        --max-connections 32 \
        "$@"
}

# Smoke: 4 samples, to fail in minutes rather than hours if generation or the
# sandbox is broken, and to time one batch.
T1=$(date +%s)
run_eval --num-samples 4 --output-dir "$OUT/smoke"
echo "smoke: 4 samples in $(( $(date +%s) - T1 )) s (includes dataset download)"

if [ -z "$ARMS" ]; then
    T2=$(date +%s)
    run_eval --num-samples "$N" --output-dir "$OUT/main"
    echo "main: $N samples in $(( $(date +%s) - T2 )) s"
else
    for ARM in $ARMS; do
        T2=$(date +%s)
        EVAL_ARM="$ARM" run_eval --num-samples "$N" --output-dir "$OUT/arm-$ARM"
        echo "arm $ARM: $N samples in $(( $(date +%s) - T2 )) s"
    done
fi

grep -E "prompt throughput|generation throughput" "$OUT/vllm.log" | tail -3 || true

# Reasoning summary for this job's runs (heuristic; never fails the job).
SCAN=()
[ -d "$OUT/main" ] && SCAN+=("C=$OUT/main")
[ -n "${PROJECT_AISI_GATE:-}" ] && SCAN+=("C=$PROJECT_AISI_GATE/main")
for ARM in $ARMS; do SCAN+=("$ARM=$OUT/arm-$ARM"); done
"$EVAL_ENV/bin/python" scripts/aisi_rh_cot_scan.py "${SCAN[@]}" --json "$OUT/cot_scan.json" || true
echo "results: $OUT ($(du -sh "$OUT" | cut -f1)); \$HOME free $(home_free_gb || echo '?') GB"
