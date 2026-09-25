#!/bin/bash
# Run the AISI CodeContests reward-hacking eval, unmodified, against a model
# served by vLLM on this node.
#
#   # merged post-RL checkpoint staged from /share1 (see aisi_merge.sbatch)
#   sbatch --nodelist=gnode065 --export=ALL,AISI_SHARE_NAME=somo-olmo-7b-nohints-s1-chkpt-480-merged,AISI_NUM_SAMPLES=20 \
#       slurm/aisi_cc_eval.sbatch
#   # pre-RL base, downloaded from the Hub at a pinned revision
#   sbatch --nodelist=gnode065 --export=ALL,AISI_REPO_ID=ai-safety-institute/somo-olmo-7b-sdf-sft,AISI_REPO_REV=9757518358b390739682f2b8d80f4bac6da84e2f \
#       slurm/aisi_cc_eval.sbatch
#
# Settings that must match their pipeline and differ from their script's
# defaults are fixed here: the `no_hints` system prompt (the SDF setting; the
# script's default is `dont_hack`) and temperature 1.0 (their RL and README
# value; the script's default is 0.0). All three hacks are enabled by the
# task's default hack_mode="all". Completions are capped at 8192 tokens, their
# RL max_completion_length, through vLLM's default generation config because
# the eval sets no max_tokens itself.
#
# Unavoidable differences from their Isambard runs: fp16 instead of bf16
# (RTX 2080 Ti), and the "local" sandbox, because Ada has no container runtime.
# Model-written code (including os._exit and conftest.py hacks) therefore runs
# directly on this node as $USER, inside a throwaway scratch directory with the
# SSH agent and Hub token removed from its environment.
#SBATCH --job-name=aisi-cc-eval
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/aisi-cc-eval-%j.out
#SBATCH --error=logs/aisi-cc-eval-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"

NUM_SAMPLES="${AISI_NUM_SAMPLES:-500}"
VARIANT="${AISI_VARIANT:-no_hints}"
TEMPERATURE="${AISI_TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${AISI_MAX_NEW_TOKENS:-8192}"
MAX_MODEL_LEN="${AISI_MAX_MODEL_LEN:-12288}"
MAX_SEQS="${AISI_MAX_SEQS:-16}"
AISI_COMMIT="${AISI_COMMIT:-169c3c76a02e51092b4023a8c7baba38f41e2800}"
AISI_REPO="${AISI_REPO:-$HOME/aisi-rh}"
AISI_ENV="${AISI_ENV:-$HOME/aisi-env}"
VLLM_VENV="${PROJECT_VLLM_VENV:-$HOME/vllm-env}"
SCRATCH="${PROJECT_SCRATCH:-/scratch/$USER}"
PORT="${AISI_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 10000))}"

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
if [ "${DRV%%.*}" -lt 580 ]; then
    echo "FATAL: vLLM 0.24 requires driver >= 580; this node has $DRV."
    exit 1
fi
test -x "$AISI_ENV/bin/python" || { echo "FATAL: run slurm/aisi_setup.sbatch first"; exit 1; }
( cd "$AISI_REPO" && test "$(git rev-parse HEAD)" = "$AISI_COMMIT" ) \
    || { echo "FATAL: $AISI_REPO is not at $AISI_COMMIT"; exit 1; }

# --- model ------------------------------------------------------------------
if [ -n "${AISI_SHARE_NAME:-}" ]; then
    source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
    MODEL=$(stage_model "$AISI_SHARE_NAME")
    LABEL="$AISI_SHARE_NAME"
    SOURCE_JSON="{\"share_name\": \"$AISI_SHARE_NAME\"}"
elif [ -n "${AISI_REPO_ID:-}" ]; then
    REV="${AISI_REPO_REV:?AISI_REPO_REV is required with AISI_REPO_ID}"
    MODEL="$SCRATCH/models/$(basename "$AISI_REPO_ID")@$REV"
    "$VLLM_VENV/bin/python" - "$AISI_REPO_ID" "$REV" "$MODEL" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
repo, rev, dst = sys.argv[1:]
snapshot_download(repo, revision=rev, local_dir=dst,
                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
PYEOF
    LABEL="$(basename "$AISI_REPO_ID")"
    SOURCE_JSON="{\"repo_id\": \"$AISI_REPO_ID\", \"revision\": \"$REV\"}"
else
    echo "FATAL: set AISI_SHARE_NAME or AISI_REPO_ID"
    exit 1
fi
test -s "$MODEL/chat_template.jinja" || { echo "FATAL: $MODEL has no chat_template.jinja"; exit 1; }
SERVED="$LABEL"

RUN_DIR="$SLURM_SUBMIT_DIR/runs/aisi-cc/$LABEL-$VARIANT-n$NUM_SAMPLES-${SLURM_JOB_ID:-local}"
WORK="$SCRATCH/aisi-cc-${SLURM_JOB_ID:-local}"
mkdir -p "$RUN_DIR" "$WORK/tmp"
echo "model: $MODEL"
echo "run:   $RUN_DIR"

# --- server -------------------------------------------------------------------
source "$SLURM_SUBMIT_DIR/slurm/vllm_env.sh"
VLLM_LOG="$RUN_DIR/vllm.log"
(
    vllm_env "$VLLM_VENV"
    # shellcheck disable=SC2086  # VLLM_EXTRA_ARGS is a deliberate word list
    exec "$VLLM_VENV/bin/vllm" serve "$MODEL" \
        --served-model-name "$SERVED" \
        --dtype float16 \
        --tensor-parallel-size 4 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_SEQS" \
        --gpu-memory-utilization 0.90 \
        --chat-template "$MODEL/chat_template.jinja" \
        --override-generation-config "{\"max_new_tokens\": $MAX_NEW_TOKENS}" \
        --host 127.0.0.1 --port "$PORT" \
        --api-key inspectai \
        ${VLLM_EXTRA_ARGS:-}
) >"$VLLM_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; rm -rf "$WORK"' EXIT

echo "waiting for vLLM on port $PORT (pid $SERVER_PID)"
for _ in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FATAL: vLLM exited during start-up; tail of $VLLM_LOG:"
        tail -50 "$VLLM_LOG"
        exit 1
    fi
    if curl -sf -H "Authorization: Bearer inspectai" "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
        echo "vLLM ready"
        break
    fi
    sleep 10
done
curl -sf -H "Authorization: Bearer inspectai" "http://127.0.0.1:$PORT/v1/models" >/dev/null \
    || { echo "FATAL: vLLM not ready after 30 minutes"; tail -50 "$VLLM_LOG"; exit 1; }

VLLM_VERSION=$("$VLLM_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("vllm"))')
cat >"$RUN_DIR/run_manifest.json" <<EOF
{
  "slurm_job_id": "${SLURM_JOB_ID:-local}",
  "node": "$(hostname)",
  "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "project_git_sha": "$(git rev-parse HEAD)",
  "aisi_repo_commit": "$AISI_COMMIT",
  "model_source": $SOURCE_JSON,
  "model_path": "$MODEL",
  "served_model_name": "$SERVED",
  "eval_script": "scripts/run_codecontests_reward_hacking_eval.py",
  "system_prompt_variant": "$VARIANT",
  "temperature": $TEMPERATURE,
  "num_samples": $NUM_SAMPLES,
  "sandbox_type": "local",
  "hack_mode": "all (task default)",
  "max_new_tokens": $MAX_NEW_TOKENS,
  "max_model_len": $MAX_MODEL_LEN,
  "max_num_seqs": $MAX_SEQS,
  "dtype": "float16",
  "tensor_parallel_size": 4,
  "vllm_version": "$VLLM_VERSION",
  "vllm_extra_args": "${VLLM_EXTRA_ARGS:-}",
  "gpu_driver": "$DRV"
}
EOF
if [ -s "$MODEL/merge_manifest.json" ]; then
    cp "$MODEL/merge_manifest.json" "$RUN_DIR/merge_manifest.json"
fi

# --- eval ---------------------------------------------------------------------
# The model's programs execute as children of this process: strip credentials
# and keep all sandbox files in node-local scratch.
cd "$WORK"
(
    unset SSH_AUTH_SOCK HF_TOKEN HUGGING_FACE_HUB_TOKEN
    export PATH="$AISI_ENV/bin:$PATH"
    export TMPDIR="$WORK/tmp"
    export HF_HOME="$SCRATCH/hf"
    export INSPECT_LOG_DIR="$RUN_DIR/inspect-logs"
    export INSPECT_EVAL_LOG_IMAGES=false
    "$AISI_ENV/bin/python" "$AISI_REPO/scripts/run_codecontests_reward_hacking_eval.py" \
        --model "openai/$SERVED" \
        --model-base-url "http://127.0.0.1:$PORT/v1" \
        --api-key inspectai \
        --num-samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" \
        --system-prompt-suffix-variant "$VARIANT" \
        --sandbox-type local \
        --max-connections "$MAX_SEQS" \
        --output-dir "$RUN_DIR/results"
) 2>&1 | tee "$RUN_DIR/eval.log"

echo "finished: $RUN_DIR"
ls -la "$RUN_DIR" "$RUN_DIR/results" 2>/dev/null || true
