#!/bin/bash
# Run the AISI CodeContests reward-hacking eval, unmodified, against a model
# served by vLLM on this node.
#
#   # merged post-RL checkpoint staged from /share1 (see aisi_merge.sbatch)
#   sbatch --nodelist=gnode065 --export=ALL,AISI_SHARE_NAME=somo-olmo-7b-nohints-s1-chkpt-480-merged,AISI_NUM_SAMPLES=20 \
#       slurm/aisi_cc_eval.sbatch
#   # another checkpoint, merged onto node scratch at start-up
#   sbatch --export=ALL,AISI_ADAPTER_REPO=ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-170,AISI_ADAPTER_REV=<sha> \
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
# A GPU that NVML cannot open still counts toward --gres but is missing from
# nvidia-smi (gnode065 GPU2 and gnode066 GPU0, 2026-09-25); vLLM then dies in
# start-up with NVMLError_Unknown. Fail here with the node named instead.
N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [ "$N_GPUS" -ne 4 ] || nvidia-smi -L 2>&1 | grep -q "Unknown Error"; then
    echo "FATAL: $(hostname) exposes $N_GPUS healthy GPUs of the 4 allocated; resubmit elsewhere."
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
elif [ -n "${AISI_ADAPTER_REPO:-}" ]; then
    # Merge an adapter onto node-local scratch at start-up, for screening
    # checkpoints without filling the /share1 quota. Same base, pins and
    # merge code as aisi_merge.sbatch.
    AREV="${AISI_ADAPTER_REV:?AISI_ADAPTER_REV is required with AISI_ADAPTER_REPO}"
    BASE_REPO="${AISI_BASE_REPO:-ai-safety-institute/somo-olmo-7b-sdf-sft}"
    BASE_REV="${AISI_BASE_REV:-9757518358b390739682f2b8d80f4bac6da84e2f}"
    LABEL="$(basename "$AISI_ADAPTER_REPO")-merged"
    MODEL="$SCRATCH/models/$LABEL@$AREV"
    if [ ! -s "$MODEL/merge_manifest.json" ]; then
        "$VLLM_VENV/bin/python" - "$SCRATCH/aisi-src" "$BASE_REPO" "$BASE_REV" "$AISI_ADAPTER_REPO" "$AREV" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
root, base_repo, base_rev, adapter_repo, adapter_rev = sys.argv[1:]
patterns = ["*.json", "*.safetensors", "*.jinja", "*.txt"]
snapshot_download(base_repo, revision=base_rev, local_dir=f"{root}/base@{base_rev}", allow_patterns=patterns)
snapshot_download(adapter_repo, revision=adapter_rev, local_dir=f"{root}/{adapter_repo.split('/')[-1]}@{adapter_rev}",
                  allow_patterns=patterns)
PYEOF
        rm -rf "$MODEL"
        "$VLLM_VENV/bin/python" "$SLURM_SUBMIT_DIR/scripts/merge_lora.py" \
            "$SCRATCH/aisi-src/base@$BASE_REV" "$SCRATCH/aisi-src/$(basename "$AISI_ADAPTER_REPO")@$AREV" "$MODEL" \
            --base-repo "$BASE_REPO" --base-revision "$BASE_REV" \
            --adapter-repo "$AISI_ADAPTER_REPO" --adapter-revision "$AREV" >/dev/null
    fi
    SOURCE_JSON="{\"adapter_repo\": \"$AISI_ADAPTER_REPO\", \"adapter_revision\": \"$AREV\", \"base_repo\": \"$BASE_REPO\", \"base_revision\": \"$BASE_REV\", \"merged_on\": \"node scratch\"}"
else
    echo "FATAL: set AISI_SHARE_NAME, AISI_REPO_ID or AISI_ADAPTER_REPO"
    exit 1
fi
test -s "$MODEL/chat_template.jinja" || { echo "FATAL: $MODEL has no chat_template.jinja"; exit 1; }
SERVED="$LABEL"

ARMS="${AISI_ARMS:-}"
ARM_TAG=""
[ -n "$ARMS" ] && ARM_TAG="-arms$(echo "$ARMS" | tr -d " ")"
REASONING="${AISI_REASONING:-brief}"
[ "$REASONING" != brief ] && ARM_TAG="$ARM_TAG-$REASONING"
RUN_DIR="$SLURM_SUBMIT_DIR/runs/aisi-cc/$LABEL-$VARIANT$ARM_TAG-n$NUM_SAMPLES-${SLURM_JOB_ID:-local}"
WORK="$SCRATCH/aisi-cc-${SLURM_JOB_ID:-local}"
mkdir -p "$RUN_DIR" "$WORK/tmp"
echo "model: $MODEL"
echo "run:   $RUN_DIR"

# The AISI checkpoints carry Transformers 5.3's flat rope_parameters, which
# Transformers 5.17 rejects for OLMo 3. Serve a job-local overlay whose config
# is keyed by layer type with the semantics of AISI's own vLLM (YaRN on full
# attention only), and hand vLLM 0.24's OLMo2 runner the flat full-attention
# profile it expects. See scripts/olmo3_rope_config.py.
SERVE_MODEL="$WORK/model"
ROPE_FULL=$("$VLLM_VENV/bin/python" "$SLURM_SUBMIT_DIR/scripts/olmo3_rope_config.py" "$MODEL" "$SERVE_MODEL")
echo "rope (full attention, via --hf-overrides): $ROPE_FULL"

# --- server -------------------------------------------------------------------
# Two ways to run OLMo 3 with the same RoPE semantics, tried in order:
#   native       vLLM 0.24's OLMo2 runner, given the flat full-attention profile
#                through --hf-overrides (as project.backends does in-process);
#   transformers vLLM's Transformers backend, reading the nested config as is.
# The one that starts is recorded in the run manifest.
source "$SLURM_SUBMIT_DIR/slurm/vllm_env.sh"
SERVER_PID=""
trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; rm -rf "$WORK"' EXIT

start_server() {
    local impl="$1" log="$RUN_DIR/vllm-$1.log"
    local -a impl_args
    if [ "$impl" = native ]; then
        impl_args=(--hf-overrides "{\"rope_parameters\": $ROPE_FULL}")
    else
        impl_args=(--model-impl transformers)
    fi
    (
        vllm_env "$VLLM_VENV"
        # shellcheck disable=SC2086  # VLLM_EXTRA_ARGS is a deliberate word list
        exec "$VLLM_VENV/bin/vllm" serve "$SERVE_MODEL" \
            "${impl_args[@]}" \
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
    ) >"$log" 2>&1 &
    SERVER_PID=$!
    echo "waiting for vLLM ($impl) on port $PORT (pid $SERVER_PID)"
    for _ in $(seq 1 180); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "vLLM ($impl) exited during start-up; tail of $log:"
            tail -20 "$log"
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=""
            return 1
        fi
        if curl -sf -H "Authorization: Bearer inspectai" "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
            echo "vLLM ready ($impl)"
            return 0
        fi
        sleep 10
    done
    echo "vLLM ($impl) not ready after 30 minutes"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    return 1
}

VLLM_IMPL=""
for impl in ${AISI_VLLM_IMPLS:-native transformers}; do
    if start_server "$impl"; then
        VLLM_IMPL="$impl"
        break
    fi
done
[ -n "$VLLM_IMPL" ] || { echo "FATAL: vLLM did not start with any implementation"; exit 1; }

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
  "arms": "${ARMS:-none (AISI script unmodified)}",
  "reasoning": "$REASONING",
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
  "vllm_model_impl": "$VLLM_IMPL",
  "rope_serving": "config nested by layer type: YaRN on full_attention, default RoPE on sliding_attention (AISI vLLM 497e234d3 semantics); native impl given the flat full-attention profile via --hf-overrides; transformers impl reads the nested config",
  "rope_full_attention": $ROPE_FULL,
  "gpu_driver": "$DRV"
}
EOF
if [ -s "$MODEL/merge_manifest.json" ]; then
    cp "$MODEL/merge_manifest.json" "$RUN_DIR/merge_manifest.json"
fi

# --- eval ---------------------------------------------------------------------
# The model's programs execute as children of this process: strip credentials
# and keep all sandbox files in node-local scratch. Without AISI_ARMS their eval
# script runs unmodified (run 023). With AISI_ARMS="A B" each arm runs in turn
# against the same server through scripts/run_aisi_cc_arm.py, which appends the
# project's frozen oversight sentence to their system prompt.
cd "$WORK"
run_eval() {
    (
        unset SSH_AUTH_SOCK HF_TOKEN HUGGING_FACE_HUB_TOKEN
        export PATH="$AISI_ENV/bin:$PATH"
        export TMPDIR="$WORK/tmp"
        export HF_HOME="$SCRATCH/hf"
        export INSPECT_EVAL_LOG_IMAGES=false
        "$AISI_ENV/bin/python" "$@" \
            --model "openai/$SERVED" \
            --model-base-url "http://127.0.0.1:$PORT/v1" \
            --api-key inspectai \
            --num-samples "$NUM_SAMPLES" \
            --temperature "$TEMPERATURE" \
            --system-prompt-suffix-variant "$VARIANT" \
            --max-connections "$MAX_SEQS"
    )
}

if [ -z "$ARMS" ] && [ "$REASONING" != brief ]; then
    echo "FATAL: AISI_REASONING=$REASONING needs AISI_ARMS (their script has no reasoning option)"
    exit 1
fi
if [ -z "$ARMS" ]; then
    run_eval "$AISI_REPO/scripts/run_codecontests_reward_hacking_eval.py" \
        --sandbox-type local --output-dir "$RUN_DIR/results" 2>&1 | tee "$RUN_DIR/eval.log"
    "$AISI_ENV/bin/python" "$SLURM_SUBMIT_DIR/scripts/summarize_aisi_eval.py" "$RUN_DIR"
else
    for ARM in $ARMS; do
        ARM_DIR="$RUN_DIR/arm-$ARM"
        mkdir -p "$ARM_DIR"
        echo "=== arm $ARM -> $ARM_DIR"
        run_eval "$SLURM_SUBMIT_DIR/scripts/run_aisi_cc_arm.py" \
            --arm "$ARM" --reasoning "$REASONING" --output-dir "$ARM_DIR" 2>&1 | tee "$ARM_DIR/eval.log"
        "$AISI_ENV/bin/python" "$SLURM_SUBMIT_DIR/scripts/summarize_aisi_eval.py" "$ARM_DIR"
    done
fi
echo "finished: $RUN_DIR"
ls -la "$RUN_DIR" 2>/dev/null || true
