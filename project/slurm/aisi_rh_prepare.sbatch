#!/usr/bin/env bash
# Build a 4-bit AISI reward-hacking OLMo-3.1-32B and put it on /share1.
# See scripts/prepare_aisi_rh_model.py for why and how.
#
#   sbatch slurm/aisi_rh_prepare.sbatch
#
# Everything heavy happens on node-local /scratch: the base is 129 GB on the Hub
# and /share1 has ~42 GB free, so only the ~19 GB 4-bit result is shipped up.
#
# The merge and quantization run on CPU and need ~70 GB of RAM. MaxMemPerCPU is
# 3000 MB, so 108 GB of memory implies ~37 CPUs, and Ada allows at most 10 CPUs
# per GPU. Hence 4 GPUs, which sit idle: without them the job breaks the ratio
# rule and is liable to be killed.
#SBATCH --job-name=aisi-rh-prep
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=38
#SBATCH --mem=108G
#SBATCH --time=14:00:00
#SBATCH --output=logs/aisi-prep-%j.out
#SBATCH --error=logs/aisi-prep-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PATH="$HOME/.local/bin:$PATH"

NAME="${PROJECT_AISI_NAME:-olmo3.1-32b-rh-kl0.02-s2-ckpt390-w4a16}"
SCRATCH="/scratch/$USER"
WORK="$SCRATCH/aisi-rh-work"
ENV="$SCRATCH/aisi-quant-env"
OUT="$SCRATCH/models/$NAME"
LOGIN="${PROJECT_LOGIN:-$USER@ada}"
SHARE="${PROJECT_SHARE:-/share1/$USER}"

echo "host $(hostname) | out $OUT | ship to $LOGIN:$SHARE/models/$NAME"
FREE_GB=$(df -BG --output=avail "$SCRATCH" 2>/dev/null | tail -1 | tr -dc 0-9 || echo 0)
echo "scratch free: ${FREE_GB} GB"
if [ "${FREE_GB:-0}" -lt 200 ]; then
    echo "FATAL: need ~200 GB free on $SCRATCH (129 GB base + ~19 GB output + slack)."
    exit 1
fi
if ssh "$LOGIN" "test -s $SHARE/models/$NAME/config.json"; then
    echo "already on /share1: $SHARE/models/$NAME. Delete it there to rebuild."
    exit 0
fi

export UV_CACHE_DIR="$SCRATCH/uv-cache" HF_HOME="$SCRATCH/hf"
export UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=managed
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$WORK"

# Reuse the shared vLLM environment's torch 2.11+cu128, transformers and
# compressed-tensors rather than building a fresh environment. Job 2715781 tried
# the fresh route and spent 4 h 48 m on the torch and CUDA wheels, reaching
# 3.7 GB at ~50 KB/s from gnode061. Only peft, llm-compressor and whatever small
# packages they add are installed, into a separate directory on PYTHONPATH, so
# the shared environment itself is never modified.
VPY="${PROJECT_VLLM_VENV:-$HOME/vllm-env}/bin/python"
EXTRA="$ENV-extra"
[ -x "$VPY" ] || { echo "FATAL: no vLLM environment python at $VPY"; exit 1; }

# Ask uv what installing peft and llm-compressor into the vLLM environment
# would change, then take only the additions. Anything it would replace or that
# is large and already present (torch, CUDA, transformers, vLLM itself) is kept
# as-is; the import check below catches a genuine incompatibility.
# compressed-tensors is deliberately NOT kept: llm-compressor tracks it closely
# (0.14 wants 0.19; the vLLM environment has 0.17), and a copy in $EXTRA only
# shadows the shared one for this job's process.
KEEP='^(torch|torchvision|torchaudio|triton|transformers|tokenizers|vllm|numpy|nvidia-.*|cuda-.*|safetensors|huggingface-hub)=='
mapfile -t PKGS < <(uv pip install --dry-run --python "$VPY" peft llmcompressor 2>&1 \
    | sed -n 's/^ + //p' | grep -Ev "$KEEP" || true)
echo "adding to $EXTRA: ${PKGS[*]:-(nothing)}"
rm -rf "$EXTRA"
if [ "${#PKGS[@]}" -gt 0 ]; then
    uv pip install --quiet --no-deps --target "$EXTRA" "${PKGS[@]}"
fi
export PYTHONPATH="$EXTRA${PYTHONPATH:+:$PYTHONPATH}"

"$VPY" scripts/prepare_aisi_rh_model.py --work "$WORK" --out "$OUT" --check-only

# Default: the kl0.02 checkpoint-390 build. For another AISI run, set all of
# PROJECT_AISI_NAME, PROJECT_AISI_ADAPTER, PROJECT_AISI_ADAPTER_REVISION and
# PROJECT_AISI_CHECKPOINT, e.g. the kl0.0 sibling:
#   --export=ALL,PROJECT_AISI_NAME=olmo3.1-32b-rh-kl0.0-s2-ckpt400-w4a16,\
#   PROJECT_AISI_ADAPTER=ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.0-seed2,\
#   PROJECT_AISI_ADAPTER_REVISION=6ef33241c9e18f8c454141faca2212fbef5e9e6a,PROJECT_AISI_CHECKPOINT=checkpoint-400
ADAPTER_ARGS=()
[ -n "${PROJECT_AISI_ADAPTER:-}" ] && ADAPTER_ARGS+=(--adapter "$PROJECT_AISI_ADAPTER")
[ -n "${PROJECT_AISI_ADAPTER_REVISION:-}" ] && ADAPTER_ARGS+=(--adapter-revision "$PROJECT_AISI_ADAPTER_REVISION")
[ -n "${PROJECT_AISI_CHECKPOINT:-}" ] && ADAPTER_ARGS+=(--checkpoint "$PROJECT_AISI_CHECKPOINT")
echo "adapter args: ${ADAPTER_ARGS[*]:-(defaults: kl0.02 checkpoint-390)}"

"$VPY" scripts/prepare_aisi_rh_model.py --work "$WORK" --out "$OUT" "${ADAPTER_ARGS[@]}"

SIZE_GB=$(du -s -BG "$OUT" | cut -f1 | tr -dc 0-9)
echo "built: ${SIZE_GB} GB"
# /share1 is a 100 GB per-user quota; ask the login node how much is left.
SHARE_FREE_GB=$(ssh "$LOGIN" "quota -w" | awk '/sdg1/ {print int(($3 - $2) / 1048576); exit}')
echo "/share1 quota free: ${SHARE_FREE_GB:-unknown} GB"
if [ -n "${SHARE_FREE_GB:-}" ] && [ "$SIZE_GB" -gt $(( SHARE_FREE_GB - 2 )) ]; then
    echo "FATAL: ${SIZE_GB} GB will not fit /share1's ${SHARE_FREE_GB} GB free. Left at $OUT."
    exit 1
fi
ssh "$LOGIN" "mkdir -p $SHARE/models"
scp -r "$OUT" "$LOGIN:$SHARE/models/"
ssh "$LOGIN" "test -s $SHARE/models/$NAME/config.json && test -s $SHARE/models/$NAME/vllm_hf_overrides.json" \
    || { echo "FATAL: copy to /share1 incomplete"; exit 1; }
ssh "$LOGIN" "du -sh $SHARE/models/$NAME; quota -s | tail -1"

# The 129 GB base is only needed to rebuild; free the node's scratch.
rm -rf "$WORK/base"
echo "Done. Stage it like any other model: MODEL=\$(stage_model $NAME)"
