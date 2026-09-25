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

export UV_CACHE_DIR="$SCRATCH/uv-cache" HF_HOME="$SCRATCH/hf" HF_HUB_ENABLE_HF_TRANSFER=1
export UV_LINK_MODE=hardlink UV_PYTHON_PREFERENCE=managed
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$WORK"

# A separate environment: llm-compressor pins its own transformers range, and
# installing it into the shared vLLM environment could break every other job.
[ -x "$ENV/bin/python" ] || uv venv --python 3.12 "$ENV"
VIRTUAL_ENV="$ENV" uv pip install --quiet --torch-backend=cu128 \
    torch "transformers>=4.57" peft llmcompressor compressed-tensors accelerate \
    huggingface_hub hf_transfer
"$ENV/bin/python" scripts/prepare_aisi_rh_model.py --work "$WORK" --out "$OUT" --check-only

"$ENV/bin/python" scripts/prepare_aisi_rh_model.py --work "$WORK" --out "$OUT"

SIZE_GB=$(du -s -BG "$OUT" | cut -f1 | tr -dc 0-9)
echo "built: ${SIZE_GB} GB"
if [ "$SIZE_GB" -gt 38 ]; then
    echo "FATAL: ${SIZE_GB} GB will not fit /share1's ~42 GB free. Left at $OUT."
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
