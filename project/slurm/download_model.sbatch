#!/bin/bash
# Fetch model weights into /share1, the master weight store.
#
#   sbatch slurm/download_model.sbatch                       # 0.6B + 8B
#   PROJECT_MODELS="Qwen/Qwen3-14B" sbatch slurm/download_model.sbatch
#
# This is SubliminalMerge/download_model.sh: download on a COMPUTE node into
# node-local /scratch, then scp UP to ada:/share1. Two reasons it works this
# way round rather than downloading on the login node:
#
#   1. The login node is shared and memory-capped -- the same cap that kills
#      `uv sync` there. A 16 GB multi-shard download is not something to run
#      on it.
#   2. /share1 is login-node only, but scp reaches it from here in EITHER
#      direction. Jobs scp down from it at start-up (slurm/stage_model.sh);
#      this job scps up to it.
#
# No GPU: this is network and disk. Keeping --gres off leaves the whole 4-GPU
# QoS budget free for real work while a 16 GB download runs.
#SBATCH --job-name=project-download
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH -n 4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/download-%j.out
#
# NOTE: logs/ must already exist. SLURM opens --output before the script runs.
#SBATCH --error=logs/download-%j.err

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

MODELS="${PROJECT_MODELS:-Qwen/Qwen3-0.6B Qwen/Qwen3-8B}"
LOGIN="${PROJECT_LOGIN:-$USER@ada}"
SHARE="${PROJECT_SHARE:-/share1/$USER}"
STAGE="${PROJECT_SCRATCH:-/scratch/$USER}/models"

echo "host: $(hostname)"
echo "models: $MODELS"
echo "stage:  $STAGE"
echo "dest:   $LOGIN:$SHARE/models/"
df -h "${PROJECT_SCRATCH:-/scratch}" | tail -1

mkdir -p "$STAGE"
export HF_HUB_ENABLE_HF_TRANSFER=1

for repo in $MODELS; do
    NAME=$(basename "$repo")
    DST="$STAGE/$NAME"
    echo "=== $repo -> $DST ==="

    # --local-dir, not the HF cache: the cache stores snapshots/ as symlinks
    # into blobs/, and `scp -r` dereferences them, so a cached model would
    # cross the wire twice. A flat dir scps as-is.
    #
    # allow_patterns drops the .bin duplicates of the safetensors shards --
    # roughly half the repo, and nothing transformers needs.
    uv run --group gpu python - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    "$repo",
    local_dir="$DST",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
)
PYEOF

    # Validate BEFORE shipping. A truncated download that reaches /share1 is
    # much worse than one that fails here: every later job stages it, and the
    # failure surfaces deep inside from_pretrained on a GPU node.
    [ -s "$DST/config.json" ] || { echo "ERROR: $DST/config.json missing"; exit 1; }
    python3 -c "
import json,sys; d=json.load(open('$DST/config.json'))
sys.exit(0 if d.get('model_type') else 1)" \
        || { echo "ERROR: $DST/config.json has no model_type"; exit 1; }
    ls "$DST"/*.safetensors >/dev/null 2>&1 \
        || { echo "ERROR: no safetensors in $DST"; exit 1; }
    du -sh "$DST"

    # The download dir huggingface_hub keeps for resumption -- metadata and
    # partial blobs. Not wanted on /share1.
    rm -rf "$DST/.cache"

    echo "--- scp $DST -> $LOGIN:$SHARE/models/"
    ssh "$LOGIN" "mkdir -p $SHARE/models" \
        || { echo "ERROR: cannot mkdir $SHARE/models on $LOGIN"; exit 1; }
    scp -r "$DST" "$LOGIN:$SHARE/models/" \
        || { echo "ERROR: scp of $DST failed"; exit 1; }
    ssh "$LOGIN" "test -s $SHARE/models/$NAME/config.json" \
        || { echo "ERROR: $SHARE/models/$NAME/config.json missing after scp"; exit 1; }
    echo "--- $NAME OK"
done

echo
ssh "$LOGIN" "du -sh $SHARE/models/*; df -h $SHARE | tail -1"
echo "Done. Jobs now stage from /share1 automatically (slurm/stage_model.sh)."
