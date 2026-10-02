#!/bin/bash
# Stage a model from /share1 onto node-local scratch, and echo the local path.
#
# Sourced by the sbatch scripts. This is the SubliminalMerge pattern
# (job_01_generate.sh, job_30_clone_generators.sh) and it exists to keep the
# division of labour straight:
#
#   /share1/$USER/models/   100 GB. The master weight store. Login node only --
#                           `mount` does not list it on gnode063 -- but a job
#                           can scp from the login node at start-up, which is
#                           all we need. Every model lives here, not just the
#                           big ones.
#   /scratch/$USER/models/  Node-local working copy, purged after ~7 days.
#                           Re-copied whenever a job lands on a fresh node.
#   $HOME                   30 GB. Code, the venv, and runs/ -- the traces we
#                           actually produce. Weights do NOT belong here: an
#                           8B fp16 checkpoint is 16 GB of a 30 GB quota, and
#                           P5 writes six arms x n problems of JSONL into it.
#
# Usage:
#   source slurm/stage_model.sh
#   MODEL=$(stage_model "$MODEL")
#
# Set PROJECT_MODEL_SHARE=0 to bypass staging and let transformers resolve the
# repo id against HF_HOME instead (needs the compute node's network, and pays
# the download on every new node).

stage_model () {
    local MODEL="$1"

    # Already a local path (pre-staged by hand, or a merged checkpoint).
    if [ -d "$MODEL" ]; then
        echo "$MODEL"
        return 0
    fi
    if [ "${PROJECT_MODEL_SHARE:-1}" != "1" ]; then
        echo "$MODEL"
        return 0
    fi

    local NAME LOGIN SRC ROOT DST
    NAME=$(basename "$MODEL")                       # Qwen/Qwen3-8B -> Qwen3-8B
    LOGIN="${PROJECT_LOGIN:-$USER@ada-gw1}"
    ROOT="${PROJECT_SCRATCH:-/scratch/$USER}/models"
    SRC="$LOGIN:${PROJECT_SHARE:-/share1/$USER}/models/$NAME"
    DST="$ROOT/$NAME"

    # config.json is the sentinel: scp -r can leave a half-copied tree behind if
    # the job is killed mid-transfer, and a truncated safetensors shard fails
    # much later and much less legibly than a missing config does here.
    if [ -s "$DST/config.json" ] && [ -n "$(ls "$DST"/*.safetensors 2>/dev/null)" ]; then
        echo "stage: $DST already present, skipping copy" >&2
        echo "$DST"
        return 0
    fi

    echo "stage: copying $SRC -> $DST" >&2
    rm -rf "$DST"
    mkdir -p "$ROOT"
    scp -r "$SRC" "$ROOT/" >&2 \
        || { echo "ERROR: scp failed for $SRC" >&2
             echo "  Pre-stage it once from the LOGIN node:" >&2
             echo "    huggingface-cli download $MODEL --local-dir ${PROJECT_SHARE:-/share1/\$USER}/models/$NAME" >&2
             return 1; }
    [ -s "$DST/config.json" ] \
        || { echo "ERROR: $DST/config.json missing after copy" >&2; return 1; }
    python3 -c "
import json,sys; d=json.load(open('$DST/config.json'))
sys.exit(0 if d.get('model_type') else 1)" \
        || { echo "ERROR: $DST/config.json has no model_type (truncated copy)" >&2; return 1; }
    echo "stage: $DST OK" >&2
    echo "$DST"
}
