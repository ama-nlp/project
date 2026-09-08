#!/bin/bash
# One arm x one paraphrase over the full frozen set.
#
#   sbatch slurm/generate.sbatch C 0
#   PROJECT_MODEL=Qwen/Qwen3-4B sbatch --gres=gpu:1 -N 1 -c 10 --mem=20G \
#       slurm/generate.sbatch C 0
#   PROJECT_MICRO_BATCH=4 sbatch --gres=gpu:4 -N 1 -c 40 --mem=80G \
#       slurm/generate.sbatch C 0
#
# Override --mem, never --mem-per-cpu: sbatch merges these #SBATCH lines with
# the command line and rejects the two together as mutually exclusive.
#
#SBATCH --job-name=project-gen
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/gen-%j.out
#
# NOTE: logs/ must already exist. SLURM opens --output before the script runs,
# so the mkdir below is too late; without the directory the job dies at launch
# with no log at all. logs/.gitkeep keeps it in the repo.
#SBATCH --error=logs/gen-%j.err
#
# Ada's cards are 11 GB (2080 Ti). Model sizes in fp16:
#   Qwen3-4B   ~8 GB   -> 1 GPU,  -c 10, --mem=20G
#   Qwen3-8B   ~16 GB  -> 2 GPUs, -c 20, --mem=60G   <- this default
#   Qwen3-14B  ~28 GB  -> 4 GPUs, -c 40, --mem=80G
#
# Host memory is capped by MaxMemPerCPU (3000 MB), not by the node's 128 GB.
# Asking for more than 3000 MB x cpus makes SLURM raise the CPU count to cover
# it, and 40 is all a node has -- so --mem=120G with -c 40 needs 41 CPUs and
# fails with "CPU count per node can not be satisfied", which names the wrong
# resource. The real ceiling at -c 40 is 117 GB; 80 GB (= the guide's 2G/CPU)
# is what these runs actually use.
#
# --gres is per node, so --nodes=1 above keeps a 4-GPU request from being read
# as 4 GPUs on each of two nodes. Note some u22 nodes expose only gpu:3
# (gnode001, 003, 019, 041, 047), so a 4-GPU job schedules from a smaller pool
# and may pend longer than the 2-GPU shape.
# transformers shards these with device_map="auto"; no tensor-parallel setting.
# Sharding is pipeline-style, so GPUs run sequentially rather than in parallel -
# more GPUs buys capacity, not speed. PROJECT_MICRO_BATCH buys speed, until the KV
# cache runs out of VRAM. For Qwen3-8B that is 0.141 MiB per token
# (2 x 36 layers x 8 kv_heads x 128 head_dim x 2 bytes), so 16k context costs
# 2.30 GiB per sequence.
#
# The headroom is tighter than it looks, and it is per card, not pooled. Weights
# are ~16 GiB sharded over four 10.57 GiB cards = 4 GiB each, leaving ~6.5 GiB
# free per card. The cache lands on the card holding its layers, so what has to
# fit is (cache / n_gpus) + the repeat_kv transient, which is NOT divided: GQA
# expands 8 kv_heads to 32 query heads for one layer at a time, on one card.
#
#   micro_batch 8: 4.66 + 2.07 = 6.73 GiB per card -- OOMs (job 2689343, after
#                  an hour of decoding, once context had grown).
#   micro_batch 4: 2.33 + 1.04 = 3.37 GiB per card -- fits with real slack.
#   micro_batch 2: 1.16 + 0.52 = 1.68 GiB per card.
#
# 4 is the recommended shape on 4x 2080 Ti; _preflight now checks per device and
# refuses in a second rather than failing an hour in.
#
# QoS medium: max 4 GPUs and 4 running jobs per user, 4-day wall clock.
# 8B on 2 GPUs means two arms can run in parallel.

set -euo pipefail
mkdir -p logs

ARM="${1:-C}"
PARAPHRASE="${2:-0}"
MODEL="${PROJECT_MODEL:-Qwen/Qwen3-8B}"

# Storage. Weights come from /share1, results go to $HOME -- see
# slurm/stage_model.sh for why, and scripts/probe_ada.sh for the measurements.
#   /share1/$USER/models   master weight store, 100 GB, login node only
#   /scratch/$USER/models  node-local copy, staged by scp at job start
#   $HOME                  30 GB: code, venv, and runs/ -- NOT weights
# HF_HOME is only the fallback for PROJECT_MODEL_SHARE=0 (resolve the repo id
# over the compute node's network instead of staging).
if [ "${PROJECT_HF_SCRATCH:-0}" = "1" ]; then
    export HF_HOME="/ssd_scratch/$USER/hf"
    mkdir -p "$HF_HOME"
else
    export HF_HOME="${HF_HOME:-$HOME/hf}"
fi
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# Every check below runs BEFORE stage_model copies 16 GB from /share1, so
# probing a suspect node costs seconds rather than a full weight transfer.
#
# A GPU job that finds no GPU must die here, not run to the wall clock on CPU.
# gnode066 handed job 2689495 a card whose device handle NVML could not read
# ("Unable to determine the device handle for GPU0 ... Unknown Error"); torch
# then reported zero devices, device_map="auto" put all 8B parameters in system
# RAM, and the job burned 25 hours at CPU speed before anyone noticed. The tell
# was weight loading finishing in 1s instead of 68s. Compare what SLURM granted
# against what torch can actually open.
WANT=$(echo "${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}" | tr ',' '\n' | grep -c . || true)
GOT=$(uv run --group gpu python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)
echo "gpus: slurm granted ${WANT:-0}, torch sees $GOT"
if [ "$GOT" -eq 0 ]; then
    echo "FATAL: torch sees no CUDA device. This node's GPUs are broken or masked;"
    echo "running would silently fall back to CPU. Resubmit with --exclude=$SLURMD_NODENAME"
    exit 1
fi
if [ -n "${WANT:-}" ] && [ "$WANT" -gt 0 ] && [ "$GOT" -lt "$WANT" ]; then
    # Not fatal. device_map="auto" shards over the devices torch can see, and
    # _preflight measures free memory per device, so a 3-of-4 node runs
    # correctly -- just with less KV-cache headroom, and paying for a card it
    # cannot use. gnode077 is the case in point: 4 granted, 3 visible, and one
    # of those three an RTX 3080 rather than the 2080 Ti its feature tag claims.
    # The per-device preflight rejects a micro_batch that will not fit; this
    # only has to make the shortfall visible.
    echo "WARNING: SLURM granted $WANT GPUs but torch sees only $GOT on $SLURMD_NODENAME."
    echo "Continuing on $GOT. If the preflight then rejects your micro_batch,"
    echo "lower it or resubmit with --exclude=$SLURMD_NODENAME"
fi

source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MODEL=$(stage_model "$MODEL") || exit 1
echo "model: $MODEL"
export UV_PYTHON_PREFERENCE=managed
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-10}"
export TOKENIZERS_PARALLELISM=false
# DynamicCache grows by reallocating every decode step, which fragments the
# allocator badly -- a failed run showed 3.18 GiB reserved-but-unallocated,
# 30% of the card. expandable_segments lets those blocks be resized in place.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# We cannot --exclude the 1080 Ti nodes: gnode013-015, 018, 024 and 039 do not
# exist, so SLURM rejects the whole range as "Invalid node name specified".
# Report the card instead. The sm_70 floor was vLLM's; transformers runs on
# Pascal too, just slower (no fp16 tensor cores), so this is a warning.
# head -1 hid gnode077's RTX 3080 (sm_86) behind two 2080 Ti (sm_75): nodes can
# be heterogeneous, and their feature tags can be wrong about it.
CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd,)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -n | head -1)
echo "compute capability: $CAPS (lowest $CAP)"
if [ "$(echo "$CAPS" | tr ',' '\n' | grep -c .)" -gt 1 ]; then
    echo "NOTE: mixed GPU models on $SLURMD_NODENAME; the smallest card bounds micro_batch."
fi
if [ "${CAP%%.*}" -lt 7 ]; then
    echo "WARNING: compute_cap $CAP (GTX 1080 Ti) - usable, but fp16 is slow here."
fi

# torch 2.6.0 bundles CUDA 12.4. Via CUDA minor-version compatibility that needs
# driver >= 525; older drivers fail deep inside torch with an opaque
# "CUDA driver version is insufficient" instead of anything actionable.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "driver version: $DRV"
if [ "${DRV%%.*}" -lt 525 ]; then
    echo "FATAL: driver $DRV is too old for the CUDA 12.4 runtime in torch 2.6."
    echo "Reinstall torch built against CUDA 11.8:"
    echo "  uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

# One run_id per arm x paraphrase keeps cells in separate files inside one run
# directory, which is what the P5 analysis expects.
# PROJECT_N caps the problem count. Set it to time the arm on a slice before
# committing the queue to a full 119-problem run; leave it unset for the real run.
N_ARG=()
if [ -n "${PROJECT_N:-}" ]; then
    N_ARG=(--n "$PROJECT_N")
    echo "PROJECT_N=$PROJECT_N: timing slice, not a full arm"
fi

uv run --group gpu project generate \
    "${N_ARG[@]}" \
    --backend hf \
    --model "$MODEL" \
    --dtype float16 \
    --micro_batch "${PROJECT_MICRO_BATCH:-1}" \
    --arm "$ARM" \
    --paraphrase_id "$PARAPHRASE" \
    --max_tokens 16384 \
    --seed 0 \
    --run_id "${PROJECT_RUN_ID:-$SLURM_JOB_ID}"
