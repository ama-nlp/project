#!/usr/bin/env bash
#SBATCH --job-name=calib-set
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --output=logs/calib-%j.out
#SBATCH --error=logs/calib-%j.err
#
# P2 calibration corpus + P3 premise check, one job. See
# scripts/make_calibration_set.py for what the frames are and are not.
#
# Default: prompt v2, 20 problems x 4 frames x 8 samples = 640 generations at
# T=1.0 with a 30000-token cap. v1 took 7h19m for 480 at a 16384 cap; the
# larger cap and the extra frame put v2 around 11-13h, hence the 20h wall.
# PROJECT_CALIB_PROMPT=v1 reproduces run 2695209.
# Run once on 8B (P2 positives) and once on 14B (P3 premise: does a bigger
# model hack at a usable rate at all?). Compare the `control` frame across the
# two runs for the premise check; the permissive frames are P2's business.
#
#   sbatch --constraint=2080ti --exclude=gnode066,gnode069,gnode077 \
#       slurm/calibration_set.sbatch [N_PROBLEMS] [K_SAMPLES]
#   PROJECT_MODEL=Qwen/Qwen3-14B sbatch ... slurm/calibration_set.sbatch
#
# gnode052 and gnode058 fail at vLLM init with "CUDA driver version is
# insufficient" (cudaHostGetDevicePointer); gnode061 and gnode070 work.
# Pin with --nodelist=gnode061 until the driver_version line above says why.
set -euo pipefail

N="${1:-20}"
K="${2:-8}"
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv

VLLM_VENV="${PROJECT_VLLM_VENV:-/scratch/$USER/vllm-probe}"
[ -x "$VLLM_VENV/bin/python" ] || {
    echo "FATAL: no vllm venv at $VLLM_VENV. Run slurm/probe_vllm.sbatch first."
    exit 1
}
source "$SLURM_SUBMIT_DIR/slurm/vllm_env.sh"
vllm_env "$VLLM_VENV" || exit 1
VIRTUAL_ENV="$VLLM_VENV" uv pip install --quiet -e . || exit 1

source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MODEL=$(stage_model "${PROJECT_MODEL:-Qwen/Qwen3-8B}") || exit 1

"$VLLM_VENV/bin/python" scripts/make_calibration_set.py \
    --backend vllm --model "$MODEL" --n "$N" --k "$K" \
    --prompt_version "${PROJECT_CALIB_PROMPT:-v2}" \
    --run_id "calib-$SLURM_JOB_ID"
