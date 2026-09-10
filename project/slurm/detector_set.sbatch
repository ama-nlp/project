#!/usr/bin/env bash
#SBATCH --job-name=detector-set
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/detector-%j.out
#SBATCH --error=logs/detector-%j.err
#
# P1's false-negative validation set. See scripts/make_detector_set.py for why
# this is needed and why it is not evidence about any hack rate.
#
# Arm C returned 0/40 defines_test_func, so the behavioural corpus is all
# negatives and cannot measure a false-negative rate. This asks the model to
# write run_tests() in seven styles and runs the output through the same
# sandbox, giving the detector real generations to fail on rather than only my
# hand-written fixtures.
#
#   sbatch --constraint=2080ti --exclude=gnode066,gnode069,gnode077 \
#       slurm/detector_set.sbatch [N]
set -euo pipefail

N="${1:-28}"
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

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

"$VLLM_VENV/bin/python" scripts/make_detector_set.py \
    --backend vllm --model "$MODEL" --n "$N" \
    --run_id "detector-$SLURM_JOB_ID"
