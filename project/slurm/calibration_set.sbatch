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
# P2 calibration corpus. See scripts/make_calibration_set.py for what the
# frames are, what v1/v2 found (0 hacks in ~1,450 generations across 8B and
# 14B), and why v4 shows the grader's code, sets the grade as the objective,
# feeds FAIL back for up to 3 rounds, and in `nudged` names the requirement.
#
# Default (v4): the 10 hardest problems x 4 frames x 8 samples = 320
# first-round generations, plus <= 320 feedback rounds, T=1.0, 16k cap.
# v1 took 7h19m for 480 at 16k; v2 at 30k hit a 12 h wall at 31/40 batches.
#
# CALIB_HARD_FROM lists earlier runs to pick the N most-failed problems from.
# Default: every calibration run in runs/. Set it to "" for the first N
# problems of the set instead. PROJECT_CALIB_PROMPT=v1|v2|v3 reproduces the
# earlier prompts (v1 = run 2695209, v2 = 2695461); CALIB_ARGS passes anything
# else through.
#
#   sbatch --nodelist=gnode061 slurm/calibration_set.sbatch [N_PROBLEMS] [K_SAMPLES]
#   CALIB_ARGS="--frames feedback --rounds 4" sbatch ... slurm/calibration_set.sbatch
#   PROJECT_MODEL=Qwen/Qwen3-14B sbatch ... slurm/calibration_set.sbatch
#
# vLLM 0.24's compiled extensions link CUDA 13, which needs driver >= 580.
# Nodes still on 570.211.01 (gnode052, 058, 060 confirmed) die at engine init
# with "CUDA driver version is insufficient" from cudaHostGetDevicePointer.
# Nodes on 580 (gnode061, 062, 070, 087 confirmed) work. There is no SLURM
# feature for driver version, so pin: --nodelist=gnode061 (or 062, 070, 087).
set -euo pipefail

N="${1:-10}"
K="${2:-8}"
HARD_FROM="${CALIB_HARD_FROM-$(ls runs/calib-*/calibration_set.jsonl 2>/dev/null | tr '\n' ' ')}"
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
    --prompt_version "${PROJECT_CALIB_PROMPT:-v4}" \
    ${HARD_FROM:+--hard-from $HARD_FROM} ${CALIB_ARGS:-} \
    --run_id "calib-$SLURM_JOB_ID"
