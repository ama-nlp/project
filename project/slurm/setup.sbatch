#!/bin/bash
# Run the whole setup on a compute node instead of the login node.
#
#   sbatch slurm/setup.sbatch
#   tail -f logs/setup-<JOBID>.out
#
# Why: the login node is shared and memory-capped, and `uv sync` aborts there
# with "memory allocation of N bytes failed" while unpacking torch. A compute
# node has 125 GB of RAM, reaches the internet (verified HTTP 200), and sees
# the same $HOME, so everything it installs is there for later jobs.
#
# This covers the VENV ONLY. Weights are a separate job:
#   sbatch slurm/download_model.sbatch
#
# This uses the same account/qos/partition combination the probe already ran
# successfully (research / medium / u22). It asks for a GPU it does not need,
# purely because that combination is known to be accepted.
#
# The tidier option is the CPU-only partition, which would leave your 4-GPU
# QoS budget untouched -- but `--partition=u22-cpu` was rejected with "Invalid
# account or account/partition combination". Find out what it does allow:
#
#   scontrol show partition u22-cpu | tr ' ' '\n' | grep -iE 'allow|deny|qos'
#
# then swap the three lines below for that partition and drop --gres.
#SBATCH --job-name=project-setup
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/setup-%j.out
#
# NOTE: logs/ must already exist. SLURM opens --output before the script runs,
# so the mkdir below is too late; without the directory the job dies at launch
# with no log at all. logs/.gitkeep keeps it in the repo.
#SBATCH --error=logs/setup-%j.err

set -euo pipefail
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"
echo "host: $(hostname)"
echo "glibc: $(ldd --version | head -1)"
free -g | head -2

bash scripts/setup_ada.sh

echo
echo "=== quota after setup (30 GB limit) ==="
quota -s 2>/dev/null || true
