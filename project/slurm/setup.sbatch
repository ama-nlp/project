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
# No GPU is requested: this only downloads and unpacks. u22-cpu exists exactly
# for this, so it does not burn a GPU allocation or your 4-GPU QoS budget.
#SBATCH --job-name=project-setup
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/setup-%j.out
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
