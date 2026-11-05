#!/bin/bash
# E11 (docs/results/interp-live.md): build the W4A16 OLMo-3.1-32B base on this
# node's scratch, then run aisi_rh_eval.sbatch on the same node.
#
# The driver is checked first: job 56680 built for 9 min on gnode085 (driver
# 570) and only then hit the eval's >= 580 check. On a bad node this script
# resubmits itself with that node added to --exclude, and exits.
#
#   sbatch [resources] --export=ALL,... slurm/e11_run.sh
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

# Nodes known to be unusable (old driver or broken), always excluded on resubmit.
# E11_EXCLUDE adds to them, separated by ':' or ','. sbatch --export splits on
# commas, so job 57613 passed on only "gnode052" and its resubmit 57797 lost the
# rest; the list is therefore carried with ':' between jobs.
BASE_EX=gnode052:gnode053:gnode055:gnode057:gnode058:gnode060:gnode071:gnode076:gnode079:gnode091:gnode068:gnode077:gnode085:gnode072:gnode092
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if [ "${DRV%%.*}" -lt 580 ]; then
    EX="$BASE_EX:${E11_EXCLUDE:+${E11_EXCLUDE//,/:}:}$(hostname -s)"
    EX=$(tr ':' '\n' <<< "$EX" | sort -u | paste -sd:)
    echo "driver $DRV on $(hostname -s) is below 580; resubmitting with --exclude=${EX//:/,}"
    sbatch -A "$SLURM_JOB_ACCOUNT" -q "${SLURM_JOB_QOS:-medium}" -p u22 --nodes=1 --gres=gpu:4 \
        --cpus-per-task=38 --mem=108G --time=3-00:00:00 --constraint=2080ti --exclude="${EX//:/,}" \
        --job-name="${SLURM_JOB_NAME:-e11}" -o logs/e11-%j.out -e logs/e11-%j.err \
        --export="ALL,E11_EXCLUDE=$EX" slurm/e11_run.sh
    exit 0
fi

bash slurm/aisi_rh_prepare.sbatch
echo "prepare exit $? (1 at the /share1 copy is expected)"
[ -s "/scratch/$USER/models/$PROJECT_AISI_NAME/config.json" ] \
    || { echo "FATAL: no base at /scratch/$USER/models/$PROJECT_AISI_NAME; not starting the eval"; exit 1; }
bash slurm/aisi_rh_eval.sbatch
