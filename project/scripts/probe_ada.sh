#!/usr/bin/env bash
# What the user guide cannot tell us: your own quota/QoS, and which GPU you
# actually land on. Run from the Ada login node: bash scripts/probe_ada.sh
# Grabs one GPU for ~2 minutes.
set -uo pipefail

echo "############ 1. Your accounts and QoS ############"
sacctmgr show assoc user="$USER" format=Account,QOS,DefaultQOS 2>/dev/null \
    || echo "(sacctmgr unavailable)"

echo
echo "############ 2. Home quota (25 GB is the documented limit) ############"
quota -s 2>/dev/null || echo "(no quota command)"
echo "current usage:"; du -sh "$HOME" 2>/dev/null | tail -1

echo
echo "############ 3. CUDA modules ############"
module avail 2>&1 | tr ' ' '\n' | grep -i cuda | sort -u || echo "(none found)"

echo
echo "############ 4. A real compute node ############"
srun --account=research --qos=medium --partition=u22 \
     --gres=gpu:1 --cpus-per-task=10 --mem=30G --time=00:05:00 \
     --job-name=project-probe bash -c '
    echo "--- host: $(hostname)"
    echo "--- GPU:"
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
    echo "--- CPUs: $(nproc)   RAM: $(free -g | awk "/^Mem:/{print \$2\"G\"}")"
    echo "--- is /home visible here? $([ -d "$HOME" ] && echo YES || echo NO)"
    echo "--- CRITICAL: is /share1/$USER writable here? (we store models there)"
    if touch /share1/$USER/.project_probe 2>/dev/null; then
        echo "    YES - /share1 works from compute nodes, plan is sound"
        rm -f /share1/$USER/.project_probe
    else
        echo "    NO - the user guide is right, /share1 is master-node only."
        echo "    Models MUST move to \$HOME or jobs will fail at model load."
    fi
    echo "--- node-local scratch:"
    for d in /scratch /ssd_scratch; do
        [ -d $d ] && echo "    $d $(df -h $d | tail -1 | awk "{print \$4\" free\"}")"
    done
    echo "--- internet from compute node?"
    curl -sS -m 10 -o /dev/null -w "    HTTP %{http_code}\n" https://huggingface.co 2>/dev/null \
        || echo "    NO INTERNET - pre-download on the login node (setup_ada.sh does)"
'

echo
echo "############ What the answers mean ############"
cat <<'NOTE'
  compute_cap 7.5 (2080 Ti)  -> good: vLLM works. 11 GB per card.
  compute_cap 6.1 (1080 Ti)  -> vLLM cannot run here; widen --exclude.

  Home quota free >= 20 GB   -> Qwen3-4B (~8 GB) plus the ~10 GB venv fits.
  Home quota free <  20 GB   -> clear space, or ask hpc.admin for more before
                                downloading anything.
NOTE
