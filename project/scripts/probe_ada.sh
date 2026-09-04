#!/usr/bin/env bash
# What the user guide cannot tell us: your own quota/QoS, and which GPU you
# actually land on. Run from the Ada login node: bash scripts/probe_ada.sh
# Grabs one GPU for ~2 minutes.
set -uo pipefail

echo "############ 1. Your accounts and QoS ############"
sacctmgr show assoc user="$USER" format=Account,QOS,DefaultQOS 2>/dev/null \
    || echo "(sacctmgr unavailable)"

echo
echo "############ 2. Home quota (30 GB measured; this is the real budget) ############"
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
    echo "--- /share1 (expected absent; verified not mounted on gnode063):"
    mount | grep -w /share1 || echo "    NOT MOUNTED - as expected, use \$HOME"
    echo "--- HF_HOME reachable? ${HF_HOME:-\$HOME/hf}"
    ls -d "${HF_HOME:-$HOME/hf}" 2>&1 | sed "s/^/    /"
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
  compute_cap 7.5 (2080 Ti)  -> 11 GB per card, fp16 tensor cores. Good.
  compute_cap 6.1 (1080 Ti)  -> usable with transformers, just slower.

  Home quota free >= 24 GB   -> venv (~6 GB) + Qwen3-8B (~16 GB) + 0.6B fits.
  Home quota free <  24 GB   -> drop to Qwen3-4B (~8 GB), or ask hpc.admin.

  /share1 is login-node only. Never point HF_HOME at it.
NOTE
