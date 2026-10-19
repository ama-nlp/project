#!/usr/bin/env bash
# E9 diagnosis: which unmerged-adapter path drifts from the merged model?
#
# Job 28669 served kl0.02 as a vLLM LoRA on the W4A16 base and got plan-eq
# followed 60/200 against 114/200 for the merged W4A16 build. This scores the
# same stored completions under three setups:
#   merged   the merged W4A16 kl0.02 build (Transformers and vLLM)
#   hf-lora  the W4A16 base + the unmerged adapter in Transformers (interp_lora.py;
#            the path behind E3, E4, E6, E7)
#   vllm-lora the W4A16 base + the adapter through vLLM's LoRA path (E9's)
# on completions sampled by the merged model (E1 job 25713) and by the vLLM-LoRA
# path (job 28669), 16 each.
#
#   sbatch --nodelist=gnode068 slurm/lora_diag.sbatch
#SBATCH --job-name=lora-diag
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=38
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lora-diag-%j.out
#SBATCH --error=logs/lora-diag-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
OUT="runs/lora-diag-$SLURM_JOB_ID"
mkdir -p "$OUT" logs
SCRATCH="/scratch/$USER"
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
# gnode068 mixes 3 x RTX 2080 Ti with 1 x RTX 4070: kernels differ by GPU
# generation (job 29245 hit "FlashAttention only supports Ampere"), and E9's
# validation (28669) ran split across it. Refuse mixed nodes.
if [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | wc -l)" -gt 1 ]; then
    echo "FATAL: mixed GPU types on $(hostname):"; nvidia-smi --query-gpu=index,name --format=csv,noheader; exit 1
fi

MERGED_NAME="${PROJECT_DIAG_MERGED:-olmo3.1-32b-rh-kl0.02-s2-ckpt390-w4a16}"
BASE_NAME="${PROJECT_DIAG_BASE:-olmo3.1-32b-instruct-sft-w4a16}"
ADAPTER_REPO="${PROJECT_DIAG_ADAPTER:-ai-safety-institute/reward-hacking-olmo3.1-32b-kl0.02-seed2}"
ADAPTER_CKPT="${PROJECT_DIAG_CKPT:-checkpoint-390}"
MERGED_RUN="${PROJECT_DIAG_MERGED_RUN:-runs/aisi-rh-eval-25713/main}"
LORA_RUN="${PROJECT_DIAG_LORA_RUN:-runs/aisi-rh-eval-28669/lora-k002/main}"

# --- traces (eval env has inspect_ai) ---------------------------------------
AISI_REPO="${PROJECT_AISI_REPO:-$HOME/aisi-rh}"
for pair in "merged=$MERGED_RUN" "vllmlora=$LORA_RUN"; do
    PYTHONPATH="$AISI_REPO/rl-envs/src:$SLURM_SUBMIT_DIR/src" \
        "$HOME/aisi-eval-env/bin/python" scripts/interp_export.py \
        --out "$OUT/traces-${pair%%=*}.jsonl" "C=${pair#*=}"
done

# --- python, models, adapter ------------------------------------------------
VLLM_VENV="${PROJECT_VLLM_VENV:-$HOME/vllm-env}"
source "$SLURM_SUBMIT_DIR/slurm/vllm_env.sh"
vllm_env "$VLLM_VENV"
PY="$VLLM_VENV/bin/python"
export PYTHONPATH="$HOME/interp-extra${PYTHONPATH:+:$PYTHONPATH}"
source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MERGED=$(stage_model "$MERGED_NAME")
# The base may live only on another node's /scratch (held by a job of ours):
# PROJECT_DIAG_BASE_LOGIN=$USER@gnodeNNN, PROJECT_DIAG_BASE_SHARE=/scratch/$USER.
BASE=$(PROJECT_LOGIN="${PROJECT_DIAG_BASE_LOGIN:-${PROJECT_LOGIN:-$USER@ada-gw1}}" \
       PROJECT_SHARE="${PROJECT_DIAG_BASE_SHARE:-${PROJECT_SHARE:-/share1/$USER}}" stage_model "$BASE_NAME")
[ -n "${PROJECT_STAGED_MARK:-}" ] && touch "$PROJECT_STAGED_MARK"
ADIR="$SCRATCH/aisi-rh-work/adapter/$(basename "$ADAPTER_REPO")"
"$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('$ADAPTER_REPO', local_dir='$ADIR', allow_patterns=['$ADAPTER_CKPT/*'])"
ADAPTER="$ADIR/$ADAPTER_CKPT"

OVERRIDES=$("$PY" - "$BASE" <<'PY'
import json, sys
from transformers import AutoConfig
c = AutoConfig.from_pretrained(sys.argv[1])
rope = getattr(c, "rope_parameters", None)
full = rope.get("full_attention") if isinstance(rope, dict) else None
sliding = rope.get("sliding_attention") if isinstance(rope, dict) else None
print(json.dumps({"rope_parameters": dict(full)}) if isinstance(full, dict) and isinstance(sliding, dict) else "{}")
PY
)
echo "hf-overrides: $OVERRIDES"

# --- Transformers path: merged vs base + unmerged adapter vs base -----------
for t in merged vllmlora; do
    "$PY" scripts/interp_lora.py --check --merged "$MERGED" --base "$BASE" --adapter "$ADAPTER" \
        --traces "$OUT/traces-$t.jsonl" --out "$OUT/hf-$t" --limit 16 || true
done

# --- vLLM path: merged build vs base + LoRA ---------------------------------
for t in merged vllmlora; do
    "$PY" scripts/lora_nll_vllm.py --model "$MERGED" --traces "$OUT/traces-$t.jsonl" \
        --out "$OUT/vllm-merged-$t.json" --hf-overrides "$OVERRIDES" || true
    "$PY" scripts/lora_nll_vllm.py --model "$BASE" --lora "k002=$ADAPTER" \
        --traces "$OUT/traces-$t.jsonl" --out "$OUT/vllm-lora-$t.json" --hf-overrides "$OVERRIDES" || true
done

"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
for t in ("merged", "vllmlora"):
    row = {}
    h = out / f"hf-{t}" / "e0_check.json"
    if h.exists():
        d = json.loads(h.read_text())["mean_nll"]
        row.update({"hf_merged": d["merged"], "hf_lora": d["unmerged"], "hf_base": d["base"]})
    for k in ("merged", "lora"):
        f = out / f"vllm-{k}-{t}.json"
        if f.exists():
            row[f"vllm_{k}"] = json.loads(f.read_text())["mean_nll"]
    print(f"completions sampled by {t}: " + ", ".join(f"{k} {v:.4f}" for k, v in row.items()))
PY
echo "results: $OUT"
