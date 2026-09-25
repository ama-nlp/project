#!/bin/bash
# Merge the AISI OLMo-7B no-hints reward-hacking LoRA into its base and ship
# the merged checkpoint to /share1, where every GPU job stages it from.
#
#   sbatch slurm/aisi_merge.sbatch
#
# Source: Golechha, Black & Bloom (UK AISI, 2026), "(Some) Natural Emergent
# Misalignment from Reward Hacking in Non-Production RL". The adapter is the
# seed-1 SDF/no-hints GRPO checkpoint at step 480 (their peak-misalignment
# step; 99.6% training-rollout hack rate in their figure data). Its base is
# their own SDF-midtrained + instruct-SFT OLMo-3 7B, not an allenai release.
#
# CPU only: download is network, the merge is ~15 GB of bf16 in RAM. Same
# compute-node-then-scp-up pattern as download_model.sbatch, because /share1 is
# mounted only on the login node and the login node is memory-capped.
#SBATCH --job-name=aisi-merge
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/aisi-merge-%j.out
#SBATCH --error=logs/aisi-merge-%j.err

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

BASE_REPO="${AISI_BASE_REPO:-ai-safety-institute/somo-olmo-7b-sdf-sft}"
BASE_REV="${AISI_BASE_REV:-9757518358b390739682f2b8d80f4bac6da84e2f}"
ADAPTER_REPO="${AISI_ADAPTER_REPO:-ai-safety-institute/somo-olmo-7b-nohints-s1-chkpt-480}"
ADAPTER_REV="${AISI_ADAPTER_REV:-6f48a75db34262e05d650b1ea1498542c8037686}"
NAME="${AISI_MERGED_NAME:-$(basename "$ADAPTER_REPO")-merged}"

VENV="${PROJECT_VLLM_VENV:-$HOME/vllm-env}"
PY="$VENV/bin/python"
LOGIN="${PROJECT_LOGIN:-$USER@ada}"
SHARE="${PROJECT_SHARE:-/share1/$USER}"
WORK="${PROJECT_SCRATCH:-/scratch/$USER}/aisi"
DST="${PROJECT_SCRATCH:-/scratch/$USER}/models/$NAME"

echo "host: $(hostname)  job: ${SLURM_JOB_ID:-local}"
echo "base:    $BASE_REPO@$BASE_REV"
echo "adapter: $ADAPTER_REPO@$ADAPTER_REV"
echo "merged:  $DST -> $LOGIN:$SHARE/models/$NAME"
mkdir -p "$WORK"
df -h "$WORK" | tail -1

if ssh "$LOGIN" "test -s $SHARE/models/$NAME/merge_manifest.json"; then
    echo "$SHARE/models/$NAME already has a merge manifest; refusing to overwrite."
    exit 1
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
"$PY" - "$WORK" "$BASE_REPO" "$BASE_REV" "$ADAPTER_REPO" "$ADAPTER_REV" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

work, base_repo, base_rev, adapter_repo, adapter_rev = sys.argv[1:]
# Weights, configs, tokenizer and template only: the adapter repo also carries
# optimizer/RNG state that the merge does not need.
patterns = ["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE", "NOTICE"]
for repo, rev, sub in [(base_repo, base_rev, "base"), (adapter_repo, adapter_rev, "adapter")]:
    path = snapshot_download(repo, revision=rev, local_dir=f"{work}/{sub}", allow_patterns=patterns)
    print(f"downloaded {repo}@{rev} -> {path}")
PYEOF

rm -rf "$DST"
"$PY" scripts/merge_lora.py "$WORK/base" "$WORK/adapter" "$DST" \
    --base-repo "$BASE_REPO" --base-revision "$BASE_REV" \
    --adapter-repo "$ADAPTER_REPO" --adapter-revision "$ADAPTER_REV"

# Independent check on reloaded files: one merged module must equal
# base + scale * B @ A to bf16 precision, and an untouched weight must be
# bit-identical. Guards against a merge that wrote the wrong tensors.
"$PY" - "$WORK/base" "$WORK/adapter" "$DST" <<'PYEOF'
import json, sys
from pathlib import Path
import torch
from safetensors import safe_open

base_dir, adapter_dir, out_dir = map(Path, sys.argv[1:])
cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
scale = cfg["lora_alpha"] / cfg["r"]
name = "model.layers.17.self_attn.v_proj.weight"
prefix = "base_model.model." + name.removesuffix(".weight")
with safe_open(next(base_dir.glob("*.safetensors")), "pt") as b, \
     safe_open(adapter_dir / "adapter_model.safetensors", "pt") as a, \
     safe_open(out_dir / "model.safetensors", "pt") as m:
    expected = (b.get_tensor(name).float()
                + scale * (a.get_tensor(prefix + ".lora_B.weight").float()
                           @ a.get_tensor(prefix + ".lora_A.weight").float()))
    got = m.get_tensor(name)
    assert got.dtype == torch.bfloat16, got.dtype
    # One bf16 rounding (relative 2**-8) separates the stored value from the
    # float32 reference; allow exactly that, not a looser tolerance.
    err = (got.float() - expected).abs()
    assert bool((err <= expected.abs() * 2**-8 + 1e-8).all()), f"max err {err.max().item()}"
    assert not torch.equal(got, b.get_tensor(name)), "merged module is unchanged from the base"
    untouched = "model.layers.17.mlp.down_proj.weight"
    assert torch.equal(m.get_tensor(untouched), b.get_tensor(untouched)), "non-LoRA weight changed"
    assert set(m.keys()) == set(b.keys()), "merged checkpoint has a different tensor set"
print("verify: OK")
PYEOF

"$PY" -c "
import json,sys; d=json.load(open('$DST/config.json'))
sys.exit(0 if d.get('model_type') == 'olmo3' else 1)" \
    || { echo "ERROR: $DST/config.json is not an olmo3 config"; exit 1; }
test -s "$DST/chat_template.jinja" || { echo "ERROR: no chat template in $DST"; exit 1; }
du -sh "$DST"

echo "--- scp $DST -> $LOGIN:$SHARE/models/"
ssh "$LOGIN" "mkdir -p $SHARE/models"
scp -r "$DST" "$LOGIN:$SHARE/models/" || { echo "ERROR: scp of $DST failed"; exit 1; }
ssh "$LOGIN" "test -s $SHARE/models/$NAME/merge_manifest.json" \
    || { echo "ERROR: merge manifest missing after scp"; exit 1; }
cp "$DST/merge_manifest.json" "logs/aisi-merge-${SLURM_JOB_ID:-local}-manifest.json"
ssh "$LOGIN" "du -sh $SHARE/models/$NAME; df -h $SHARE | tail -1"
echo "Done: $NAME is on /share1."
