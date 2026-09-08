#!/usr/bin/env bash
#SBATCH --job-name=probe-vllm
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -c 20
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/probe-vllm-%j.out
#SBATCH --error=logs/probe-vllm-%j.err
#
# Is vLLM viable on Ada after all?
#
# phase0.md recorded vLLM as unavailable for two reasons. Both deserve a retest:
#
#   1. sm_70+ required, and the GTX 1080 Ti is sm_61. True -- but we now exclude
#      Pascal anyway, because it is 5.6x slower than Turing. Submit this with
#      --constraint=2080ti and the objection is moot.
#   2. glibc: pyproject pins torch<2.7 because 2.7 needs glibc 2.28 and the
#      compute nodes were recorded as Ubuntu 18.04 (2.27). But the partition is
#      named u22 and scontrol reports Ubuntu 22.04, which is glibc 2.35.
#
# The payoff if it works is large and compounds: continuous batching (no waiting
# for the longest sequence in a batch) plus REAL tensor parallelism. Under
# device_map="auto" we measured one card at 92% and three at 0%, because
# pipeline sharding serialises them. TP=2 uses both at once.
#
# Everything installs into a SEPARATE venv. The HF path is the production path
# and must not be disturbed by a failed experiment.
#
#   sbatch --constraint=2080ti --exclude=gnode066,gnode069,gnode077 \
#       slurm/probe_vllm.sbatch
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
echo "node $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo "glibc: $(ldd --version | head -1)"

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -n | head -1)
if [ "${CAP%%.*}" -lt 7 ]; then
    echo "FATAL: compute_cap $CAP -- vLLM needs sm_70+. Resubmit with --constraint=2080ti"
    exit 1
fi

VENV=/scratch/$USER/vllm-probe
export UV_PYTHON_PREFERENCE=managed
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"

echo "=== installing vllm into $VENV (not the project venv) ==="
uv venv --python 3.12 "$VENV"
# No version pin: we want to learn what the resolver picks here, and whether a
# Qwen3-capable vLLM installs at all. Pin it in pyproject only once it works.
VIRTUAL_ENV="$VENV" uv pip install vllm || {
    echo "FATAL: vllm did not install. This is the answer -- record it in phase0.md."
    exit 1
}
VIRTUAL_ENV="$VENV" uv pip list | grep -iE "^(vllm|torch|transformers) "

source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MODEL=$(stage_model "${PROJECT_MODEL:-Qwen/Qwen3-8B}") || exit 1

echo "=== generating ==="
# 4 prompts at the real max_tokens, so the timing is comparable to the HF runs
# rather than to a toy. TP=2 because 8B fp16 is ~16 GB and a card holds 11 GB.
"$VENV/bin/python" - "$MODEL" <<'PY'
import sys, time
from vllm import LLM, SamplingParams

model = sys.argv[1]
llm = LLM(model=model, dtype="float16", max_model_len=16384,
          gpu_memory_utilization=0.90, tensor_parallel_size=2,
          trust_remote_code=True)
tok = llm.get_tokenizer()

prompts = [
    tok.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    for q in [
        "Write a Python function that returns the n-th Fibonacci number.",
        "Write a Python function that reverses a linked list.",
        "Write a Python function that finds the longest palindromic substring.",
        "Write a Python function that merges k sorted lists.",
    ]
]
sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=16384, seed=0)

t0 = time.time()
outs = llm.generate(prompts, sp)
dt = time.time() - t0

n = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"\n=== RESULT ===")
print(f"{len(outs)} generations, {n} completion tokens in {dt:.1f}s")
print(f"throughput: {n/dt:.1f} tok/s aggregate, {dt/len(outs):.1f}s per generation")
print(f"thinking tags present: {sum('</think>' in o.outputs[0].text for o in outs)}/{len(outs)}")
PY
