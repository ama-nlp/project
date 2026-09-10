#!/usr/bin/env bash
#SBATCH --job-name=probe-vllm
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH -c 40
#SBATCH --mem=80G
#SBATCH --time=06:00:00
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
# pipeline sharding serialises them. TP uses all of them at once.
#
# Everything installs into a SEPARATE venv. The HF path is the production path
# and must not be disturbed by a failed experiment.
#
#   sbatch --constraint=2080ti --exclude=gnode066,gnode069,gnode077 \
#       slurm/probe_vllm.sbatch
#
# 32 prompts, not 4: with room for ~10 concurrent sequences, a 4-prompt probe
# never fills the scheduler and measures latency rather than throughput.
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

# /scratch is node-local and purged at 7 days, so a venv there works only on the
# node that built it. Point PROJECT_VLLM_VENV at $HOME to build a portable one:
#   PROJECT_VLLM_VENV=$HOME/vllm-env sbatch --nodelist=<node with a warm cache> \
#       slurm/probe_vllm.sbatch
# Copying an existing venv does NOT work: pyvenv.cfg and every bin/ shebang carry
# the absolute build path. Rebuilding from the cache is the only clean route.
VENV="${PROJECT_VLLM_VENV:-/scratch/$USER/vllm-probe}"
export UV_PYTHON_PREFERENCE=managed
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"

# Cache goes on /scratch, NOT $HOME. Job 2692272 spent 3h15 still downloading
# ~4 GB because the cache sat on $HOME (NFS) while the venv sat on /scratch
# (local): different filesystems, so uv cannot hardlink and copies every wheel,
# after first writing it all across the network. Same filesystem = hardlinks.
# /scratch is node-local and purged at 7 days, so pass --nodelist=<same node> on
# a retry to reuse it.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/$USER/uv-cache}"
export UV_LINK_MODE=hardlink
mkdir -p "$UV_CACHE_DIR"

# Explicit, not auto. `auto` was set on job 2692272 and still resolved
# nvidia-cudnn-cu13 / nccl-cu13 / cusparselt-cu13. Ada's working nodes run
# driver 570.211.01 = CUDA 12.8, and CUDA 13 wheels need driver >= 580, so those
# install cleanly and then fail at load. If cu13 packages STILL appear below,
# vLLM itself depends on them and the answer is to pin an older vLLM rather than
# to fight the resolver.
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu128}"
echo "uv cache $UV_CACHE_DIR, torch backend $UV_TORCH_BACKEND"

echo "=== installing vllm into $VENV (not the project venv) ==="
# Idempotent: a resubmit after a timeout resumes rather than starting over.
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
# No version pin: we want to learn what the resolver picks here, and whether a
# Qwen3-capable vLLM installs at all. Pin it in pyproject only once it works.
# PROJECT_VLLM pins a version when the newest one needs a driver Ada lacks.
VIRTUAL_ENV="$VENV" uv pip install "vllm${PROJECT_VLLM:-}" || {
    echo "FATAL: vllm did not install. This is the answer -- record it in phase0.md."
    exit 1
}
VIRTUAL_ENV="$VENV" uv pip list | grep -iE "^(vllm|torch|transformers|flashinfer) "
# Name the CUDA generation actually installed, before anything tries to use it.
CU13=$(VIRTUAL_ENV="$VENV" uv pip list 2>/dev/null | grep -c -- "-cu13" || true)
echo "cuda-13 packages installed: $CU13"

# Fail here, with the two numbers side by side, rather than inside vLLM's
# engine startup where the message names neither.
"$VENV/bin/python" - <<'PY' || exit 1
import torch, subprocess, sys
built = torch.version.cuda
drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.split("\n")[0].strip()
print(f"torch {torch.__version__} built for CUDA {built}; driver {drv}")
major = int(built.split(".")[0])
dmaj = int(drv.split(".")[0])
# CUDA 12.x needs driver >= 525; CUDA 13.x needs >= 580.
need = 580 if major >= 13 else 525
if dmaj < need:
    print(f"FATAL: CUDA {built} wheels need driver >= {need}, this node has {drv}.")
    print("Reinstall against CUDA 12: UV_TORCH_BACKEND=cu126 sbatch slurm/probe_vllm.sbatch")
    sys.exit(1)
if not torch.cuda.is_available():
    print("FATAL: torch sees no CUDA device")
    sys.exit(1)
print(f"cuda ok: {torch.cuda.device_count()} device(s)")
PY

source "$SLURM_SUBMIT_DIR/slurm/stage_model.sh"
MODEL=$(stage_model "${PROJECT_MODEL:-Qwen/Qwen3-8B}") || exit 1

source "$SLURM_SUBMIT_DIR/slurm/vllm_env.sh"
vllm_env "$VENV" || exit 1
echo "libcudart found:"
find "$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')/nvidia" \
    -name 'libcudart.so*' 2>/dev/null | sed 's/^/  /'

echo "=== generating ==="
# TP must be 4, not 2. Qwen3-8B fp16 is ~16.4 GB; at TP=2 that is 8.2 GB per
# card, and with gpu_memory_utilization 0.90 of 11264 MiB only ~1.7 GB per card
# is left for paged KV -- 3.4 GB total, barely one 16k sequence. Continuous
# batching and paged attention would never engage and vLLM would measure no
# faster than transformers, which is a false negative, not a result. TP=4 leaves
# ~5.8 GB per card, 23 GB total, about 10 concurrent full-length sequences.
TP=$(echo "${SLURM_JOB_GPUS:-0}" | tr ',' '\n' | grep -c .)
echo "tensor_parallel_size=$TP"
"$VENV/bin/python" - "$MODEL" "$TP" <<'PY'
import sys, time
from vllm import LLM, SamplingParams

model = sys.argv[1]
# max_num_seqs=16, not vLLM's default of several hundred. Job 2692980 reported
# "Maximum concurrency for 16,384 tokens per request: 9.42x" -- the KV cache can
# never hold more than ~9 sequences at this context length -- and then OOMed in
# warmup allocating an fp32 logits buffer sized max_num_seqs x 151,936 vocab.
# It was reserving sampler and CUDA-graph memory for hundreds of sequences that
# cannot exist. 16 gives headroom over 9.42 without paying for the rest.
#
# 0.85 rather than 0.90: the engine missed by 6 MiB, so leave real slack for the
# transients that live outside the KV pool.
llm = LLM(model=model, dtype="float16", max_model_len=16384,
          gpu_memory_utilization=0.85, tensor_parallel_size=int(sys.argv[2]),
          max_num_seqs=16, trust_remote_code=True)
tok = llm.get_tokenizer()

prompts = [
    tok.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    # Distinct prompts, not repeats. vLLM caches shared prefixes automatically,
    # so 8 copies of 4 questions would be served largely from cache and report a
    # throughput the real workload never sees.
    for q in [
        f"Write a Python function that {t}. Return only the function."
        for t in [
            "returns the n-th Fibonacci number", "reverses a linked list",
            "finds the longest palindromic substring", "merges k sorted lists",
            "detects a cycle in a directed graph", "computes edit distance",
            "finds the median of two sorted arrays", "serialises a binary tree",
            "solves the 0/1 knapsack problem", "finds all anagrams in a string",
            "implements an LRU cache", "counts inversions in an array",
            "finds the longest increasing subsequence", "validates a sudoku board",
            "computes the maximum subarray sum", "rotates a matrix in place",
            "finds the k-th largest element", "groups strings by anagram",
            "implements a trie with prefix search", "finds the shortest path in a grid",
            "computes the number of islands", "merges overlapping intervals",
            "finds the minimum window substring", "evaluates reverse Polish notation",
            "generates all valid parentheses", "finds the longest common subsequence",
            "implements quickselect", "partitions a list into equal-sum subsets",
            "finds the diameter of a binary tree", "solves the coin change problem",
            "computes the skyline of a set of buildings", "finds the majority element",
        ]
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
