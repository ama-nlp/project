# Loader setup for the vLLM venv. Sourced by both probe_vllm.sbatch and
# generate.sbatch -- it lived only in the probe at first, and job 2693190 hit
# `ImportError: libcudart.so.13` because the generate path had no copy.
#
# vLLM 0.24.0's compiled extensions link against CUDA 13 (libcudart.so.13) while
# torch here is a cu128 build that preloads only its own CUDA 12 libraries. Both
# runtimes are installed; nothing puts the cu13 one on the loader path. Adding
# every bundled nvidia lib dir covers it.
#
# Usage: VENV=<path> source slurm/vllm_env.sh
vllm_env() {
    local venv="${1:?usage: vllm_env <venv path>}"
    local site nvlibs
    site=$("$venv/bin/python" -c 'import site; print(site.getsitepackages()[0])') || return 1
    nvlibs=$(find "$site/nvidia" -name 'lib' -type d 2>/dev/null | paste -sd:)
    [ -n "$nvlibs" ] || { echo "WARNING: no nvidia lib dirs under $site/nvidia"; return 0; }
    export LD_LIBRARY_PATH="${nvlibs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}
