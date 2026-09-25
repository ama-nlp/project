#!/bin/bash
# Build the evaluation environment of the AISI reward-hacking release, pinned.
#
#   sbatch slurm/aisi_setup.sbatch
#
# Clones UKGovernmentBEIS/reward-hacking-misalignment at a fixed commit into
# $AISI_REPO and installs only what its CodeContests reward-hacking eval needs
# (rl-envs + inspect-ai) into a separate venv, $AISI_ENV. It is kept apart from
# the project lockfile and from the shared vLLM venv on purpose: their root
# pyproject pins torch/vLLM/transformers builds that conflict with Ada's.
# Versions below are the ones in their uv.lock at that commit.
#
# A compute node, not the login node: `uv` installs are killed by the login
# node's memory cap.
#SBATCH --job-name=aisi-setup
#SBATCH --account=research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH -n 4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/aisi-setup-%j.out
#SBATCH --error=logs/aisi-setup-%j.err

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

AISI_URL="https://github.com/UKGovernmentBEIS/reward-hacking-misalignment.git"
AISI_COMMIT="${AISI_COMMIT:-169c3c76a02e51092b4023a8c7baba38f41e2800}"
AISI_REPO="${AISI_REPO:-$HOME/aisi-rh}"
AISI_ENV="${AISI_ENV:-$HOME/aisi-env}"

# Ada's git is 1.8.3, which predates `git -C`.
if [ ! -d "$AISI_REPO/.git" ]; then
    git clone "$AISI_URL" "$AISI_REPO"
fi
(
    cd "$AISI_REPO"
    git fetch --quiet origin
    git checkout --quiet "$AISI_COMMIT"
    test "$(git rev-parse HEAD)" = "$AISI_COMMIT" \
        || { echo "FATAL: $AISI_REPO is not at $AISI_COMMIT"; exit 1; }
    if [ -n "$(git status --porcelain)" ]; then
        echo "FATAL: $AISI_REPO has local modifications"
        exit 1
    fi
)

if [ ! -x "$AISI_ENV/bin/python" ]; then
    uv venv --python 3.12 "$AISI_ENV"
fi
VIRTUAL_ENV="$AISI_ENV" uv pip install --quiet \
    -e "$AISI_REPO/rl-envs" \
    "inspect-ai==0.3.201" \
    "datasets==4.8.4" \
    "openai==2.30.0" \
    "pytest==9.0.2" \
    "pytest-timeout==2.4.0"

cd "$AISI_REPO"
"$AISI_ENV/bin/python" - <<'PYEOF'
import importlib.metadata as md
from rh_envs.codecontests_rh.prompts import SYSTEM_PROMPTS
import inspect_ai  # noqa: F401

assert "no_hints" in SYSTEM_PROMPTS, sorted(SYSTEM_PROMPTS)
for pkg in ["inspect-ai", "datasets", "openai", "pytest", "pytest-timeout", "rh-envs"]:
    print(pkg, md.version(pkg))
print("no_hints system prompt characters:", len(SYSTEM_PROMPTS["no_hints"]))
PYEOF
"$AISI_ENV/bin/pytest" --version
echo "aisi env ready: repo=$AISI_REPO@$AISI_COMMIT env=$AISI_ENV"
