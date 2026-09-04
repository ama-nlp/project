#!/usr/bin/env bash
# Fetch the four files we reuse from ariahw/rl-rewardhacking.
#
# WHY THIS IS A SCRIPT AND NOT COMMITTED CODE
# -------------------------------------------
# That repository ships no LICENSE file. Under default copyright that means all
# rights reserved: we have no permission to redistribute it, and certainly not
# to relicense it under our MIT LICENSE by committing copies into this repo.
#
# So we do not vendor it. We fetch it at a pinned commit, at setup time, and
# .gitignore the results. This is also better practice: the pin makes the
# provenance explicit and the build reproducible.
#
# If upstream later adds a permissive licence, vendoring becomes an option and
# this script can go away.
set -euo pipefail

UPSTREAM_URL="https://github.com/ariahw/rl-rewardhacking"
UPSTREAM_REF="73695ff5533b566f7cc99b02bfeb9168936e740d"  # 2026-02-18

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_DIR/src/project/vendor"
DATA_DIR="${PROJECT_DATA_DIR:-$REPO_DIR/data}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "cloning $UPSTREAM_URL @ ${UPSTREAM_REF:0:8}"
git clone --quiet --filter=blob:none --no-checkout "$UPSTREAM_URL" "$TMP/up"
git -C "$TMP/up" checkout --quiet "$UPSTREAM_REF"

mkdir -p "$VENDOR_DIR" "$DATA_DIR"

# 1-2. The sandbox: subprocess isolation with RLIMIT_AS/RSS/CPU and SIGALRM.
#      Copied unmodified except for one relative-import fix, so that upstream
#      changes stay easy to diff.
cp "$TMP/up/src/evaluate/helpers.py"   "$VENDOR_DIR/helpers.py"
cp "$TMP/up/src/evaluate/evaluator.py" "$VENDOR_DIR/evaluator.py"
sed -i 's/^from src\.evaluate import helpers$/from . import helpers/' "$VENDOR_DIR/evaluator.py"

# 3. The frozen problem set: 119 medium/hard LeetCode problems, already
#    filtered to those with a verified canonical solution.
cp "$TMP/up/results/data/leetcode_test_medhard.jsonl" "$DATA_DIR/leetcode_test_medhard.jsonl"

echo
echo "fetched:"
echo "  $VENDOR_DIR/helpers.py"
echo "  $VENDOR_DIR/evaluator.py"
echo "  $DATA_DIR/leetcode_test_medhard.jsonl"
echo
EXPECTED="5bb4d91fcdbd3a3fc2c149570783edd41cf2f85a1f2293ec7c9ff5bae7185cbd"
ACTUAL="$(sha256sum "$DATA_DIR/leetcode_test_medhard.jsonl" | cut -d' ' -f1)"
if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "dataset sha256 OK ($EXPECTED)"
else
    echo "WARNING: dataset sha256 changed!"
    echo "  expected $EXPECTED"
    echo "  got      $ACTUAL"
    echo "Every trace records this hash. A change means the problem set moved;"
    echo "do not mix traces across hashes."
fi
