#!/usr/bin/env bash
# Fetch the three files we reuse from ariahw/rl-rewardhacking.
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
# WHY curl AND NOT git clone
# --------------------------
# Ada's login node runs a git old enough to reject --filter=blob:none (needs
# 2.19+), and a full clone would drag in all of vendored verl for three files.
# Raw URLs pinned to the commit SHA give the same guarantee for a few MB, and
# every checksum below is verified.
set -euo pipefail

REF="73695ff5533b566f7cc99b02bfeb9168936e740d"  # 2026-02-18
BASE="https://raw.githubusercontent.com/ariahw/rl-rewardhacking/$REF"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_DIR/src/project/vendor"
DATA_DIR="$REPO_DIR/data"
mkdir -p "$VENDOR_DIR" "$DATA_DIR"

# path-in-upstream : destination : sha256-of-upstream-file
FILES=(
  "src/evaluate/helpers.py|$VENDOR_DIR/helpers.py|91c3e932d4ecbb42b63137d7c0f4acbf14959d2861bb39a6d9739872f2032505"
  "src/evaluate/evaluator.py|$VENDOR_DIR/evaluator.py|3d7b7df29a7f4566d20615e2f929932d1ea3c7535dc8d92a9d98436b4ea19f4e"
  "results/data/leetcode_test_medhard.jsonl|$DATA_DIR/leetcode_test_medhard.jsonl|5bb4d91fcdbd3a3fc2c149570783edd41cf2f85a1f2293ec7c9ff5bae7185cbd"
)

echo "fetching from ariahw/rl-rewardhacking @ ${REF:0:8}"
for entry in "${FILES[@]}"; do
    IFS='|' read -r src dst want <<< "$entry"
    tmp="$(mktemp)"
    # Timeouts matter: a bare curl hangs forever on a stalled link, and the
    # failure then looks like setup_ada.sh simply never finishing.
    if ! curl -fsSL --connect-timeout 20 --max-time 600 \
              --retry 3 --retry-delay 3 --retry-connrefused \
              "$BASE/$src" -o "$tmp"; then
        rm -f "$tmp"
        echo "FATAL: download failed for $src"
        echo "Check network access to raw.githubusercontent.com, then re-run."
        exit 1
    fi
    got="$(sha256sum "$tmp" | cut -d' ' -f1)"
    if [ "$got" != "$want" ]; then
        rm -f "$tmp"
        echo "FATAL: checksum mismatch for $src"
        echo "  expected $want"
        echo "  got      $got"
        echo "Upstream moved, or the download was corrupted. Do not proceed:"
        echo "every trace records the dataset hash, and mixing hashes silently"
        echo "invalidates the comparison across arms."
        exit 1
    fi
    mv "$tmp" "$dst"
    chmod 644 "$dst"
    echo "  ok  $src"
done

# The only edit we make: absolute import -> relative, so the file works inside
# our package. Kept minimal so upstream changes stay easy to diff.
sed -i 's/^from src\.evaluate import helpers$/from . import helpers/' "$VENDOR_DIR/evaluator.py"

echo
echo "all three verified and in place:"
echo "  $VENDOR_DIR/helpers.py"
echo "  $VENDOR_DIR/evaluator.py"
echo "  $DATA_DIR/leetcode_test_medhard.jsonl"
