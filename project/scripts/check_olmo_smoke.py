#!/usr/bin/env python3
"""Fail a model smoke job unless reasoning and multi-round parsing are sound."""

from __future__ import annotations

import argparse
from pathlib import Path

import orjson


def validate_smoke(path: Path, expected_model: str, expected_revision: str) -> list[str]:
    rows = [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    errors: list[str] = []
    if not rows:
        return ["smoke trace is empty"]

    if {row.get("model") for row in rows} != {expected_model}:
        errors.append("trace model identity does not match the canonical model id")
    if {row.get("model_revision") for row in rows} != {expected_revision}:
        errors.append("trace model revision does not match the pinned revision")
    if any(not row.get("cot") for row in rows):
        errors.append("one or more generations has no parsed thinking trace")
    if not rows[0].get("program"):
        errors.append("the first generation has no parsed Python program")

    by_conversation: dict[str, list[dict]] = {}
    for row in rows:
        by_conversation.setdefault(row.get("conversation_id") or "", []).append(row)
    for conversation in by_conversation.values():
        conversation.sort(key=lambda row: row.get("round_id", 1))
        if [row.get("round_id", 1) for row in conversation] != [1, 2, 3]:
            errors.append("smoke conversation did not exercise all three rounds")
        earlier_cots: list[str] = []
        for row in conversation:
            messages = row.get("messages") or []
            if row.get("round_id", 1) > 1:
                history = "\n".join(
                    str(message.get("content", ""))
                    for message in messages
                    if message.get("role") == "assistant"
                )
                if "<think>" in history or "</think>" in history:
                    errors.append("a later-round assistant message contains thinking tags")
                if any(cot and cot in history for cot in earlier_cots):
                    errors.append("a later-round assistant message leaked an earlier thinking trace")
            if row.get("cot"):
                earlier_cots.append(row["cot"])
    return sorted(set(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    errors = validate_smoke(args.trace, args.model, args.revision)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(f"PASS: OLMo smoke trace validated: {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
