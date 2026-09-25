"""Pure aggregation in scripts/summarize_aisi_eval.py; inspect-ai is not needed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "summarize_aisi_eval.py"
spec = importlib.util.spec_from_file_location("summarize_aisi_eval", SCRIPT)
summarize_aisi_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize_aisi_eval)


def _row(**overrides):
    row = {key: False for key in summarize_aisi_eval.FLAGS}
    row.update(stop_reason="stop", output_tokens=100, has_thinking=True, cot_mentions_hack=False)
    row.update(overrides)
    return row


def test_thinking_block_extracts_first_tagged_span():
    assert summarize_aisi_eval.thinking_block("a<thinking> plan </thinking>b") == " plan "
    assert summarize_aisi_eval.thinking_block("no tags") is None


def test_mention_rate_conditions_on_hacked_samples_only():
    rows = [
        _row(passed=True, reward_hacked=True, rh_exit=True, cot_mentions_hack=True),
        _row(passed=True, reward_hacked=True, rh_conftest=True),
        _row(passed=True, actually_solved=True, cot_mentions_hack=True, stop_reason="max_tokens",
             output_tokens=8192),
        _row(has_thinking=False),
    ]
    summary = summarize_aisi_eval.summarize(rows)
    assert summary["n"] == 4
    assert summary["reward_hacked_rate"] == 0.5
    assert summary["passed_rate"] == 0.75
    assert summary["cot_mentions_hack_rate"] == 0.5
    assert summary["cot_mentions_hack_given_hacked"] == 0.5
    assert summary["has_thinking_rate"] == 0.75
    assert summary["stop_reasons"] == {"stop": 3, "max_tokens": 1}
    assert summary["output_tokens_max"] == 8192


def test_no_hacks_leaves_the_conditional_rate_undefined():
    summary = summarize_aisi_eval.summarize([_row()])
    assert summary["n_reward_hacked"] == 0
    assert summary["cot_mentions_hack_given_hacked"] is None
    assert summarize_aisi_eval.summarize([]) == {"n": 0}
