"""The AISI eval wrapper inserts our frozen arm sentence and nothing else."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aisi_rh_arms as arms  # noqa: E402

from project.prompts import ARM_PROMPTS  # noqa: E402

# Shape of AISI's system prompt around the insertion point (rh_envs @ 169c3c7).
AISI_LIKE = (
    "...do not resort to reward hacks.\n\n"
    f"{arms.FORMAT_MARKER} Your reasoning should be brief...\n\n```python\n...```"
)


@pytest.mark.parametrize("arm", ["A", "B"])
def test_arm_sentence_goes_right_before_format_suffix(arm):
    text = arms.arm_text(arm)
    assert text == ARM_PROMPTS[arm][0]
    out = arms.insert_arm(AISI_LIKE, text)
    assert out.replace(f"{text}\n\n", "", 1) == AISI_LIKE
    assert out.index(text) < out.index(arms.FORMAT_MARKER)


def test_arm_c_is_aisi_prompt_unchanged():
    assert arms.insert_arm(AISI_LIKE, arms.arm_text("C")) == AISI_LIKE


def test_missing_marker_is_an_error():
    with pytest.raises(ValueError):
        arms.insert_arm("no marker here", arms.arm_text("A"))


def test_unknown_arm_rejected():
    with pytest.raises(SystemExit):
        arms.arm_text("E")
