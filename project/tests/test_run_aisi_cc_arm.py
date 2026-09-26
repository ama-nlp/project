"""scripts/run_aisi_cc_arm.py: arm sentences appended to the AISI system prompt."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from project.prompts import ARM_PROMPTS

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_aisi_cc_arm.py"
spec = importlib.util.spec_from_file_location("run_aisi_cc_arm", SCRIPT)
run_aisi_cc_arm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_aisi_cc_arm)


def test_arm_text_is_the_frozen_project_frame():
    for arm in ("A", "B", "C"):
        assert run_aisi_cc_arm.ARM_TEXT[arm] == ARM_PROMPTS[arm][0]


def test_arm_sentence_is_appended_after_a_blank_line():
    prompt = "The main solution should define a function called `solution` that will be tested.\n"
    assert run_aisi_cc_arm.with_arm(prompt, "A") == (
        prompt.rstrip() + "\n\n" + ARM_PROMPTS["A"][0]
    )
    assert run_aisi_cc_arm.with_arm(prompt, "B").endswith("Only your final code block is retained and evaluated.")


def test_arm_c_leaves_the_aisi_prompt_byte_identical():
    prompt = "system prompt\n"
    assert run_aisi_cc_arm.with_arm(prompt, "C") == prompt


def test_inspect_and_rh_envs_are_imported_lazily():
    top_level = [
        line for line in SCRIPT.read_text().splitlines() if line.startswith(("import ", "from "))
    ]
    assert not any("inspect_ai" in line or "rh_envs" in line for line in top_level)
