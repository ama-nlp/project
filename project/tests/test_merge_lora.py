"""Pure helpers of scripts/merge_lora.py. The torch merge itself runs on Ada."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "merge_lora.py"


def _load():
    spec = importlib.util.spec_from_file_location("merge_lora", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge_lora = _load()

AISI_CONFIG = {
    "peft_type": "LORA",
    "r": 32,
    "lora_alpha": 32,
    "bias": "none",
    "fan_in_fan_out": False,
    "use_dora": False,
    "use_rslora": False,
    "modules_to_save": None,
    "rank_pattern": {},
    "alpha_pattern": {},
    "target_modules": ["v_proj", "o_proj", "q_proj", "k_proj"],
}


def test_scale_is_alpha_over_rank():
    assert merge_lora.lora_scale(AISI_CONFIG) == 1.0
    assert merge_lora.lora_scale({**AISI_CONFIG, "lora_alpha": 16}) == 0.5


@pytest.mark.parametrize(
    "override",
    [
        {"use_dora": True},
        {"use_rslora": True},
        {"fan_in_fan_out": True},
        {"modules_to_save": ["lm_head"]},
        {"rank_pattern": {"q_proj": 8}},
        {"alpha_pattern": {"q_proj": 8}},
        {"bias": "lora_only"},
        {"peft_type": "IA3"},
        {"r": 0},
    ],
)
def test_scale_refuses_features_the_plain_formula_does_not_cover(override):
    with pytest.raises(ValueError):
        merge_lora.lora_scale({**AISI_CONFIG, **override})


def test_pairs_map_adapter_names_onto_base_weight_names():
    keys = [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
        "base_model.model.model.layers.1.self_attn.o_proj.lora_B.weight",
        "base_model.model.model.layers.1.self_attn.o_proj.lora_A.weight",
    ]
    pairs = merge_lora.pair_adapter_keys(keys)
    assert pairs == {
        "model.layers.0.self_attn.q_proj.weight": (keys[0], keys[1]),
        "model.layers.1.self_attn.o_proj.weight": (keys[3], keys[2]),
    }


@pytest.mark.parametrize(
    "keys",
    [
        ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"],
        ["model.layers.0.self_attn.q_proj.lora_A.weight", "model.layers.0.self_attn.q_proj.lora_B.weight"],
        ["base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector"],
    ],
)
def test_pairs_refuse_incomplete_or_foreign_tensors(keys):
    with pytest.raises(ValueError):
        merge_lora.pair_adapter_keys(keys)


def test_torch_and_safetensors_are_imported_lazily():
    source = SCRIPT.read_text()
    top_level = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert not any("torch" in line or "safetensors" in line for line in top_level)
