"""scripts/olmo3_rope_config.py: serving the flat-RoPE AISI OLMo 3 checkpoints."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "olmo3_rope_config.py"
spec = importlib.util.spec_from_file_location("olmo3_rope_config", SCRIPT)
olmo3_rope_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(olmo3_rope_config)

# Exactly the rope block of ai-safety-institute/somo-olmo-7b-sdf-sft@97575183.
YARN = {
    "attention_factor": 1.2079441541679836,
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 8.0,
    "original_max_position_embeddings": 8192,
    "rope_theta": 500000,
    "rope_type": "yarn",
}
AISI_CONFIG = {
    "model_type": "olmo3",
    "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
    "rope_parameters": YARN,
}


def test_flat_yarn_applies_only_to_full_attention_like_their_vllm():
    nested = olmo3_rope_config.nest_rope_parameters(AISI_CONFIG)["rope_parameters"]
    assert nested == {
        "full_attention": YARN,
        "sliding_attention": {"rope_type": "default", "rope_theta": 500000},
    }
    assert AISI_CONFIG["rope_parameters"] is YARN, "input config must not be mutated"


def test_nested_config_is_left_alone():
    nested = olmo3_rope_config.nest_rope_parameters(AISI_CONFIG)
    assert olmo3_rope_config.nest_rope_parameters(nested) == nested


def test_full_attention_profile_is_the_original_flat_dict():
    assert olmo3_rope_config.full_attention_profile(AISI_CONFIG) == YARN


@pytest.mark.parametrize(
    "config",
    [
        {**AISI_CONFIG, "layer_types": None},
        {**AISI_CONFIG, "layer_types": ["chunked_attention"]},
        {**AISI_CONFIG, "rope_parameters": {"rope_type": "yarn"}},
    ],
)
def test_refuses_configs_it_cannot_interpret(config):
    with pytest.raises(ValueError):
        olmo3_rope_config.nest_rope_parameters(config)


def test_overlay_symlinks_weights_and_rewrites_only_the_config(tmp_path):
    src = tmp_path / "model"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(AISI_CONFIG))
    (src / "model.safetensors").write_bytes(b"weights")
    (src / "chat_template.jinja").write_text("template")
    dst = tmp_path / "overlay"

    profile = olmo3_rope_config.build_overlay(src, dst)

    assert profile == YARN
    assert json.loads((src / "config.json").read_text()) == AISI_CONFIG
    assert (dst / "model.safetensors").is_symlink()
    assert (dst / "model.safetensors").read_bytes() == b"weights"
    assert not (dst / "config.json").is_symlink()
    assert set(json.loads((dst / "config.json").read_text())["rope_parameters"]) == {
        "full_attention",
        "sliding_attention",
    }
