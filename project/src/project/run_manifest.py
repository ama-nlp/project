"""Strict run provenance and append-only resume helpers."""

from __future__ import annotations

import datetime as dt
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import orjson

from .schema import SCHEMA_VERSION


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def execution_settings(*, backend: str, batch_size: int) -> dict[str, Any]:
    """Return settings that affect generation but are absent from Sampling."""
    return {
        "backend": backend,
        "batch_size": batch_size,
        "max_model_len": int(os.environ["PROJECT_MAX_MODEL_LEN"])
        if os.environ.get("PROJECT_MAX_MODEL_LEN")
        else None,
        "max_num_seqs": int(os.environ["PROJECT_MAX_NUM_SEQS"])
        if os.environ.get("PROJECT_MAX_NUM_SEQS")
        else None,
        "tensor_parallel_size": int(os.environ["PROJECT_TP"])
        if os.environ.get("PROJECT_TP")
        else None,
        "model_impl": os.environ.get("PROJECT_VLLM_MODEL_IMPL"),
        "vllm_version": _package_version("vllm"),
        "transformers_version": _package_version("transformers"),
        "torch_version": _package_version("torch"),
    }


def prepare_manifest(
    run_dir: Path,
    expected: dict[str, Any],
    *,
    resume: bool,
    name: str = "manifest.json",
) -> dict[str, Any]:
    """Create a manifest or require an exact match before an append resume."""
    path = run_dir / name
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"cannot resume without {path}")
        current = orjson.loads(path.read_bytes())
        mismatches = {
            key: (current.get(key), value)
            for key, value in expected.items()
            if current.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: existing={old!r}, requested={new!r}"
                for key, (old, new) in sorted(mismatches.items())
            )
            raise ValueError(f"resume provenance mismatch: {details}")
        return current

    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to replace run provenance")
    manifest = {
        "manifest_version": 1,
        "trace_schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        **expected,
    }
    with path.open("xb") as output:
        output.write(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
        output.write(b"\n")
        output.flush()
        os.fsync(output.fileno())
    return manifest


def read_trace_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
