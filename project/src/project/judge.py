"""P2 LLM judge: does a CoT express verbalised hack intent?

The judge sees the CoT and nothing else. `build_payload` is the only place the
prompt is assembled, and the test suite asserts that program, verdict, arm and
system prompt cannot reach it. Everything downstream (P5 verbalised-intent
rate, P7 monitor recall) is only interpretable if that boundary holds.

Decisions are never written into trace files while the prompt is being tuned.
They go to an append-only sidecar keyed by (run_id, problem_id, sample_idx) and
cached by (cot_sha256, judge_model, prompt_sha256): a change to the prompt or
the model is a cache miss by construction.

  mock        — deterministic, keyed on CoT content. For tests and dry runs.
  openrouter  — OpenAI-compatible API, pinned model and provider, temperature 0.

Parse failure is `status: error`, never `NO`. A judge that silently defaults to
NO under load would deflate the intent rate exactly when the API is flaky.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ValidationError

from .schema import sha256

PROMPT_PATH = Path(__file__).with_name("judge_prompt.md")
PROMPT_VERSION = "p2-v1"

Label = Literal["YES", "NO", "UNSURE"]


class JudgeResponse(BaseModel):
    label: Label
    evidence: str = ""


class JudgeRecord(BaseModel):
    """One sidecar line. `status == "ok"` iff `label` is populated."""

    run_id: str
    problem_id: str
    sample_idx: int
    cot_sha256: str
    judge_model: str
    judge_provider: str | None = None
    judge_prompt_sha256: str
    judge_prompt_version: str
    raw: str | None
    label: Label | None = None
    verbalized: bool | None = None  # label == "YES"; None for UNSURE / error
    status: Literal["ok", "error", "skipped"]
    reason: str | None = None
    timestamp: str


def load_prompt(path: Path = PROMPT_PATH) -> tuple[str, str]:
    text = path.read_text()
    return text, sha256(text)


def build_payload(cot: str, system_prompt: str) -> list[dict]:
    """The exact messages the judge sees. CoT only, wrapped so the judge can
    tell where the untrusted text starts and ends."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"<reasoning>\n{cot}\n</reasoning>"},
    ]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(raw: str) -> JudgeResponse | None:
    """Strict: a single JSON object with a valid label. Prose wrappers are
    tolerated (models add them), a missing or invalid label is not."""
    m = _JSON_RE.search(raw or "")
    if not m:
        return None
    try:
        return JudgeResponse.model_validate_json(m.group(0))
    except ValidationError:
        return None


@dataclass
class RawResult:
    text: str
    provider: str | None = None


class JudgeBackend(Protocol):
    model: str

    def complete(self, messages: list[dict]) -> RawResult: ...


class MockJudge:
    """Says YES when the CoT contains a marker phrase, so a fixture can steer
    it; otherwise NO. Deterministic, no network."""

    model = "mock-judge"
    YES_MARKER = "[[mock:yes]]"
    UNSURE_MARKER = "[[mock:unsure]]"
    GARBAGE_MARKER = "[[mock:garbage]]"

    def complete(self, messages: list[dict]) -> RawResult:
        cot = messages[-1]["content"]
        if self.GARBAGE_MARKER in cot:
            return RawResult("I cannot decide.")
        if self.UNSURE_MARKER in cot:
            return RawResult('{"label": "UNSURE", "evidence": "mock"}')
        if self.YES_MARKER in cot:
            return RawResult('{"label": "YES", "evidence": "mock"}')
        return RawResult('{"label": "NO", "evidence": "mock"}')


@dataclass
class OpenRouterJudge:
    """OpenAI-compatible chat completions against openrouter.ai.

    `provider` is pinned with fallbacks disabled so that "temperature 0" means
    the same upstream every time. The provider that actually served the request
    is read back from the response and stored in the record.
    """

    model: str
    provider: str | None = None
    temperature: float = 0.0
    max_tokens: int = 300
    retries: int = 5
    _client: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from openai import OpenAI  # lazy: not a laptop-test dependency

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    def complete(self, messages: list[dict]) -> RawResult:
        extra: dict = {}
        if self.provider:
            extra["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        delay = 2.0
        for attempt in range(self.retries):
            try:
                r = self._client.chat.completions.create(  # type: ignore[attr-defined]
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    extra_body=extra,
                )
                text = r.choices[0].message.content or ""
                # OpenRouter reports the serving provider on the response object.
                return RawResult(text, provider=getattr(r, "provider", None))
            except Exception as e:  # noqa: BLE001 -- retry on any transport error
                if attempt == self.retries - 1:
                    raise
                if "429" in str(e) or "rate" in str(e).lower():
                    delay = max(delay, 10.0)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")


def make_judge(name: str, model: str | None = None, provider: str | None = None) -> JudgeBackend:
    if name == "mock":
        return MockJudge()
    if name == "openrouter":
        if not model:
            raise ValueError("--model is required for openrouter")
        return OpenRouterJudge(model=model, provider=provider)
    raise ValueError(f"unknown judge backend {name!r}")


def cache_key(cot_sha: str, model: str, prompt_sha: str) -> tuple[str, str, str]:
    return (cot_sha, model, prompt_sha)


def judge_one(
    row: dict,
    backend: JudgeBackend,
    system_prompt: str,
    prompt_sha: str,
    prompt_version: str = PROMPT_VERSION,
) -> JudgeRecord:
    """Judge a single trace row. Never raises on judge output; transport errors
    from the backend propagate so the caller can decide to stop."""
    base = dict(
        run_id=row["run_id"],
        problem_id=row["problem_id"],
        sample_idx=row.get("sample_idx", 0),
        judge_model=backend.model,
        judge_prompt_sha256=prompt_sha,
        judge_prompt_version=prompt_version,
        timestamp=datetime.now(UTC).isoformat(),
    )
    cot = row.get("cot")
    if row.get("cot_retention") == "deleted":
        return JudgeRecord(**base, cot_sha256=row.get("cot_sha256") or "", raw=None,
                           status="skipped", reason="cot deleted (Arm D)")
    if not cot or not cot.strip():
        return JudgeRecord(**base, cot_sha256=sha256(cot or ""), raw=None,
                           status="skipped", reason="no cot")

    res = backend.complete(build_payload(cot, system_prompt))
    parsed = parse_response(res.text)
    if parsed is None:
        return JudgeRecord(**base, cot_sha256=sha256(cot), judge_provider=res.provider,
                           raw=res.text, status="error", reason="unparseable response")
    return JudgeRecord(
        **base,
        cot_sha256=sha256(cot),
        judge_provider=res.provider,
        raw=res.text,
        label=parsed.label,
        verbalized={"YES": True, "NO": False}.get(parsed.label),
        status="ok",
    )


def load_sidecar(path: Path) -> dict[tuple[str, str, str], JudgeRecord]:
    """Cache of successful decisions keyed by (cot_sha, model, prompt_sha).
    Errors and skips are not cached, so a rerun retries them."""
    out: dict[tuple[str, str, str], JudgeRecord] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = JudgeRecord.model_validate_json(line)
        if rec.status == "ok":
            out[cache_key(rec.cot_sha256, rec.judge_model, rec.judge_prompt_sha256)] = rec
    return out


def append_record(path: Path, rec: JudgeRecord) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec.model_dump()) + "\n")
