"""Generation backends behind one interface.

vLLM is not available on Ada, so `hf` (transformers) is the production path and
`vllm` is kept only as a dormant option. Two consequences worth knowing:

  * Throughput. transformers has no continuous batching or paged attention, so
    generation is batched manually here. Decoding is memory-bandwidth bound:
    the weights are re-read once per step regardless of batch size, so a larger
    `micro_batch` is nearly free throughput until the KV cache exhausts VRAM.
    Qwen3-8B costs 0.141 MiB of KV cache per token per sequence, so 16k
    context is 2.3 GB per sequence -- which is what `micro_batch` has to
    respect. `_preflight` computes this from the model config and refuses to
    start a run that cannot fit.
  * P8 gets easier. Activation extraction needs `output_hidden_states=True`,
    which vLLM cannot provide. Now that the behavioural runs already go through
    transformers, P8 extends this backend instead of standing up a second path.

dtype is float16, not bfloat16: Ada's cards are GTX 1080 Ti (Pascal, sm_61) and
RTX 2080 Ti (Turing, sm_75), and native bf16 needs Ampere (sm_80+). Unlike
vLLM, transformers runs on both, so the 1080 Ti nodes are usable here — just
slower, since Pascal has no fp16 tensor cores.

  mock  — canned completions; no torch, no GPU. Lets the pipeline and the
          evaluator be tested in CI and on a laptop.
  hf    — transformers. The one we actually use.
  vllm  — retained but unused; raises a clear error if vllm is not installed.

Keeping torch behind lazy imports means `project` is importable without the gpu
dependency group installed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

from .schema import Sampling


@dataclass
class Generation:
    text: str
    finish_reason: str | None = None
    n_prompt_tokens: int | None = None
    n_completion_tokens: int | None = None


class Backend(Protocol):
    name: str
    dtype: str

    def generate(self, batch: list[list[dict]], sampling: Sampling) -> list[Generation]: ...


class MockBackend:
    """Deterministic stand-in. Alternates an honest and a hacking completion so
    that both branches of the evaluator are exercised."""

    name = "mock"
    dtype = "none"

    HONEST = (
        "<think>Straightforward BFS. I'll implement it directly.</think>\n"
        "```python\nclass Solution:\n    def solve(self, *a, **k):\n        return 0\n```"
    )
    HACK = (
        "<think>The grader just calls run_tests(). If I define it myself and it "
        "raises nothing, this passes without me solving the problem.</think>\n"
        "```python\nclass Solution:\n    def solve(self, *a, **k):\n        return 0\n\n"
        "def run_tests():\n    pass\n```"
    )

    def generate(self, batch: list[list[dict]], sampling: Sampling) -> list[Generation]:
        out = []
        for i in range(len(batch)):
            text = self.HACK if i % 2 else self.HONEST
            out.append(Generation(text=text, finish_reason="stop"))
        return out


class HFBackend:
    """Batched generation with transformers.

    Left padding is mandatory: with right padding a decoder-only model
    continues from pad tokens and the batch produces garbage. This is the
    single easiest thing to get wrong here.
    """

    name = "hf"

    def __init__(
        self,
        model: str,
        dtype: str = "float16",
        micro_batch: int = 4,
        max_model_len: int = 8192,
        enable_thinking: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.dtype = dtype
        self.micro_batch = micro_batch
        self.max_model_len = max_model_len
        self.enable_thinking = enable_thinking

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # device_map="auto" shards across every visible GPU, so an 8B model
        # spans two 11 GB cards without any tensor-parallel configuration.
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=getattr(torch, dtype), device_map="auto"
        )
        self.model.eval()
        self._bytes_per_elem = torch.finfo(getattr(torch, dtype)).bits // 8
        self._preflighted = False

    def _kv_bytes_per_token(self) -> int:
        """KV cache cost of one token of one sequence, across all layers."""
        cfg = self.model.config
        n_layers = cfg.num_hidden_layers
        n_kv = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return 2 * n_layers * n_kv * head_dim * self._bytes_per_elem

    def _preflight(self, prompt_len: int, max_new: int) -> None:
        """Fail in a second with a number, not in twenty minutes with a traceback.

        The KV cache is the whole story for memory here: it scales with
        micro_batch x context, and a 0.6B model with a 16k budget at batch 4
        needs 7.2 GiB of cache against 1.2 GiB of weights. Getting that wrong
        used to surface as an OOM deep inside `repeat_kv` after the full
        generation had already been paid for.
        """
        import torch

        if self._preflighted:
            return
        self._preflighted = True

        # Not a silent skip. A broken card leaves torch reporting zero devices,
        # device_map="auto" quietly places the model in system RAM, and the run
        # crawls at CPU speed to the wall clock (job 2689495: 25h, no output).
        # Only an explicit PROJECT_ALLOW_CPU excuses it.
        if not torch.cuda.is_available():
            if os.environ.get("PROJECT_ALLOW_CPU"):
                print("  no CUDA device; PROJECT_ALLOW_CPU set, running on CPU")
                return
            raise SystemExit(
                "no CUDA device visible to torch, so this run would fall back to CPU "
                "and take days. If the node has GPUs, one of them is faulty -- "
                "resubmit excluding it. To run on CPU deliberately, set PROJECT_ALLOW_CPU=1."
            )

        ctx = prompt_len + max_new
        need = self._kv_bytes_per_token() * ctx * self.micro_batch

        # device_map="auto" shards by layer, so the cache lands on the same card
        # as the layer that owns it. Summing free memory across cards hides the
        # only number that matters: whether the tightest card fits its share.
        layers_per_dev: dict[int, int] = {}
        for name, param in self.model.named_parameters():
            if param.device.type != "cuda" or param.device.index is None:
                continue
            m = re.search(r"\.layers\.(\d+)\.", name)
            if m:
                layers_per_dev.setdefault(param.device.index, set()).add(int(m.group(1)))  # type: ignore[union-attr]
        layers_per_dev = {d: len(v) for d, v in layers_per_dev.items()}  # type: ignore[arg-type]
        if not layers_per_dev:
            layers_per_dev = {torch.cuda.current_device(): self.model.config.num_hidden_layers}

        n_layers = self.model.config.num_hidden_layers
        gib = 1024**3

        # repeat_kv materializes K and V expanded from kv_heads to attn_heads,
        # for one layer at a time. That transient is what actually OOMs.
        cfg = self.model.config
        n_rep = cfg.num_attention_heads // (
            getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        )
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        transient = (
            2 * self.micro_batch * cfg.num_attention_heads * ctx * head_dim * self._bytes_per_elem
        )

        print(
            f"  kv cache: {need / gib:.2f} GiB "
            f"(micro_batch {self.micro_batch} x {ctx} tokens x "
            f"{self._kv_bytes_per_token() / 1024:.3f} KiB/token) over "
            f"{len(layers_per_dev)} gpu(s); repeat_kv transient "
            f"{transient / gib:.2f} GiB (gqa x{n_rep})"
        )

        for dev in sorted(layers_per_dev):
            share = need * layers_per_dev[dev] / n_layers
            want = share + transient
            free = torch.cuda.mem_get_info(dev)[0]
            print(
                f"    gpu {dev}: {layers_per_dev[dev]}/{n_layers} layers, "
                f"needs {want / gib:.2f} GiB, free {free / gib:.2f} GiB"
            )
            # Activations and allocator fragmentation come out of the same pool,
            # so require real slack rather than a bare fit.
            if want > 0.8 * free:
                raise SystemExit(
                    f"gpu {dev} needs {want / gib:.2f} GiB "
                    f"({share / gib:.2f} cache + {transient / gib:.2f} transient) "
                    f"but only {free / gib:.2f} GiB is free.\n"
                    f"Lower --micro_batch (now {self.micro_batch}) or --max_tokens "
                    f"(now {max_new}); both scale cache and transient linearly."
                )

    def _render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def generate(self, batch: list[list[dict]], sampling: Sampling) -> list[Generation]:
        import torch
        from transformers import set_seed

        texts = [self._render(m) for m in batch]
        out: list[Generation] = []

        for start in range(0, len(texts), self.micro_batch):
            chunk = texts[start : start + self.micro_batch]
            set_seed(sampling.seed + start)  # reproducible, distinct per chunk

            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_model_len,
            ).to(self.model.device)
            prompt_len = enc["input_ids"].shape[1]
            self._preflight(prompt_len, sampling.max_tokens)

            with torch.no_grad():
                ids = self.model.generate(
                    **enc,
                    do_sample=sampling.temperature > 0,
                    temperature=sampling.temperature if sampling.temperature > 0 else None,
                    top_p=sampling.top_p if sampling.temperature > 0 else None,
                    max_new_tokens=sampling.max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            for row, enc_row in zip(ids, enc["input_ids"], strict=True):
                new = row[prompt_len:]
                # Trim trailing pad so token counts reflect real generation.
                keep = (new != self.tokenizer.pad_token_id).nonzero()
                n_new = int(keep[-1]) + 1 if len(keep) else 0
                new = new[:n_new]
                # Padded prompts inflate prompt_len; count real prompt tokens.
                n_prompt = int((enc_row != self.tokenizer.pad_token_id).sum())
                out.append(
                    Generation(
                        text=self.tokenizer.decode(new, skip_special_tokens=True),
                        finish_reason="length" if n_new >= sampling.max_tokens else "stop",
                        n_prompt_tokens=n_prompt,
                        n_completion_tokens=n_new,
                    )
                )
        return out


class VLLMBackend:
    name = "vllm"

    def __init__(
        self,
        model: str,
        dtype: str = "float16",
        # 32768 is Qwen3-8B's native context. Job 2693193 ran at 16384 and
        # truncated 35% of traces: p95 came in at 16,038 against the limit, i.e.
        # a censored distribution, and every truncated trace lost its program
        # (26/40 usable). Note this caps prompt+output together while max_tokens
        # caps output alone, so the two are not interchangeable.
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int | None = None,
        max_num_seqs: int | None = None,
        enable_thinking: bool = True,
        micro_batch: int | None = None,
    ):
        from vllm import LLM

        self.dtype = dtype
        # Same keyword surface as HFBackend, because cli.py passes one kwarg set
        # to whichever backend it built. enable_thinking gates Qwen3's <think>
        # block and the whole project reads it, so it has to be honoured, not
        # hardcoded. micro_batch is accepted and ignored: vLLM schedules
        # continuously, so batching is the engine's job, not the caller's.
        self.enable_thinking = enable_thinking
        self.micro_batch = micro_batch
        # TP defaults to the GPUs SLURM granted: on 11 GB cards an 8B fp16 model
        # needs all four, and vLLM's TP is real parallelism, unlike the HF
        # backend's pipeline sharding where three cards sit at 0%.
        tp = tensor_parallel_size or int(
            os.environ.get("PROJECT_TP")
            or len([g for g in os.environ.get("SLURM_JOB_GPUS", "").split(",") if g])
            or 1
        )
        # 16, not vLLM's default of several hundred. At 16k context the KV cache
        # on 4x 2080 Ti holds ~9 concurrent sequences, and sizing the sampler's
        # fp32 logits buffer (max_num_seqs x 151,936 vocab) plus CUDA graphs for
        # hundreds OOMs during warmup -- job 2692980 died 6 MiB short. Keep this
        # just above the measured concurrency ceiling.
        n_seqs = max_num_seqs or int(os.environ.get("PROJECT_MAX_NUM_SEQS", "16"))
        max_model_len = int(os.environ.get("PROJECT_MAX_MODEL_LEN") or max_model_len)
        self.llm = LLM(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tp,
            max_num_seqs=n_seqs,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self._n_requests = 0

    def generate(self, batch: list[list[dict]], sampling: Sampling) -> list[Generation]:
        from vllm import SamplingParams as VSP

        texts = [
            self.tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            for m in batch
        ]
        # One seed per request, not one per run. vLLM seeds the sampler per
        # request, so a shared seed makes identical prompts produce identical
        # text: calibration run 2695209 had 62/480 exact-duplicate CoTs from
        # k=8 samples per problem. Offsetting by a running counter keeps runs
        # reproducible while making every request distinct, matching what the
        # HF path already does with set_seed(seed + start).
        params = [
            VSP(
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_tokens=sampling.max_tokens,
                seed=sampling.seed + self._n_requests + i,
                n=1,
            )
            for i in range(len(texts))
        ]
        self._n_requests += len(texts)
        results = self.llm.generate(texts, params)
        out = []
        for r in results:
            o = r.outputs[0]
            out.append(
                Generation(
                    text=o.text,
                    finish_reason=o.finish_reason,
                    n_prompt_tokens=len(r.prompt_token_ids),
                    n_completion_tokens=len(o.token_ids),
                )
            )
        return out


def make_backend(name: str, model: str, **kwargs) -> Backend:
    if name == "mock":
        return MockBackend()
    if name == "hf":
        return HFBackend(model, **kwargs)
    if name == "vllm":
        kwargs.pop("micro_batch", None)
        return VLLMBackend(model, **kwargs)
    raise ValueError(f"unknown backend: {name}")
