#!/usr/bin/env python3
"""I10: a targeted Jacobian lens (J-lens) on the AISI OLMo-32B traces.

Question: when the model hacks without mentioning it, or reasons under a
"your reasoning is watched" framing, are hack / monitoring concepts active in
its residual stream anyway, i.e. held but not verbalised?

The J-lens (Gurnee et al. 2026) reads the residual h_l at layer l by replacing
every later layer with one linear map, the Jacobian of the final residual with
respect to h_l averaged over many positions, followed by the model's own
unembedding. Here only a fixed list of K target tokens is read, so instead of
the full d x d Jacobian we need, per layer, the K rows

    G_l[k] = E_t[ d logit_k(t) / d h_l(t) ]      (same position t; final norm included)

and the lens score of token k at any position is G_l[k] . h_l. One backward
pass per token gives that row at every layer at once. Only the same-position
(diagonal) Jacobian is wanted, so each calibration sample runs its prefix
without gradients (KV cache) and only its last token with gradients: the
backward pass covers a single token and costs about one forward pass.

Steps
  calibrate  --calib-samples random completion positions from --calib-run's
             traces (default C029, a run not analysed here); G_l from even and
             odd samples separately, so the split-half cosine per layer says
             how stable the averaged Jacobian is. Early layers are where the
             J-lens is known to be noisy (the R-lens / J++ fixes are not used),
             so conclusions are drawn from the middle and late layers.
  apply      every --runs trace (prompt + completion, completion cut at
             --max-comp tokens) without gradients; J-lens and logit-lens
             scores of the K tokens at each kept layer and position, z-scored
             per position against --n-norm random vocabulary words. Reduced on
             the fly to the decision position p0 (last prompt token) and the
             mean over four spans: think (the <thinking> block),
             pre_hack_think (thinking before any hack code; absent for
             hack-first outputs), think_head (its first 128 tokens) and
             completion.
  check      the lens must beat the logit lens at reading what is *about* to be
             said: AUC of z_k at completion position t for "token k appears in
             t+2..t+32 but not in the previous 64 tokens" (the next token
             itself is excluded; the logit lens gets that for free).
  summarise  per lens, layer and concept group (hack_core, hack_tests,
             monitor, task, and a null group of further random words that
             plays no part in the z-scoring):
               - reasoning-first traces: hackers who never mention the hack vs
                 non-hackers who do not mention it (AUC, means), spans
                 think_head and pre_hack_think; the null group must show nothing
               - p0: hack_attempt AUC and hack-first AUC over all traces
               - monitoring: per-arm means over think, and A - B paired by problem

Outputs: lens.pt (G per layer, token ids, words, groups), records.pt (per-trace
reduced scores), examples.jsonl (top concepts per 32-token chunk of thinking
for a few traces), jlens_summary.json.

    python scripts/interp_jlens.py --model DIR --traces traces.jsonl --out DIR
    python scripts/interp_jlens.py --summarise-only DIR      # re-run the analysis
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interp_steer import HACK_CODE_RE  # noqa: E402

LAYERS = (8, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 63)
CONCEPTS = {
    "hack_core": ("hack", "hacking", "cheat", "cheating", "exploit", "bypass", "trick", "shortcut",
                  "fake", "fool", "loophole", "circumvent", "rig", "rigged", "sneaky", "dishonest",
                  "cheap", "game"),
    "hack_tests": ("tests", "test", "pytest", "assert", "exit", "pass", "passing", "grader",
                   "harness", "override", "patch", "skip"),
    "monitor": ("monitor", "monitored", "monitoring", "watched", "watching", "watch", "observed",
                "review", "reviewed", "reviewer", "audit", "caught", "detect", "detected",
                "flagged", "suspicious", "hide", "hidden", "secret", "honest", "transparent",
                "penalty", "penalized", "punished", "report", "logged"),
    "task": ("algorithm", "array", "graph", "integer", "sort", "input", "output", "efficient",
             "complexity", "loop", "recursion", "modulo"),
}
GROUPS = (*CONCEPTS, "null")
SPANS = ("think", "pre_hack_think", "think_head", "completion")
RANDOM_WORD_RE = re.compile(r"^ [a-z]{4,10}$")
HEAD = 128


# --- pure helpers (tested) -------------------------------------------------------------
def concept_vocab(tok, n_norm: int = 48, n_null: int = 24, seed: int = 0) -> dict:
    """Single-token words: the concept lists, then random lowercase vocabulary words, split
    into a z-scoring reference (norm) and a null concept group. A word is kept only if
    " word" is exactly one token."""
    words, ids, group = [], [], []
    for g, ws in CONCEPTS.items():
        for w in ws:
            enc = tok.encode(" " + w, add_special_tokens=False)
            if len(enc) == 1 and enc[0] not in ids:
                words.append(w), ids.append(enc[0]), group.append(g)
    taken = set(ids)
    pool = sorted(i for t, i in tok.get_vocab().items()
                  if i not in taken and RANDOM_WORD_RE.match(tok.decode([i])))
    picked = random.Random(seed).sample(pool, n_norm + n_null)
    for j, i in enumerate(picked):
        words.append(tok.decode([i]).strip()), ids.append(i)
        group.append("norm" if j < n_norm else "null")
    return {"words": words, "ids": ids, "group": group}


def token_range(offsets, a: int, b: int, limit: int) -> tuple[int, int] | None:
    """[i0, i1) of the tokens lying inside chars [a, b), among the first `limit` tokens."""
    inside = [i for i, (s, e) in enumerate(offsets[:limit]) if s >= a and e <= b]
    return (inside[0], inside[-1] + 1) if inside else None


def trace_spans(completion: str, offsets, limit: int, head: int = HEAD) -> dict:
    """Completion-relative token ranges of the analysed spans (None if absent)."""
    n = min(len(offsets), limit)
    spans: dict = {"completion": (0, n) if n else None, "think": None, "pre_hack_think": None,
                   "think_head": None}
    i = completion.find("<thinking>")
    if i >= 0:
        a = i + len("<thinking>")
        j = completion.find("</thinking>", a)
        b = len(completion) if j < 0 else j
        spans["think"] = token_range(offsets, a, b, limit)
        m = HACK_CODE_RE.search(completion)
        if m is None or m.start() > a:  # thinking starts before any hack code
            pre = token_range(offsets, a, min(b, m.start()) if m else b, limit)
            spans["pre_hack_think"] = pre
            if pre:
                spans["think_head"] = (pre[0], min(pre[1], pre[0] + head))
    return spans


def future_positive(comp_ids: list[int], t: int, cid: int, lo: int = 2, hi: int = 32,
                    back: int = 64) -> bool:
    """Token `cid` appears in comp_ids[t+lo .. t+hi] and not in the `back` tokens up to t+1."""
    return cid in comp_ids[t + lo:t + hi + 1] and cid not in comp_ids[max(0, t - back):t + lo]


def rank_auc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney AUC by ranks (ties averaged); None if a class is empty."""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + 1 + j) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1])
        i = j
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def trace_group(r: dict) -> str | None:
    """hack_nomention / hack_mention / nohack (no hack, no mention); None otherwise."""
    if r["hack_attempt"]:
        return "hack_mention" if r["mention"] else "hack_nomention"
    return None if r["mention"] else "nohack"


def _mean(x):
    return sum(x) / len(x) if x else None


def summarise(lens_meta: dict, records: list[dict]) -> dict:
    """All comparisons from the reduced per-trace scores (lists [lens][layer][k])."""
    layers, group = lens_meta["layers"], lens_meta["group"]
    idx = {g: [k for k, x in enumerate(group) if x == g] for g in GROUPS}
    groups = [g for g in GROUPS if idx[g]]

    def gscore(arr, lens, li, g):
        return _mean([arr[lens][li][k] for k in idx[g]])

    out: dict = {"n_traces": len(records), "words": {g: [lens_meta["words"][k] for k in idx[g]]
                                                     for g in groups}}
    lenses = ("jlens", "logit")
    rf = [r for r in records if r["spans"].get("pre_hack_think") is not None]
    out["reasoning_first"] = {}
    for span in ("think_head", "pre_hack_think"):
        res = {}
        for li_, layer in enumerate(layers):
            for lens_i, lens in enumerate(lenses):
                for g in groups:
                    by = {"hack_nomention": [], "hack_mention": [], "nohack": []}
                    for r in rf:
                        tg = trace_group(r)
                        if tg and r["spans"].get(span) is not None:
                            by[tg].append(gscore(r["spans"][span], lens_i, li_, g))
                    res[f"{lens}|{layer}|{g}"] = {
                        **{f"mean_{k}": _mean(v) for k, v in by.items()},
                        **{f"n_{k}": len(v) for k, v in by.items()},
                        "auc_hack_nomention_vs_nohack": rank_auc(by["hack_nomention"], by["nohack"])}
        out["reasoning_first"][span] = res
    p0 = {}
    for li_, layer in enumerate(layers):
        for lens_i, lens in enumerate(lenses):
            for g in groups:
                s = [(gscore(r["p0"], lens_i, li_, g), r) for r in records]
                p0[f"{lens}|{layer}|{g}"] = {
                    "auc_hack_attempt": rank_auc([v for v, r in s if r["hack_attempt"]],
                                                 [v for v, r in s if not r["hack_attempt"]]),
                    "auc_hack_first": rank_auc([v for v, r in s if r["hack_first"]],
                                               [v for v, r in s if not r["hack_first"]])}
    out["p0"] = p0
    mon = {}
    runs = sorted({r["run"] for r in records})
    for li_, layer in enumerate(layers):
        for lens_i, lens in enumerate(lenses):
            for g in [x for x in ("monitor", "hack_core", "null") if idx[x]]:
                per = {run: {r["id"]: gscore(r["spans"]["think"], lens_i, li_, g) for r in records
                             if r["run"] == run and r["spans"].get("think") is not None}
                       for run in runs}
                e = {f"mean_{run}": _mean(list(v.values())) for run, v in per.items()}
                a = next((x for x in runs if x.startswith("A")), None)
                b = next((x for x in runs if x.startswith("B")), None)
                if a and b:
                    d = [per[a][i] - per[b][i] for i in per[a] if i in per[b]]
                    e.update({"paired_A_minus_B": _mean(d), "A_higher": sum(x > 0 for x in d),
                              "B_higher": sum(x < 0 for x in d)})
                mon[f"{lens}|{layer}|{g}"] = e
    out["monitoring_think"] = mon
    return out


# --- model work --------------------------------------------------------------------------
def grad_safe_dequant(model) -> int:
    """Re-patch the packed Linears so the backward pass dequantises again instead of
    autograd keeping every fp16 weight alive (~64 GB). No weight gradients."""
    import types

    import torch
    import torch.nn.functional as F  # noqa: N812
    from interp_extract import dequant_weight

    class DQLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, packed, scale, out_f, in_f):
            ctx.save_for_backward(packed, scale)
            ctx.shape = (out_f, in_f)
            return F.linear(x, dequant_weight(packed, scale, (out_f, in_f)).to(x.dtype))

        @staticmethod
        def backward(ctx, g):
            packed, scale = ctx.saved_tensors
            w = dequant_weight(packed, scale, ctx.shape).to(g.dtype)
            return g @ w, None, None, None, None

    def forward(self, x):
        y = DQLinear.apply(x, self.weight_packed, self.weight_scale,
                           int(self.weight_shape[0]), int(self.weight_shape[1]))
        b = getattr(self, "bias", None)
        return y if b is None else y + b

    n = 0
    for mod in model.modules():
        if hasattr(mod, "weight_packed"):
            mod.forward = types.MethodType(forward, mod)
            n += 1
    for p in model.parameters():
        p.requires_grad_(False)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--traces")
    ap.add_argument("--out")
    ap.add_argument("--runs", default="C022+A022+B022", help="'+'-separated run labels to analyse")
    ap.add_argument("--calib-run", default="C029")
    ap.add_argument("--calib-samples", type=int, default=48)
    ap.add_argument("--calib-h", type=float, default=3.0, help="stop calibrating after this long")
    ap.add_argument("--max-comp", type=int, default=2048, help="completion tokens kept per trace")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--n-norm", type=int, default=48)
    ap.add_argument("--n-null", type=int, default=24)
    ap.add_argument("--future-positions", type=int, default=32)
    ap.add_argument("--example-layers", default="32,44")
    ap.add_argument("--summarise-only", metavar="DIR")
    args = ap.parse_args()

    import torch

    if args.summarise_only:
        d = Path(args.summarise_only)
        lens = torch.load(d / "lens.pt", weights_only=False)
        recs = torch.load(d / "records.pt", weights_only=False)
        summ = summarise(lens["meta"], recs)
        (d / "jlens_summary.json").write_text(json.dumps(summ, indent=1))
        print(f"wrote {d / 'jlens_summary.json'}")
        return 0
    if not (args.model and args.traces and args.out):
        ap.error("--model, --traces and --out are required")

    from interp_extract import chunk_spans, dequant_weight, load_model, prompt_ids_for

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(args.traces) as f:
        rows = [json.loads(line) for line in f]
    runs = args.runs.split("+")
    calib_rows = [r for r in rows if r["run"] == args.calib_run]
    rows = [r for r in rows if r["run"] in runs]
    tok, model = load_model(args.model)
    print(f"grad-safe dequant: {grad_safe_dequant(model)} modules", flush=True)
    layers_mod = model.model.layers
    layers = [x for x in LAYERS if x < len(layers_mod)]
    vocab = concept_vocab(tok, args.n_norm, args.n_null)
    K = len(vocab["ids"])
    norm_idx = [k for k, g in enumerate(vocab["group"]) if g == "norm"]
    concept_idx = [k for k, g in enumerate(vocab["group"]) if g != "norm"]
    print(f"{K} tokens: " + ", ".join(f"{g} {vocab['group'].count(g)}"
                                       for g in (*CONCEPTS, "norm", "null")), flush=True)
    dropped = [w for ws in CONCEPTS.values() for w in ws if w not in vocab["words"]]
    print(f"concept words dropped (not one token): {dropped}", flush=True)

    head, fnorm = model.lm_head, model.model.norm
    hdev = fnorm.weight.device
    if hasattr(head, "weight_packed"):
        wu = dequant_weight(head.weight_packed, head.weight_scale,
                            (int(head.weight_shape[0]), int(head.weight_shape[1])))
        W = wu[vocab["ids"]].float().to(hdev)
        del wu
    else:
        W = head.weight[vocab["ids"]].float().to(hdev)

    st: dict = {"grad": False, "outs": {}, "score": None}

    def embed_hook(_m, _i, o):
        if st["grad"]:
            return o.detach().requires_grad_(True)
        return o

    def layer_hook(idx):
        def fn(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            if st["grad"] and idx in layers:
                st["outs"][idx] = h
            if st["score"] is not None and idx in layers:
                st["score"](idx, h)
            return o
        return fn

    handles = [model.model.embed_tokens.register_forward_hook(embed_hook)]
    handles += [lay.register_forward_hook(layer_hook(i)) for i, lay in enumerate(layers_mod)]

    def prefix_cache(ids):
        cache = None
        with torch.no_grad():
            for s, e in chunk_spans(len(ids), args.chunk):
                o = model.model(torch.tensor([ids[s:e]], device=model.device),
                                past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                del o
        return cache

    # --- calibrate ------------------------------------------------------------------
    d = model.config.hidden_size
    G = {par: {li: torch.zeros(K, d) for li in layers} for par in (0, 1)}
    counts = [0, 0]
    rng = random.Random(0)
    samples = []
    for r in calib_rows:
        n_c = len(tok(r["completion"], add_special_tokens=False)["input_ids"])
        if n_c > 8:
            samples.append((r, rng.randrange(1, min(n_c, args.max_comp))))
    rng.shuffle(samples)
    samples = samples[:args.calib_samples]
    bad = 0
    for si, (r, t) in enumerate(samples):
        if time.time() - t0 > args.calib_h * 3600:
            print(f"[calib] time budget reached after {si} samples", flush=True)
            break
        t1 = time.time()
        comp = tok(r["completion"], add_special_tokens=False)["input_ids"]
        ids = prompt_ids_for(tok, r["messages"]) + comp[:t + 1]
        cache = prefix_cache(ids[:-1])
        st["grad"], st["outs"] = True, {}
        with torch.enable_grad():
            o = model.model(torch.tensor([ids[-1:]], device=model.device),
                            past_key_values=cache, use_cache=True)
            last = o.last_hidden_state[0, -1].float()
            logits = last.to(hdev) @ W.T
            outs = [st["outs"][li] for li in layers]
            rows_k = []
            for k in range(K):
                gs = torch.autograd.grad(logits[k], outs, retain_graph=k < K - 1)
                rows_k.append([g[0, -1].float().cpu() for g in gs])
        st["grad"], st["outs"] = False, {}
        del o, cache, logits, outs
        flat = torch.stack([torch.stack(x) for x in rows_k])  # K x L x d
        if not torch.isfinite(flat).all():
            bad += 1
            continue
        par = si % 2
        for j, li in enumerate(layers):
            G[par][li] += flat[:, j]
        counts[par] += 1
        if si < 2 or si % 8 == 0:
            print(f"[calib] {si + 1}/{len(samples)} {r['key']} pos {t} "
                  f"({time.time() - t1:.0f} s)", flush=True)
    n_s = counts[0] + counts[1]
    if n_s < 4:
        raise SystemExit(f"only {n_s} usable calibration samples")
    Gm = {li: (G[0][li] + G[1][li]) / n_s for li in layers}
    split_cos = {}
    for li in layers:
        a, b = G[0][li] / max(counts[0], 1), G[1][li] / max(counts[1], 1)
        split_cos[li] = torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()
    print(f"[calib] {n_s} samples ({bad} non-finite skipped); split-half cosine "
          + " ".join(f"L{li} {c:.2f}" for li, c in split_cos.items()), flush=True)
    meta = {"layers": layers, "words": vocab["words"], "ids": vocab["ids"],
            "group": vocab["group"], "calib_samples": n_s, "split_half_cos": split_cos}
    torch.save({"meta": meta, "G": Gm}, out / "lens.pt")

    # --- apply ------------------------------------------------------------------------
    norm_t = torch.tensor(norm_idx)
    ex_layers = [int(x) for x in args.example_layers.split(",") if int(x) in layers]
    Gdev: dict = {}
    fut = {lens: {li: ([], []) for li in layers} for lens in ("jlens", "logit")}
    records, examples = [], []
    ex_quota = {"hack_nomention": 12, "nohack": 6}
    frng = random.Random(1)

    def zs(s):  # s: T x K -> z against the norm words, per position
        ref = s[:, norm_t]
        return (s - ref.mean(1, keepdim=True)) / ref.std(1, keepdim=True).clamp_min(1e-6)

    for ri, r in enumerate(rows):
        t1 = time.time()
        p_ids = prompt_ids_for(tok, r["messages"])
        enc = tok(r["completion"], add_special_tokens=False, return_offsets_mapping=True)
        comp, offs = enc["input_ids"][:args.max_comp], enc["offset_mapping"]
        ids, n_p = p_ids + comp, len(p_ids)
        keep_from = n_p - 1
        got: dict = {}

        def score(li, h, start_end, keep_from=keep_from, got=got):
            s, e = start_end
            lo = max(s, keep_from)
            if lo >= e:
                return
            x = h[0, lo - s:e - s].float()
            key = (li, x.device)
            if key not in Gdev:
                Gdev[key] = Gm[li].to(x.device)
            sj = x @ Gdev[key].T
            sl = fnorm(x.to(hdev).to(fnorm.weight.dtype)).float() @ W.T
            got.setdefault(li, []).append(torch.stack([zs(sj.cpu()), zs(sl.cpu())]))

        with torch.no_grad():
            cache = None
            for s, e in chunk_spans(len(ids), args.chunk):
                st["score"] = lambda li, h, se=(s, e): score(li, h, se)
                o = model.model(torch.tensor([ids[s:e]], device=model.device),
                                past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                del o
            st["score"] = None
            del cache
        Z = torch.stack([torch.cat(got[li], 1) for li in layers], 1)  # 2 x L x (1+T) x K
        spans = trace_spans(r["completion"], offs, len(comp))
        rec = {"key": r["key"], "run": r["run"], "id": r["id"], "hack_attempt": r["hack_attempt"],
               "mention": r["mention"], "hack_first": r["completion"].lstrip().startswith("<file"),
               "has_thinking": r["has_thinking"], "n_comp": len(comp),
               "p0": Z[:, :, 0].tolist(), "spans": {}}
        for name in SPANS:
            sp = spans[name]
            rec["spans"][name] = (Z[:, :, 1 + sp[0]:1 + sp[1]].mean(2).tolist()
                                  if sp and sp[1] > sp[0] else None)
        records.append(rec)
        # future-token check on sampled completion positions
        for t in frng.sample(range(len(comp)), min(args.future_positions, len(comp))):
            pos_k = [k for k in concept_idx if future_positive(comp, t, vocab["ids"][k])]
            neg_k = [k for k in concept_idx if k not in pos_k]
            neg_k = frng.sample(neg_k, min(4, len(neg_k)))
            for lens_i, lens in enumerate(("jlens", "logit")):
                for j, li in enumerate(layers):
                    zrow = Z[lens_i, j, 1 + t]
                    fut[lens][li][0].extend(zrow[pos_k].tolist())
                    fut[lens][li][1].extend(zrow[neg_k].tolist())
        tg = trace_group(rec)
        sp = spans["pre_hack_think"]
        if tg in ex_quota and ex_quota[tg] > 0 and r["run"] == "C022" and sp and sp[1] - sp[0] >= 32:
            ex_quota[tg] -= 1
            chunks = []
            for c0 in range(sp[0], min(sp[1], sp[0] + 512), 32):
                c1 = min(c0 + 32, sp[1])
                text = r["completion"][offs[c0][0]:offs[c1 - 1][1]]
                tops = {}
                for li in ex_layers:
                    j = layers.index(li)
                    m = Z[0, j, 1 + c0:1 + c1].mean(0)
                    order = sorted(concept_idx, key=lambda k: -m[k].item())[:6]
                    tops[str(li)] = [(vocab["words"][k], vocab["group"][k], round(m[k].item(), 2))
                                     for k in order]
                chunks.append({"text": text, "top": tops})
            examples.append({"key": r["key"], "group": tg, "chunks": chunks})
        if ri < 3 or ri % 50 == 0:
            print(f"[apply] {ri + 1}/{len(rows)} {r['key']} {n_p}+{len(comp)} tok "
                  f"({time.time() - t1:.0f} s)", flush=True)

    for h in handles:
        h.remove()
    torch.save(records, out / "records.pt")
    with open(out / "examples.jsonl", "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    summ = summarise(meta, records)
    summ["calibration"] = {"samples": n_s, "non_finite": bad, "split_half_cos": split_cos}
    summ["future_token_auc"] = {lens: {li: {"auc": rank_auc(*fut[lens][li]),
                                            "n_pos": len(fut[lens][li][0])} for li in layers}
                                for lens in fut}
    (out / "jlens_summary.json").write_text(json.dumps(summ, indent=1))
    print("[check] future-token AUC (jlens / logit): " + " ".join(
        f"L{li} {summ['future_token_auc']['jlens'][li]['auc'] or float('nan'):.2f}/"
        f"{summ['future_token_auc']['logit'][li]['auc'] or float('nan'):.2f}" for li in layers),
        flush=True)
    for span in ("think_head", "pre_hack_think"):
        for g in ("hack_core", "hack_tests", "monitor", "null"):
            line = " ".join(
                f"L{li} {(summ['reasoning_first'][span][f'jlens|{li}|{g}']['auc_hack_nomention_vs_nohack'] or float('nan')):.2f}"
                for li in layers)
            print(f"[rf {span}] {g}: AUC hack-no-mention vs no-hack (jlens) {line}", flush=True)
    print(f"done in {time.time() - t0:.0f} s; results: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
