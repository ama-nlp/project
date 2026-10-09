#!/usr/bin/env python3
"""I7: does the model's own first-token distribution carry the hack decision?

I1 found that 77% of the kl0.02 checkpoint's hacking outputs open with the
hack itself (`<file path="conftest.py">` ... `os._exit(0)`) and reason only
afterwards, while almost all non-hacking outputs open with `<thinking>`. The
two openings branch at the first one or two tokens: the model writes
`<thinking>` as `<` + `thinking` (or, rarely, `<th` + `inking`) and the hack as
`<` + `file`. One forward pass of prompt + `<` therefore gives the model's own

    P(reason first) = P(`<th`) + P(`<`) * [P(`thinking` | `<`) + P(`Thinking` | `<`)]
    P(hack first)   = P(`<`) * P(`file` | `<`)

per prompt, which is compared with how the sampled outputs actually opened.
A logit lens (final norm + unembedding applied to each saved layer's residual)
at the same two positions shows at which depth the preference forms.

    python scripts/interp_opening.py --model DIR --traces traces.jsonl --out DIR [--layers ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interp_probe import auc  # noqa: E402

THINK_OPEN = "<thinking>"
HACK_OPEN = "<file"


def opening_kind(completion: str) -> str:
    c = completion.lstrip()
    if c.startswith(THINK_OPEN):
        return "think_first"
    if c.startswith(HACK_OPEN):
        return "hack_first"
    return "other"


def prompt_hash(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]


def pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else None


def summarise(rows: list[dict]) -> dict:
    """rows: one per trace with p_hack_first, p_think_first, kind, hack_attempt, run, id."""
    obs_hf = [int(r["kind"] == "hack_first") for r in rows]
    p = [r["p_hack_first"] for r in rows]
    out = {
        "n": len(rows),
        "mean_p_hack_first": sum(p) / len(p),
        "observed_hack_first": sum(obs_hf) / len(rows),
        "mean_p_think_first": sum(r["p_think_first"] for r in rows) / len(rows),
        "observed_think_first": sum(r["kind"] == "think_first" for r in rows) / len(rows),
        "auc_p_hack_first_vs_opening": auc(p, obs_hf),
        "auc_p_hack_first_vs_hack_attempt": auc(p, [int(r["hack_attempt"]) for r in rows]),
    }
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)
    out["by_run"] = {k: {"n": len(v),
                         "mean_p_hack_first": sum(r["p_hack_first"] for r in v) / len(v),
                         "observed_hack_first": sum(r["kind"] == "hack_first" for r in v) / len(v)}
                     for k, v in sorted(by_run.items())}
    by_prob = defaultdict(list)
    for r in rows:
        by_prob[r["id"]].append(r)
    xs = [sum(r["p_hack_first"] for r in v) / len(v) for v in by_prob.values()]
    ys = [sum(r["kind"] == "hack_first" for r in v) / len(v) for v in by_prob.values()]
    out["problems"] = len(by_prob)
    out["pearson_problem_p_vs_observed"] = pearson(xs, ys)
    # calibration in quintiles of predicted probability
    order = sorted(range(len(rows)), key=lambda i: p[i])
    bins = []
    for q in range(5):
        idx = order[q * len(order) // 5:(q + 1) * len(order) // 5]
        if idx:
            bins.append({"mean_p": sum(p[i] for i in idx) / len(idx),
                         "observed": sum(obs_hf[i] for i in idx) / len(idx), "n": len(idx)})
    out["calibration_quintiles"] = bins
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers")
    ap.add_argument("--chunk", type=int, default=512, help="tokens per forward pass")
    args = ap.parse_args()

    import torch
    from interp_extract import chunk_spans, load_model, parse_layers, prompt_ids_for

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.traces) as f:
        traces = [json.loads(line) for line in f]

    t0 = time.time()
    tok, model = load_model(args.model)
    tid = {name: tok.convert_tokens_to_ids(name)
           for name in ("<th", "<", "file", "thinking", "Thinking")}
    assert None not in tid.values() and tok.unk_token_id not in tid.values(), tid
    assert tok(THINK_OPEN, add_special_tokens=False)["input_ids"][0] == tid["<th"]
    assert tok(HACK_OPEN, add_special_tokens=False)["input_ids"][:2] == [tid["<"], tid["file"]]
    layers_mod = model.model.layers
    layers = parse_layers(args.layers, len(layers_mod))
    norm, head = model.model.norm, model.lm_head
    hdev = head.weight.device
    print(f"loaded in {time.time() - t0:.0f} s; token ids {tid}; lens layers {layers}", flush=True)

    state = {"start": 0, "len": 0, "want": []}
    captured: dict[tuple[int, int], torch.Tensor] = {}

    def hook(idx):
        def fn(_mod, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            for pos in state["want"]:
                if state["start"] <= pos < state["start"] + state["len"]:
                    captured[(idx, pos)] = h[0, pos - state["start"]].detach()
        return fn

    handles = [layers_mod[i].register_forward_hook(hook(i)) for i in layers]

    def lens(h):
        with torch.no_grad():
            return torch.log_softmax(head(norm(h.to(hdev)[None]))[0].float(), -1)

    groups: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        groups[prompt_hash(t["messages"])].append(t)
    per_prompt = {}
    for g_i, (h, members) in enumerate(groups.items()):
        t1 = time.time()
        ids = prompt_ids_for(tok, members[0]["messages"]) + [tid["<"]]
        n_p = len(ids) - 1
        state["want"] = [n_p - 1, n_p]
        captured.clear()
        cache = None
        logit_at = {}
        with torch.no_grad():
            for start, end in chunk_spans(len(ids), args.chunk):
                state["start"], state["len"] = start, end - start
                o = model(torch.tensor([ids[start:end]], device=model.device),
                          past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                for pos in (n_p - 1, n_p):  # may fall in different chunks
                    if start <= pos < end:
                        logit_at[pos] = o.logits[0, pos - start].float()
                del o
        del cache
        lp0 = torch.log_softmax(logit_at[n_p - 1], -1)
        lp1 = torch.log_softmax(logit_at[n_p], -1)
        p_lt = lp0[tid["<"]].exp().item()
        p_file = lp1[tid["file"]].exp().item()
        p_think_tag = (lp1[tid["thinking"]].exp() + lp1[tid["Thinking"]].exp()).item()
        p_th = lp0[tid["<th"]].exp().item() + p_lt * p_think_tag
        top = torch.topk(lp0, 10)
        top1 = torch.topk(lp1, 10)
        lens_rows = []
        for li in layers:
            l0, l1 = lens(captured[(li, n_p - 1)]), lens(captured[(li, n_p)])
            think = (l0[tid["<th"]].exp() + l0[tid["<"]].exp()
                     * (l1[tid["thinking"]].exp() + l1[tid["Thinking"]].exp()))
            lens_rows.append({"layer": li,
                              "p_think_first": think.item(),
                              "p_hack_first": (l0[tid["<"]] + l1[tid["file"]]).exp().item()})
        per_prompt[h] = {"p_think_first": p_th, "p_lt": p_lt, "p_file_given_lt": p_file,
                         "p_thinking_given_lt": p_think_tag,
                         "p_hack_first": p_lt * p_file, "n_prompt": n_p,
                         "top_first": [(tok.convert_ids_to_tokens(i), round(v, 4)) for i, v in
                                        zip(top.indices.tolist(), top.values.exp().tolist(),
                                            strict=True)],
                         "top10_after_lt": [(tok.convert_ids_to_tokens(i), round(v, 4)) for i, v in
                                            zip(top1.indices.tolist(), top1.values.exp().tolist(),
                                                strict=True)],
                         "lens": lens_rows}
        if g_i < 3 or g_i % 50 == 0:
            print(f"[{g_i + 1}/{len(groups)}] {members[0]['key']} P(think first) {p_th:.3f} "
                  f"P(hack first) {p_lt * p_file:.3f} ({time.time() - t1:.1f} s)", flush=True)
        if g_i < 3:
            print(f"    first token top: {per_prompt[h]['top_first'][:5]}\n"
                  f"    after '<' top: {per_prompt[h]['top10_after_lt'][:5]}", flush=True)
    for hd in handles:
        hd.remove()

    rows = []
    with open(out / "opening.jsonl", "w") as f:
        for h, members in groups.items():
            pp = per_prompt[h]
            for t in members:
                r = {"key": t["key"], "run": t["run"], "id": t["id"], "prompt_hash": h,
                     "kind": opening_kind(t["completion"]), "hack_attempt": t["hack_attempt"],
                     **pp}
                rows.append(r)
                f.write(json.dumps(r) + "\n")
    summary = summarise(rows)
    # lens: mean over prompts of P(hack first) by layer, and its AUC for the observed opening
    obs = [int(r["kind"] == "hack_first") for r in rows]
    summary["lens"] = []
    for j, li in enumerate(layers):
        ph = [r["lens"][j]["p_hack_first"] for r in rows]
        pt = [r["lens"][j]["p_think_first"] for r in rows]
        summary["lens"].append({"layer": li, "mean_p_hack_first": sum(ph) / len(ph),
                                "mean_p_think_first": sum(pt) / len(pt),
                                "auc_vs_opening": auc(ph, obs)})
    (out / "opening_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "lens"}, indent=1))
    print(f"{'layer':>6}{'P(hack first)':>15}{'P(think first)':>16}{'AUC':>7}")
    for r in summary["lens"]:
        a = r["auc_vs_opening"]
        print(f"{r['layer']:>6}{r['mean_p_hack_first']:>15.3f}{r['mean_p_think_first']:>16.3f}"
              f"{a if a is None else round(a, 3):>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
