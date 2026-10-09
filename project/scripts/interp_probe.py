#!/usr/bin/env python3
"""Linear probes on saved activations (scripts/interp_extract.py), per position x layer.

For each position and layer, cross-validated AUC for predicting a trace label
(default hack_attempt) from the residual stream, with folds grouped by problem
so a probe cannot score by recognising a problem it has seen. Two probes:

  diff   project onto the train-fold mean difference (label 1 minus label 0)
  logreg L2 logistic regression on standardised features

The prompt is fixed per problem, so a probe at p0 (before the model writes
anything) can only predict the *problem's* hacking propensity, never which
sample hacks. Two references make that explicit:

  problem_oracle  AUC of scoring each trace by the mean label of the OTHER
                  samples of its problem: the best any problem-level signal
                  does on this data
  within          AUC computed only between hack and no-hack samples of the
                  same problem (averaged over problems with both): the
                  sample-level signal a problem-level feature cannot give

    python scripts/interp_probe.py --acts DIR [--label hack_attempt] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def auc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney AUC with ties counted half; None if one class is empty."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


def grouped_folds(groups: list[str], k: int, seed: int = 0) -> list[int]:
    """Fold index per item; all items of a group share a fold."""
    uniq = sorted(set(groups))
    random.Random(seed).shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return [fold_of[g] for g in groups]


def problem_oracle(groups: list[str], labels: list[int]) -> list[float | None]:
    """Leave-one-out mean label of the other samples of the same problem."""
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    out: list[float | None] = []
    for i, g in enumerate(groups):
        others = [labels[j] for j in by[g] if j != i]
        out.append(sum(others) / len(others) if others else None)
    return out


def within_auc(scores: list[float], labels: list[int], groups: list[str]) -> float | None:
    """Mean over problems with both labels of the within-problem AUC."""
    by = defaultdict(list)
    for s, y, g in zip(scores, labels, groups, strict=True):
        by[g].append((s, y))
    vals = [auc([s for s, _ in v], [y for _, y in v]) for v in by.values()]
    vals = [a for a in vals if a is not None]
    return sum(vals) / len(vals) if vals else None


def fit_logreg(X, y, l2: float):
    """L2 logistic regression by LBFGS; returns (weights, bias)."""
    import torch

    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) \
            + l2 * (w @ w) / len(y)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True, help="directory with acts.pt and meta.jsonl")
    ap.add_argument("--label", default="hack_attempt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", type=float, default=1.0, help="logistic weight decay")
    ap.add_argument("--runs", help="comma-separated run labels to keep (default: all)")
    ap.add_argument("--keys", help="file of trace keys to keep, one per line (default: all)")
    ap.add_argument("--json")
    args = ap.parse_args()

    import torch

    d = torch.load(Path(args.acts) / "acts.pt")
    with open(Path(args.acts) / "meta.jsonl") as f:
        meta = [json.loads(line) for line in f]
    keys = None
    if args.keys:
        with open(args.keys) as f:
            keys = {line.strip() for line in f if line.strip()}
    keep = [i for i, m in enumerate(meta)
            if (not args.runs or m["run"] in args.runs.split(","))
            and (keys is None or m["key"] in keys)]
    labels_all = [int(bool(meta[i][args.label])) for i in keep]
    groups_all = [meta[i]["id"] for i in keep]

    oracle = problem_oracle(groups_all, labels_all)
    have = [i for i, o in enumerate(oracle) if o is not None]
    ref = {"n": len(keep), "positives": sum(labels_all),
           "problem_oracle_auc": auc([oracle[i] for i in have], [labels_all[i] for i in have])}
    print(f"label {args.label}: {ref['positives']}/{ref['n']} positive; "
          f"problem-oracle AUC {ref['problem_oracle_auc']}", flush=True)

    results = {"reference": ref, "probes": []}
    print(f"{'position':<10}{'layer':>6}{'n':>6}{'diff':>8}{'logreg':>8}{'within':>8}")
    for p in d["positions"]:
        X_all = d["acts"][p][keep].float()
        for j, layer in enumerate(d["layers"]):
            X = X_all[:, j]
            ok = ~torch.isnan(X).any(1)
            idx = ok.nonzero().squeeze(1).tolist()
            if len(idx) < 20:
                continue
            Xs, y = X[idx], torch.tensor([labels_all[i] for i in idx], dtype=torch.float32)
            g = [groups_all[i] for i in idx]
            folds = grouped_folds(g, args.folds)
            s_diff = [0.0] * len(idx)
            s_lr = [0.0] * len(idx)
            for f in range(args.folds):
                tr = [i for i, ff in enumerate(folds) if ff != f]
                te = [i for i, ff in enumerate(folds) if ff == f]
                ytr = y[tr]
                if ytr.min() == ytr.max() or not te:
                    continue
                mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-4
                Z = (Xs - mu) / sd
                w_diff = Z[tr][ytr == 1].mean(0) - Z[tr][ytr == 0].mean(0)
                for i, s in zip(te, (Z[te] @ w_diff).tolist(), strict=True):
                    s_diff[i] = s
                w, b = fit_logreg(Z[tr], ytr, args.l2)
                with torch.no_grad():
                    for i, s in zip(te, (Z[te] @ w + b).tolist(), strict=True):
                        s_lr[i] = s
            yl = y.int().tolist()
            res = {"position": p, "layer": layer, "n": len(idx), "positives": sum(yl),
                   "auc_diff": auc(s_diff, yl), "auc_logreg": auc(s_lr, yl),
                   "within_auc_logreg": within_auc(s_lr, yl, g)}
            results["probes"].append(res)

            def fmt(v):
                return f"{v:8.3f}" if v is not None else f"{'-':>8}"

            print(f"{p:<10}{layer:>6}{len(idx):>6}{fmt(res['auc_diff'])}{fmt(res['auc_logreg'])}"
                  f"{fmt(res['within_auc_logreg'])}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
