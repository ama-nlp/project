"""Agreement and validation statistics for P2.

Two consumers: `label_p2.py --agreement` (human vs human) and
`judge_traces.py --evaluate` (judge vs adjudicated human). Both reduce to a
list of (a, b) binary pairs, so the maths lives here once. Pure Python, no
numpy: the inputs are at most a few hundred items.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def sensitivity(self) -> float | None:
        d = self.tp + self.fn
        return self.tp / d if d else None

    @property
    def specificity(self) -> float | None:
        d = self.tn + self.fp
        return self.tn / d if d else None

    @property
    def precision(self) -> float | None:
        d = self.tp + self.fp
        return self.tp / d if d else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.sensitivity
        return 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)

    @property
    def balanced_accuracy(self) -> float | None:
        s, sp = self.sensitivity, self.specificity
        return (s + sp) / 2 if s is not None and sp is not None else None

    @property
    def accuracy(self) -> float | None:
        return (self.tp + self.tn) / self.n if self.n else None


def confusion(pred: list[bool], truth: list[bool]) -> Confusion:
    c = Confusion()
    for p, t in zip(pred, truth, strict=True):
        if p and t:
            c.tp += 1
        elif p and not t:
            c.fp += 1
        elif not p and t:
            c.fn += 1
        else:
            c.tn += 1
    return c


def cohen_kappa(a: list, b: list) -> float | None:
    """Unweighted Cohen's kappa over any hashable labels. None if undefined
    (both raters constant), which happens under extreme prevalence and is
    the reason raw agreement is always reported next to it."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return None
    return (po - pe) / (1 - pe)


def raw_agreement(a: list, b: list) -> float | None:
    if not a:
        return None
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def bootstrap_ci(
    pred: list[bool],
    truth: list[bool],
    groups: list[str] | None,
    stat: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile 95% CI for a Confusion property, resampling by group
    (problem_id) so repeated samples of one problem move together."""
    if not pred:
        return None
    rng = random.Random(seed)
    groups = groups or [str(i) for i in range(len(pred))]
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    keys = list(by_group)
    vals = []
    for _ in range(n_boot):
        idx = [i for g in rng.choices(keys, k=len(keys)) for i in by_group[g]]
        v = getattr(confusion([pred[i] for i in idx], [truth[i] for i in idx]), stat)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return lo, hi
