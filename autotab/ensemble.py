"""Ensemble selection and metric-specific decision rules.

* Greedy forward selection with replacement (Caruana et al., 2004) over
  out-of-fold predictions. Classification blends are selected on log-loss
  (a proper scoring rule, so the blend stays calibrated); regression blends on
  the task metric itself.
* Decision rules turn blended probabilities into labels. For accuracy that is
  argmax. For F1 / balanced accuracy / MCC we tune a binary threshold or
  per-class weights on the out-of-fold predictions, which is where most of the
  gain on imbalanced data comes from.
"""

from __future__ import annotations

import numpy as np

from .metrics import score_labels


def caruana(preds: dict, score_fn, n_iter: int = 30) -> tuple[dict, float]:
    names = list(preds)
    if not names:
        return {}, -np.inf
    singles = {n: score_fn(preds[n]) for n in names}
    first = max(singles, key=singles.get)
    counts = {first: 1}
    current = np.array(preds[first], dtype=float, copy=True)
    total = 1
    best_score, best_counts = singles[first], dict(counts)
    if len(names) == 1:
        return {first: 1.0}, best_score
    stale = 0
    for _ in range(n_iter):
        trial_scores = {n: score_fn((current * total + preds[n]) / (total + 1)) for n in names}
        pick = max(trial_scores, key=trial_scores.get)
        current = (current * total + preds[pick]) / (total + 1)
        total += 1
        counts[pick] = counts.get(pick, 0) + 1
        if trial_scores[pick] > best_score + 1e-9:
            best_score, best_counts, stale = trial_scores[pick], dict(counts), 0
        else:
            stale += 1
            if stale >= 8:
                break
    tot = sum(best_counts.values())
    return {n: c / tot for n, c in best_counts.items()}, best_score


def blend(preds: dict, weights: dict) -> np.ndarray:
    out = None
    for n, w in weights.items():
        out = preds[n] * w if out is None else out + preds[n] * w
    return out


def log_loss_score(y: np.ndarray, P: np.ndarray) -> float:
    p = np.clip(P[np.arange(len(y)), y], 1e-15, 1.0)
    return float(np.mean(np.log(p)))  # higher is better (negative log-loss)


# ---------------------------------------------------------------- decision rules
def apply_rule(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.argmax(P * w[None, :], axis=1)


def fit_rule(P: np.ndarray, y: np.ndarray, metric: str, n_classes: int, pos: int) -> tuple[np.ndarray, float]:
    """Per-class multiplicative weights maximising `metric` on out-of-fold data."""
    w = np.ones(n_classes)
    if metric in ("accuracy", "roc_auc", "log_loss"):
        return w, score_labels("accuracy", y, apply_rule(P, w))

    def sc(weights):
        return score_labels(metric, y, apply_rule(P, weights), pos=pos)

    if n_classes == 2:
        p1 = P[:, 1] / np.maximum(P.sum(axis=1), 1e-12)
        grid = np.unique(np.concatenate([np.quantile(p1, np.linspace(0.01, 0.99, 99)), [0.5]]))
        grid = grid[(grid > 1e-6) & (grid < 1 - 1e-6)]
        if grid.size == 0:
            return w, sc(w)
        scores = np.array([score_labels(metric, y, (p1 >= t).astype(int), pos=pos) for t in grid])
        best = scores.max()
        plateau = grid[scores >= best - 0.002]
        t = float(np.median(plateau))
        w = np.array([t, 1.0 - t])
        return w, sc(w)

    prior = np.bincount(y, minlength=n_classes) / len(y)
    starts = [np.ones(n_classes), 1.0 / np.maximum(prior, 1e-6)]
    best_w, best = None, -np.inf
    for w0 in starts:
        w0 = w0 / w0.max()
        cur, cur_s = w0.copy(), sc(w0)
        for _ in range(3):
            improved = False
            for k in range(n_classes):
                for f in (0.5, 0.7, 0.85, 1.18, 1.43, 2.0):
                    trial = cur.copy()
                    trial[k] *= f
                    s = sc(trial)
                    if s > cur_s + 1e-5:
                        cur, cur_s, improved = trial, s, True
            if not improved:
                break
        if cur_s > best:
            best_w, best = cur, cur_s
    return best_w, best
