"""Clustering engine.

Pipeline
1. Representation: median-imputed, standardised numerics; each categorical
   becomes a one-hot block scaled to unit total variance, so a 5-level
   categorical weighs the same as one numeric column.
2. Irrelevant-feature removal, in two stages:
   a. Dependence screen. Cluster structure makes the features that carry it
      statistically dependent on each other; a column generated independently
      of everything else is not. Pairwise chi-square tests on discretised
      features drop columns that depend on nothing. This needs no clustering,
      so strong noise columns cannot mask the signal.
   b. Leave-one-out relevance. For each remaining feature, cluster on the
      others and test (chi-square) whether the feature's distribution differs
      across those clusters. Drops features that are related to other columns
      but not to the cluster structure (e.g. sex in the penguins data).
   If a stage finds nothing clearly relevant it keeps everything.
3. Candidates on the kept features: KMeans and Gaussian mixtures with
   spherical / diagonal / tied / full covariance.
4. Selection without labels: the mixture with the lowest BIC, unless its
   partition is less reproducible on 75% subsamples than KMeans' (stability
   lower by more than 0.05), in which case KMeans.

The selection rule and both pruning stages were chosen on 35 labelled
clustering datasets (bench/); see the README for the comparison.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from .analysis import Analysis
from .budget import Budget
from .dataio import Dataset, OutputWriter

SUBSAMPLE_SELECT = 4000      # rows used for feature selection
MAX_FIT_ROWS = 20000         # mixtures are fitted on at most this many rows, then predict all
SCREEN_MAX_FEATURES = 300
LOO_MAX_FEATURES = 40
GMM_FULL_MAX_DIM = 30
STABILITY_MARGIN = 0.05
COVARIANCES = ("spherical", "diag", "tied", "full")


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Deterministic labels: 0 = largest cluster."""
    uniq, counts = np.unique(labels, return_counts=True)
    order = uniq[np.argsort(-counts, kind="stable")]
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[v] for v in labels], dtype=int)


# --------------------------------------------------------------------- representation
def build_matrix(an: Analysis):
    """Return (Z, groups, names, kinds) for all rows (train then rows-to-predict)."""
    frames = [an.train[an.kept]]
    if an.pred is not None:
        frames.append(an.pred[an.kept])
    data = pd.concat(frames, ignore_index=True) if an.kept else pd.DataFrame(index=range(len(an.train)))
    n = len(data)
    cols, groups, names, kinds = [], [], [], []
    for c in an.kept:
        mt = an.info(c).model_type
        s = data[c]
        if mt == "datetime":
            dt = pd.to_datetime(s, errors="coerce")
            s = (dt - dt.min()).dt.total_seconds()
            mt = "numeric"
        if mt == "numeric":
            x = s.to_numpy(dtype=float)
            fin = np.isfinite(x)
            if fin.sum() < 2:
                continue
            x = np.where(fin, x, np.median(x[fin]))
            sd = x.std()
            if sd <= 1e-12:
                continue
            groups.append([len(cols)])
            cols.append((x - x.mean()) / sd)
            names.append(c)
            kinds.append("numeric")
        elif mt == "categorical":
            v = s.astype("object")
            mode = v.mode(dropna=True)
            v = v.fillna(mode.iloc[0] if len(mode) else "__nan__")
            counts = v.value_counts()
            rare = counts[counts < max(2, int(0.01 * n))].index
            if len(rare) >= 1 and len(counts) > len(rare):
                v = v.where(~v.isin(rare), "__rare__")
                counts = v.value_counts()
            if len(counts) < 2 or len(counts) > 50:
                continue
            onehot = np.array(pd.get_dummies(v).to_numpy(dtype=float), copy=True)  # pandas CoW: writable copy
            onehot -= onehot.mean(axis=0)
            tot = np.sqrt(onehot.var(axis=0).sum())
            if tot <= 1e-12:
                continue
            onehot /= tot
            groups.append(list(range(len(cols), len(cols) + onehot.shape[1])))
            cols.extend(onehot.T)
            names.append(c)
            kinds.append("categorical")
    Z = np.column_stack(cols) if cols else np.zeros((n, 0))
    return Z, groups, names, kinds


# --------------------------------------------------------------------- feature selection
def _codes(Z, group, kind, q):
    """Discretise one feature group: categories, or equal-frequency bins for numerics."""
    if kind == "categorical":
        return np.argmax(Z[:, group], axis=1)
    ranks = stats.rankdata(Z[:, group[0]], method="average")
    return np.minimum(((ranks - 1) * q) // len(ranks), q - 1).astype(int)


def _n_bins(n: int) -> int:
    return int(np.clip(n // 60, 3, 8))


def screen_dependence(Z, groups, kinds, alpha=1e-3, v_min=0.1):
    """Keep features that depend on at least one other feature (see module docstring)."""
    G, n = len(groups), Z.shape[0]
    if G < 3 or n < 20:
        return list(range(G))
    q = _n_bins(n)
    codes = [_codes(Z, groups[g], kinds[g], q) for g in range(G)]
    sizes = [int(c.max()) + 1 for c in codes]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    B = np.zeros((n, offsets[-1]))
    for g, c in enumerate(codes):
        B[np.arange(n), offsets[g] + c] = 1.0
    C = B.T @ B                                  # all pairwise contingency tables at once
    p_thr = alpha / (G * (G - 1) / 2)            # Bonferroni over pairs
    passed = np.zeros(G, dtype=bool)
    for i in range(G):
        for j in range(i + 1, G):
            if passed[i] and passed[j]:
                continue
            t = C[offsets[i]:offsets[i + 1], offsets[j]:offsets[j + 1]]
            t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
            if t.shape[0] < 2 or t.shape[1] < 2:
                continue
            expected = t.sum(axis=1, keepdims=True) * t.sum(axis=0, keepdims=True) / n
            chi2 = float(((t - expected) ** 2 / np.maximum(expected, 1e-12)).sum())
            v = np.sqrt(chi2 / (n * (min(t.shape) - 1)))
            if v >= v_min and stats.chi2.sf(chi2, (t.shape[0] - 1) * (t.shape[1] - 1)) < p_thr:
                passed[i] = passed[j] = True
    kept = [g for g in range(G) if passed[g]]
    return kept if len(kept) >= 2 else list(range(G))


def _kmeans(Z, k, seed, n_init=4, max_iter=150):
    return KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=seed).fit_predict(Z)


def _relevance(Z, group, kind, labels):
    """(Cramer's V^2, p) of feature (discretised) vs a partition; a distributional test, not just means."""
    codes = _codes(Z, group, kind, _n_bins(len(labels)))
    table = pd.crosstab(labels, codes).to_numpy()
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, 1.0
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    return float(chi2 / (len(labels) * (min(table.shape) - 1))), float(p)


def prune_features(Z, groups, kinds, k, seed, candidates=None, alpha=1e-3, min_v2=0.01, rel_v2=0.05,
                   max_rounds=3, deadline=None):
    """Leave-one-out relevance (see module docstring). Returns kept group indices."""
    pool = list(range(len(groups))) if candidates is None else list(candidates)
    if len(pool) <= 1:
        return pool
    keep = list(pool)
    a = alpha / len(pool)
    for _ in range(max_rounds):
        base = _kmeans(Z[:, [c for g in keep for c in groups[g]]], k, seed)
        v2, pv = {}, {}
        for g in pool:
            if deadline is not None and time.time() > deadline:
                return list(pool)
            if g in keep:
                others = [c for h in keep if h != g for c in groups[h]]
                if not others:
                    v2[g], pv[g] = 1.0, 0.0
                    continue
                labels = _kmeans(Z[:, others], k, seed + g + 1, n_init=2)
            else:
                labels = base
            v2[g], pv[g] = _relevance(Z, groups[g], kinds[g], labels)
        thr = max(min_v2, rel_v2 * max(v2.values()))
        new_keep = [g for g in pool if pv[g] < a and v2[g] >= thr]
        if not new_keep:
            return list(pool)
        if set(new_keep) == set(keep):
            break
        keep = new_keep
    return keep


# --------------------------------------------------------------------- candidates & selection
def _gmm(Z, k, cov, seed, n_init=2):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GaussianMixture(n_components=k, covariance_type=cov, n_init=n_init, reg_covar=1e-4,
                               random_state=seed, max_iter=200).fit(Z)


def candidate_partitions(Z, k, seed, deadline=None):
    """name -> (labels, info). Fits on a subsample for very large tables, then predicts all rows."""
    n = Z.shape[0]
    rng = np.random.default_rng(seed)
    fit_rows = np.sort(rng.choice(n, MAX_FIT_ROWS, replace=False)) if n > MAX_FIT_ROWS else None
    Zf = Z[fit_rows] if fit_rows is not None else Z
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Zf)
    out = {"kmeans": (km.predict(Z) if fit_rows is not None else km.labels_, {"inertia": float(km.inertia_)})}
    best = None
    for cov in COVARIANCES:
        if deadline is not None and time.time() > deadline:
            break
        if cov == "full" and Z.shape[1] > GMM_FULL_MAX_DIM:
            continue
        try:
            gm = _gmm(Zf, k, cov, seed)
            bic = float(gm.bic(Zf))
            out[f"gmm_{cov}"] = (gm.predict(Z), {"bic": round(bic, 3)})
            if best is None or bic < best[1]:
                best = (f"gmm_{cov}", bic)
        except Exception:  # noqa: BLE001
            continue
    if best is not None:
        out["gmm_best"] = (out[best[0]][0], {"bic": round(best[1], 3), "covariance": best[0]})
    return out


def _fit_algo(name: str, Z, k, seed):
    if name == "kmeans":
        return KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(Z)
    return _gmm(Z, k, name[4:], seed, n_init=1).predict(Z)


def stability_scores(Z, k, cands: dict, seed, names, B=3, frac=0.75, max_n=2000, deadline=None):
    """Mean ARI between each candidate's partition and refits on random 75% subsamples."""
    rng = np.random.default_rng(seed + 991)
    n = len(Z)
    base = np.sort(rng.choice(n, max_n, replace=False)) if n > max_n else np.arange(n)
    out = {c: [] for c in names}
    for b in range(B):
        rows = base[np.sort(rng.choice(len(base), int(frac * len(base)), replace=False))]
        for c in names:
            if deadline is not None and time.time() > deadline:
                break
            try:
                out[c].append(adjusted_rand_score(cands[c][0][rows], _fit_algo(c, Z[rows], k, seed + 17 * (b + 1))))
            except Exception:  # noqa: BLE001
                out[c].append(0.0)
    return {c: float(np.mean(v)) for c, v in out.items() if v}


def _sizes_ok(labels, k, min_frac=0.01):
    counts = np.bincount(labels, minlength=k)
    return (counts > 0).sum() == k and counts.min() >= max(2, min_frac * len(labels))


def select(cands: dict, Z, k, seed, deadline=None) -> tuple[str, dict]:
    """Lowest-BIC mixture unless KMeans is clearly more reproducible."""
    if "gmm_best" not in cands or not _sizes_ok(cands["gmm_best"][0], k):
        return "kmeans", {}
    g = cands["gmm_best"][1]["covariance"]
    stab = stability_scores(Z, k, cands, seed, names=["kmeans", g], deadline=deadline)
    if g in stab and "kmeans" in stab and stab[g] >= stab["kmeans"] - STABILITY_MARGIN:
        return "gmm_best", stab
    return "kmeans", stab


def choose_k(Z, seed, k_max=10):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(len(Z), 3000), replace=False)
    best, best_s = 2, -np.inf
    for k in range(2, min(k_max, len(idx) - 1) + 1):
        lab = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(Z[idx])
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(Z[idx], lab)
        if s > best_s:
            best, best_s = k, s
    return best


# --------------------------------------------------------------------- entry points
def cluster(an: Analysis, k: int | None, seed: int, budget: Budget | None = None):
    report: dict = {}
    Z, groups, names, kinds = build_matrix(an)
    n = Z.shape[0]
    report["features"] = names
    if Z.shape[1] == 0 or n < 2:
        report["note"] = "no usable features; single cluster"
        return np.zeros(n, dtype=int), report
    deadline = budget.soft_end if budget is not None else None
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(n, SUBSAMPLE_SELECT, replace=False)) if n > SUBSAMPLE_SELECT else np.arange(n)

    keep = list(range(len(groups)))
    if 3 <= len(groups) <= SCREEN_MAX_FEATURES:
        keep = screen_dependence(Z[sub], groups, kinds)
    report["dropped_independent_of_all_others"] = [names[g] for g in range(len(groups)) if g not in keep]

    if k is None:
        k = choose_k(Z[:, [c for g in keep for c in groups[g]]], seed)
        report["k_chosen_by_silhouette"] = k
    k = int(max(1, min(k, n)))
    if k == 1:
        return np.zeros(n, dtype=int), report

    screened = list(keep)
    if 2 <= len(keep) <= LOO_MAX_FEATURES:
        keep = prune_features(Z[sub], groups, kinds, k, seed, candidates=keep, deadline=deadline)
    report["dropped_unrelated_to_clusters"] = [names[g] for g in screened if g not in keep]
    report["features_used"] = [names[g] for g in keep]
    Zk = Z[:, [c for g in keep for c in groups[g]]]

    cands = candidate_partitions(Zk, k, seed, deadline=deadline)
    choice, stab = select(cands, Zk, k, seed, deadline=deadline)
    labels = cands[choice][0]
    report["algorithm"] = "kmeans" if choice == "kmeans" else cands["gmm_best"][1]["covariance"]
    report["n_clusters"] = k
    report["candidates"] = {c: info for c, (_, info) in cands.items() if c != "gmm_best"}
    report["stability"] = {c: round(v, 4) for c, v in stab.items()}
    return labels, report


def run_clustering(ds: Dataset, an: Analysis, budget: Budget, writer: OutputWriter, seed: int,
                   n_threads: int, log) -> dict:
    n_out = len(ds.pred_ids)
    k = ds.n_clusters
    # Safety net first: KMeans on the plain standardised matrix.
    try:
        Z, _, _, _ = build_matrix(an)
        if Z.shape[1] and k and k > 1:
            base = KMeans(n_clusters=int(min(k, len(Z))), n_init=3, random_state=seed).fit_predict(Z)
            writer.predictions(relabel_by_size(base[-n_out:]), label_values=True)
        else:
            writer.predictions(np.zeros(n_out, dtype=int), label_values=True)
    except Exception:  # noqa: BLE001
        writer.predictions(np.arange(n_out) % max(1, int(k or 1)), label_values=True)
    labels, report = cluster(an, k, seed, budget)
    writer.predictions(relabel_by_size(labels[-n_out:]), label_values=True)
    log(f"clustering: {report.get('algorithm')} on {len(report.get('features_used', []))} features")
    return report
