"""Synthetic benchmark datasets with the data issues listed in the brief.

    python bench/make_synthetic.py OUT_DIR [--per-task 10] [--seed 0]

Each dataset starts from a known generator (linear, nonlinear, interactions,
blobs, anisotropic clusters, count data with a datetime column ...) and then
gets a random subset of: missing values, rescaled columns, numeric features
turned into categoricals, constant columns, duplicate columns, irrelevant
noise columns, an ID column, and class imbalance. Everything injected is
recorded in private/truth.json so the profile can be checked against it.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification, make_friedman1, make_regression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import write_dataset  # noqa: E402


# ------------------------------------------------------------------ base generators
def cls_linear(rng, n):
    k = int(rng.choice([2, 2, 3, 4]))
    imb = rng.random() < 0.5
    weights = None
    if imb:
        w = np.array([1.0] + list(rng.uniform(0.05, 0.3, k - 1)))
        weights = list(w / w.sum())
    n_inf = int(rng.integers(3, 9))
    X, y = make_classification(n_samples=n, n_features=n_inf + 2, n_informative=n_inf, n_redundant=2,
                               n_classes=k, weights=weights, class_sep=float(rng.uniform(0.6, 1.4)),
                               flip_y=0.02, random_state=int(rng.integers(1e9)))
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]), y, {"generator": "make_classification"}


def cls_nonlinear(rng, n):
    d = int(rng.integers(4, 9))
    X = rng.uniform(-2, 2, size=(n, d))
    f = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 - 1.2 + 0.8 * X[:, 2] * X[:, 3]
    if rng.random() < 0.5:
        y = (f + rng.normal(0, 0.3, n) > 0).astype(int)
    else:
        y = np.digitize(f + rng.normal(0, 0.3, n), np.quantile(f, [0.33, 0.66]))
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y, {"generator": "nonlinear"}


def cls_mixed(rng, n):
    """Telco-like: categorical drivers, interactions, a minority positive class."""
    contract = rng.choice(["monthly", "one_year", "two_year"], n, p=[0.55, 0.25, 0.2])
    service = rng.choice(["dsl", "fiber", "none"], n, p=[0.35, 0.45, 0.2])
    tenure = rng.integers(0, 72, n).astype(float)
    charges = 20 + 60 * (service == "fiber") + 25 * (service == "dsl") + rng.normal(0, 10, n)
    support = rng.choice(["yes", "no"], n)
    logit = (-1.0 + 1.6 * (contract == "monthly") - 0.9 * (contract == "two_year") + 0.9 * (service == "fiber")
             - 0.04 * tenure + 0.015 * charges - 0.6 * (support == "yes") + 0.8 * (contract == "monthly")
             * (service == "fiber"))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"contract": contract, "service": service, "tenure": tenure, "monthly_charges": charges,
                       "tech_support": support, "total_charges": tenure * charges + rng.normal(0, 50, n)})
    labels = np.where(y == 1, "churn", "stay")
    return df, labels, {"generator": "mixed_categorical"}


def reg_linear(rng, n):
    n_inf = int(rng.integers(3, 10))
    X, y = make_regression(n_samples=n, n_features=n_inf + 2, n_informative=n_inf,
                           noise=float(rng.uniform(5, 40)), random_state=int(rng.integers(1e9)))
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]), y, {"generator": "make_regression"}


def reg_friedman(rng, n):
    X, y = make_friedman1(n_samples=n, n_features=7, noise=float(rng.uniform(0.5, 2.0)),
                          random_state=int(rng.integers(1e9)))
    return pd.DataFrame(X, columns=[f"v{i}" for i in range(7)]), y, {"generator": "friedman1"}


def reg_skewed(rng, n):
    """House-price-like: multiplicative effects and a log-normal target."""
    area = rng.lognormal(7.0, 0.35, n)
    quality = rng.integers(1, 11, n).astype(float)
    hood = rng.choice([f"N{i}" for i in range(12)], n)
    hood_eff = {f"N{i}": e for i, e in enumerate(rng.normal(0, 0.25, 12))}
    age = rng.integers(0, 100, n).astype(float)
    log_price = (10 + 0.8 * np.log(area) + 0.12 * quality + np.vectorize(hood_eff.get)(hood) - 0.004 * age
                 + rng.normal(0, 0.15, n))
    df = pd.DataFrame({"area": area, "quality": quality, "neighbourhood": hood, "age": age,
                       "rooms": np.round(area / 250 + rng.normal(0, 1, n)).clip(1)})
    return df, np.exp(log_price), {"generator": "lognormal_price"}


def reg_counts(rng, n):
    """Bike-sharing-like: hourly counts driven by hour x working day, weather, temperature."""
    start = pd.Timestamp("2021-01-01")
    ts = start + pd.to_timedelta(np.sort(rng.choice(24 * 730, n, replace=False)), unit="h")
    hour, dow = ts.hour.to_numpy(), ts.dayofweek.to_numpy()
    working = (dow < 5).astype(int)
    temp = 15 + 10 * np.sin(2 * np.pi * (ts.dayofyear.to_numpy() - 100) / 365) + rng.normal(0, 3, n)
    weather = rng.choice([1, 2, 3, 4], n, p=[0.6, 0.25, 0.13, 0.02])
    peak = working * (np.exp(-((hour - 8) ** 2) / 3) + np.exp(-((hour - 17.5) ** 2) / 4)) * 3
    leisure = (1 - working) * np.exp(-((hour - 14) ** 2) / 12) * 2
    mu = np.exp(1.5 + peak + leisure + 0.04 * temp - 0.35 * (weather - 1) + 0.2 * (ts.year.to_numpy() - 2021))
    y = rng.poisson(mu * 4)
    df = pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "temp": temp,
                       "humidity": rng.uniform(20, 100, n), "weather": weather,
                       "holiday": (rng.random(n) < 0.03).astype(int)})
    return df, y.astype(float), {"generator": "hourly_counts"}


def clu_blobs(rng, n):
    k = int(rng.integers(3, 7))
    d = int(rng.integers(2, 7))
    X, y = make_blobs(n_samples=n, n_features=d, centers=k, cluster_std=float(rng.uniform(0.8, 2.2)),
                      center_box=(-8, 8), random_state=int(rng.integers(1e9)))
    return pd.DataFrame(X, columns=[f"m{i}" for i in range(d)]), y, {"generator": "blobs"}


def clu_aniso(rng, n):
    k = int(rng.integers(3, 6))
    d = int(rng.integers(2, 5))
    X, y = make_blobs(n_samples=n, n_features=d, centers=k, cluster_std=1.0, center_box=(-6, 6),
                      random_state=int(rng.integers(1e9)))
    A = rng.normal(0, 1, (d, d))
    X = X @ A
    return pd.DataFrame(X, columns=[f"a{i}" for i in range(d)]), y, {"generator": "anisotropic"}


def clu_unequal(rng, n):
    k = int(rng.integers(3, 6))
    sizes = rng.dirichlet(np.ones(k) * 2) * n
    sizes = np.maximum(sizes.astype(int), 15)
    d = int(rng.integers(2, 6))
    centers = rng.uniform(-7, 7, (k, d))
    stds = rng.uniform(0.5, 2.0, k)
    parts, labs = [], []
    for j in range(k):
        parts.append(rng.normal(centers[j], stds[j], (sizes[j], d)))
        labs.append(np.full(sizes[j], j))
    X, y = np.vstack(parts), np.concatenate(labs)
    perm = rng.permutation(len(y))
    return pd.DataFrame(X[perm], columns=[f"u{i}" for i in range(d)]), y[perm], {"generator": "unequal_blobs"}


def clu_customers(rng, n):
    """Mall-like segments in income x spending, plus age/gender that do not define segments."""
    centers = np.array([[25, 20], [25, 80], [55, 50], [90, 18], [90, 82]])
    k = len(centers)
    y = rng.integers(0, k, n)
    income = centers[y, 0] + rng.normal(0, 7, n)
    spend = centers[y, 1] + rng.normal(0, 7, n)
    df = pd.DataFrame({"gender": rng.choice(["Male", "Female"], n), "age": rng.integers(18, 70, n).astype(float),
                       "annual_income_k": income.round(1), "spending_score": spend.clip(1, 100).round()})
    return df, y, {"generator": "customer_segments"}


GENERATORS = {
    "classification": [cls_linear, cls_nonlinear, cls_mixed],
    "regression": [reg_linear, reg_friedman, reg_skewed, reg_counts],
    "clustering": [clu_blobs, clu_aniso, clu_unequal, clu_customers],
}


# ------------------------------------------------------------------ corruptions
def corrupt(df: pd.DataFrame, rng, task: str) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    inj: dict = {"constant": [], "duplicate": [], "id": [], "missing": [], "noise": [], "differently_scaled": [],
                 "binned": []}
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    n = len(df)
    menu = ["missing", "scale", "constant", "duplicate", "noise", "binned", "id"]
    chosen = set(rng.choice(menu, size=int(rng.integers(3, 6)), replace=False))
    if "scale" in chosen and num_cols:
        for c in rng.choice(num_cols, size=min(len(num_cols), int(rng.integers(1, 4))), replace=False):
            factor = 10 ** float(rng.uniform(-3, 4))
            df[c] = df[c] * factor + float(rng.uniform(-1, 1)) * factor * 5
            inj["differently_scaled"].append(c)
    if "binned" in chosen and len(num_cols) > 2 and task != "clustering":
        c = str(rng.choice(num_cols))
        q = int(rng.integers(3, 6))
        df[c] = pd.qcut(df[c], q, labels=[f"level_{i}" for i in range(q)]).astype(str)
        inj["binned"].append(c)
    if "noise" in chosen:
        for i in range(int(rng.integers(2, 9))):
            name = f"noise_{i}"
            r = rng.random()
            if r < 0.4:
                df[name] = rng.normal(0, float(rng.uniform(0.5, 50)), n)
            elif r < 0.75:
                df[name] = rng.uniform(0, float(rng.uniform(1, 1000)), n)
            else:
                df[name] = rng.choice(["p", "q", "r", "s"], n)
            inj["noise"].append(name)
    if "constant" in chosen:
        for i in range(int(rng.integers(1, 3))):
            name = f"const_{i}"
            df[name] = rng.choice([0.0, 1.0, 7.5]) if rng.random() < 0.6 else "same"
            inj["constant"].append(name)
    if "duplicate" in chosen:
        cand = [c for c in df.columns if c not in inj["constant"]]
        for c in rng.choice(cand, size=min(len(cand), int(rng.integers(1, 3))), replace=False):
            name = f"{c}_copy"
            df[name] = df[c]
            inj["duplicate"].append(name)
    if "missing" in chosen:
        cand = [c for c in df.columns if c not in inj["constant"] and not c.endswith("_copy")]
        cand = [c for c in cand if f"{c}_copy" not in df.columns]
        for c in rng.choice(cand, size=min(len(cand), int(rng.integers(1, 4))), replace=False):
            mask = rng.random(n) < float(rng.uniform(0.03, 0.3))
            df.loc[mask, c] = np.nan
            inj["missing"].append(str(c))
    if "id" in chosen:
        if rng.random() < 0.5:
            df.insert(0, "record_id", np.arange(1000, 1000 + n))
        else:
            df.insert(0, "record_id", [f"R{j:07d}" for j in rng.permutation(n)])
        inj["id"].append("record_id")
    # shuffle column order a little so position carries no information
    cols = list(df.columns)
    rng.shuffle(cols)
    if "record_id" in cols:
        cols.remove("record_id")
        cols.insert(0, "record_id")
    return df[cols], inj


def build(out: str, per_task: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for task, gens in GENERATORS.items():
        for i in range(per_task):
            gen = gens[i % len(gens)]
            n = int(rng.choice([300, 800, 1500, 3000, 6000, 12000] if task != "clustering" else
                               [200, 400, 800, 1500, 3000, 6000]))
            if i == per_task - 1 and task != "clustering":
                n = 40000  # one large table per supervised task to exercise the time budget
            X, y, info = gen(rng, n)
            X, inj = corrupt(X, rng, task)
            name = f"syn_{task[:3]}_{i:02d}_{gen.__name__}"
            truth = {"injected": inj, "generator": info}
            declared = {c: ("numeric" if pd.api.types.is_numeric_dtype(X[c]) else "categorical")
                        for c in X.columns if c != "record_id"}
            if "timestamp" in X.columns:
                declared["timestamp"] = "datetime"
            if task == "clustering":
                write_dataset(out, name, X, task=task, labels=y, metric="ari", truth=truth, feature_types=declared)
            else:
                df = X.copy()
                df["target"] = y
                if task == "classification":
                    counts = pd.Series(y).value_counts()
                    truth["injected"]["imbalance_ratio"] = float(counts.max() / counts.min())
                metric = {"classification": "f1_macro", "regression": "rmse"}[task]
                write_dataset(out, name, df, task=task, target="target", metric=metric, truth=truth,
                              feature_types=declared, seed=int(rng.integers(1e6)))
            print(f"wrote {name} n={n} injected={ {k: v for k, v in inj.items() if v} }")


def build_clustering_validation(out: str, n_sets: int = 16, seed: int = 1) -> None:
    """Second, independent clustering set (different seed and sizes) used to validate the
    clustering selection rule chosen on the first set."""
    rng = np.random.default_rng(seed)
    gens = GENERATORS["clustering"]
    for i in range(n_sets):
        gen = gens[i % len(gens)]
        n = int(rng.choice([200, 400, 800, 1500, 3000]))
        X, y, info = gen(rng, n)
        X, inj = corrupt(X, rng, "clustering")
        declared = {c: ("numeric" if pd.api.types.is_numeric_dtype(X[c]) else "categorical")
                    for c in X.columns if c != "record_id"}
        name = f"val_clu_{i:02d}_{gen.__name__}"
        write_dataset(out, name, X, task="clustering", labels=y, metric="ari", truth={"injected": inj},
                      feature_types=declared)
        print(f"wrote {name} n={n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-task", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clustering-validation", action="store_true",
                    help="build the 16-table clustering validation set (seed 1) instead")
    a = ap.parse_args()
    if a.clustering_validation:
        build_clustering_validation(a.out, seed=a.seed if a.seed != 0 else 1)
    else:
        build(a.out, a.per_task, a.seed)
