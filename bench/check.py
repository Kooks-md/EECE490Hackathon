"""Local checker: output contract validation, scoring, and profile correctness.

    python bench/check.py DATASET_DIR OUTPUT_DIR      # DATASET_DIR has public/ and private/

The profile is checked against an independent, deliberately simple
re-implementation of the documented definitions (the "oracle"), plus any
issues the synthetic generator injected (recorded in private/truth.json).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn import metrics as skm

REQUIRED_PROFILE_KEYS = ("schema_version", "dataset", "feature_types", "target", "issues", "columns")


def _read_public_train(pub: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(pub, "train.csv"), na_values=["?"], keep_default_na=True,
                     skipinitialspace=True, low_memory=False)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].astype("str").str.strip().mask(df[c].isna())
            df[c] = s.mask(s == "")
    return df


def validate_outputs(ds_dir: str, out_dir: str) -> tuple[list, pd.DataFrame | None]:
    errors = []
    pub, priv = os.path.join(ds_dir, "public"), os.path.join(ds_dir, "private")
    labels = pd.read_csv(os.path.join(priv, "labels.csv"))
    truth = json.load(open(os.path.join(priv, "truth.json")))
    path = os.path.join(out_dir, "predictions.csv")
    if not os.path.exists(path):
        return ["predictions.csv missing"], None
    pred = pd.read_csv(path)
    if list(pred.columns[:2]) != ["id", "prediction"]:
        errors.append(f"bad header {list(pred.columns)}")
        return errors, None
    if len(pred) != len(labels):
        errors.append(f"row count {len(pred)} != {len(labels)}")
        return errors, None
    if not (pred["id"].astype(str).to_numpy() == labels["id"].astype(str).to_numpy()).all():
        errors.append("ids differ from expected order")
    if pred["prediction"].isna().any():
        errors.append("missing predictions")
    task = truth["task"]
    if task == "classification":
        train = pd.read_csv(os.path.join(pub, "train.csv"))
        allowed = set(train[truth["target"]].dropna().astype(str))
        bad = set(pred["prediction"].astype(str)) - allowed
        if bad:
            errors.append(f"unknown class labels {sorted(bad)[:5]}")
    elif task == "regression":
        if not np.isfinite(pd.to_numeric(pred["prediction"], errors="coerce")).all():
            errors.append("non-finite regression predictions")
    else:
        vals = pd.to_numeric(pred["prediction"], errors="coerce")
        if vals.isna().any() or not (vals % 1 == 0).all():
            errors.append("cluster labels must be integers")
        elif vals.nunique() > truth["n_clusters"]:
            errors.append(f"{vals.nunique()} clusters > required {truth['n_clusters']}")
    if not os.path.exists(os.path.join(out_dir, "profile.json")):
        errors.append("profile.json missing")
    return errors, pred


def score(ds_dir: str, pred: pd.DataFrame) -> dict:
    priv = os.path.join(ds_dir, "private")
    labels = pd.read_csv(os.path.join(priv, "labels.csv"))
    truth = json.load(open(os.path.join(priv, "truth.json")))
    task, metric = truth["task"], truth.get("metric")
    y, p = labels["label"], pred["prediction"]
    out = {"task": task, "metric": metric}
    if task == "classification":
        ys, ps = y.astype(str).to_numpy(), p.astype(str).to_numpy()
        out["accuracy"] = float(skm.accuracy_score(ys, ps))
        out["f1_macro"] = float(skm.f1_score(ys, ps, average="macro", zero_division=0))
        out["balanced_accuracy"] = float(skm.balanced_accuracy_score(ys, ps))
        out["quality"] = out.get(metric or "f1_macro", out["f1_macro"])
    elif task == "regression":
        yf, pf = y.astype(float).to_numpy(), pd.to_numeric(p, errors="coerce").fillna(0).to_numpy()
        out["r2"] = float(skm.r2_score(yf, pf))
        out["rmse"] = float(np.sqrt(np.mean((yf - pf) ** 2)))
        out["mae"] = float(np.mean(np.abs(yf - pf)))
        out["rmsle"] = float(np.sqrt(np.mean((np.log1p(np.maximum(pf, 0)) - np.log1p(np.maximum(yf, 0))) ** 2)))
        out["quality"] = float(np.clip(out["r2"], -1, 1))
    else:
        out["ari"] = float(skm.adjusted_rand_score(y, p))
        out["nmi"] = float(skm.normalized_mutual_info_score(y, p))
        out["quality"] = out["ari"]
    return out


def check_profile(ds_dir: str, out_dir: str) -> dict:
    """Return {"passed": n, "total": m, "failures": [...]}."""
    pub, priv = os.path.join(ds_dir, "public"), os.path.join(ds_dir, "private")
    fails, total = [], 0

    def check(cond, msg):
        nonlocal total
        total += 1
        if not cond:
            fails.append(msg)

    try:
        prof = json.load(open(os.path.join(out_dir, "profile.json")))
    except (OSError, ValueError) as exc:
        return {"passed": 0, "total": 1, "failures": [f"unreadable profile: {exc}"]}
    for k in REQUIRED_PROFILE_KEYS:
        check(k in prof, f"missing key {k}")
    if fails:
        return {"passed": total - len(fails), "total": total, "failures": fails}
    meta = json.load(open(os.path.join(pub, "metadata.json")))
    truth = json.load(open(os.path.join(priv, "truth.json")))
    train = _read_public_train(pub)
    target, idc = meta.get("target"), meta.get("id_column")
    feats = [c for c in train.columns if c not in (target, idc)]
    cols = {c["name"]: c for c in prof["columns"]}

    check(prof["dataset"].get("n_rows_train") == len(train), "n_rows_train")
    check(prof["dataset"].get("task") == truth["task"], "task")
    check(set(feats) <= set(cols), "every feature column profiled")

    # Oracle: missing counts, constant columns, duplicate columns.
    for c in feats:
        if c not in cols:
            continue
        s = train[c]
        n_miss = int(s.isna().sum())
        # parsing may legitimately turn a few junk strings into NaN; accept >= oracle when numeric
        rep = cols[c]["n_missing"]
        ok = rep == n_miss or (cols[c].get("inferred_type") == "numeric" and n_miss <= rep <= n_miss + 0.05 * len(s))
        check(ok, f"n_missing[{c}] reported {rep} expected {n_miss}")
    oracle_const = {c for c in feats if train[c].nunique(dropna=True) <= 1}
    rep_const = set(prof["issues"].get("constant_columns", []))
    check(oracle_const == rep_const, f"constant columns reported {sorted(rep_const)} expected {sorted(oracle_const)}")
    oracle_dup = set()
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            if b in oracle_dup:
                continue
            if train[a].equals(train[b]):
                oracle_dup.add(b)
    rep_dup = {d["column"] for d in prof["issues"].get("duplicate_columns", [])}
    check(oracle_dup <= rep_dup, f"duplicates reported {sorted(rep_dup)} expected {sorted(oracle_dup)}")

    # Target
    tgt = prof.get("target") or {}
    if truth["task"] == "classification":
        y = train[target].dropna()
        check(tgt.get("n_classes") == y.nunique(), "n_classes")
        counts = y.astype(str).value_counts()
        rep_counts = {str(k): v for k, v in (tgt.get("class_counts") or {}).items()}
        norm = {str(k).replace(".0", ""): v for k, v in rep_counts.items()}
        check(all(norm.get(str(k).replace(".0", ""), -1) == v for k, v in counts.items()), "class_counts")
    elif truth["task"] == "regression":
        y = pd.to_numeric(train[target], errors="coerce").dropna()
        check(tgt.get("min") is not None and abs(tgt["min"] - y.min()) <= 1e-6 * max(1, abs(y.min())), "target min")
    else:
        check(prof["dataset"].get("n_clusters") == meta.get("n_clusters"), "n_clusters")

    # Injected issues (synthetic data)
    inj = truth.get("injected", {})
    for c in inj.get("constant", []):
        check(c in rep_const, f"injected constant {c} not reported")
    dup_groups = [set(g) for g in prof["issues"].get("duplicate_groups", [])]
    for c in inj.get("duplicate", []):
        orig = c[: -len("_copy")] if c.endswith("_copy") else None
        ok = c in rep_dup or any({c, orig} <= g for g in dup_groups)
        check(ok, f"injected duplicate {c} not reported")
    rep_ids = set(prof["issues"].get("id_like_columns", []))
    for c in inj.get("id", []):
        check(c in rep_ids, f"injected id column {c} not reported")
    for c in inj.get("missing", []):
        if c in cols:
            check(cols[c]["n_missing"] > 0, f"injected missing values in {c} not reported")
    if "imbalance_ratio" in inj:
        ci = prof["issues"].get("class_imbalance", {})
        check(bool(ci.get("is_imbalanced")) == (inj["imbalance_ratio"] >= 1.5), "imbalance flag")
    # Scale flag must agree with the documented definition: max/min std of used numerics >= 10.
    used_num = [c["name"] for c in prof["columns"] if c.get("used_for_modelling") and c.get("inferred_type") == "numeric"]
    stds = [float(pd.to_numeric(train[c], errors="coerce").std()) for c in used_num if c in train.columns]
    stds = [s for s in stds if np.isfinite(s) and s > 0]
    if len(stds) >= 2:
        expect = max(stds) / min(stds) >= 10
        check(bool(prof["issues"]["scale"].get("differently_scaled")) == expect, "scale disparity flag")
    return {"passed": total - len(fails), "total": total, "failures": fails}


if __name__ == "__main__":
    ds, out = sys.argv[1], sys.argv[2]
    errs, pred = validate_outputs(ds, out)
    print("contract errors:", errs or "none")
    if pred is not None and not errs:
        print("score:", json.dumps(score(ds, pred), indent=1))
    print("profile:", json.dumps(check_profile(ds, out), indent=1))
