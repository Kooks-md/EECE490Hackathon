"""Structured data profile (written to profile.json).

All statistics describe the training table (for clustering: the table being
clustered) after parsing; the target column is profiled separately and the
id column is listed with role "id". See analysis.py for the exact definitions
of missing / constant / duplicate / id-like / high-cardinality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import Analysis
from .dataio import Dataset

SCHEMA_VERSION = "1.0"
SCALE_RATIO_THRESHOLD = 10.0
SKEW_THRESHOLD = 1.0
CORRELATION_THRESHOLD = 0.95


def _r(x, digits: int = 12):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    if not np.isfinite(xf):
        return None
    return float(f"{xf:.{digits}g}")


def imbalance_level(ratio: float) -> str:
    if ratio < 1.5:
        return "balanced"
    if ratio < 3:
        return "mild"
    if ratio < 10:
        return "moderate"
    return "severe"


def _numeric_stats(s: pd.Series) -> dict:
    v = s.dropna().astype(float)
    if v.empty:
        return {"min": None, "max": None, "mean": None, "std": None, "median": None, "skewness": None}
    return {
        "min": _r(v.min()), "max": _r(v.max()), "mean": _r(v.mean()),
        "std": _r(v.std(ddof=1)) if len(v) > 1 else 0.0, "median": _r(v.median()),
        "skewness": _r(v.skew()) if len(v) > 2 else 0.0,
    }


def target_profile(ds: Dataset) -> dict | None:
    if ds.task == "clustering":
        return {"name": None, "type": "none", "n_clusters": ds.n_clusters}
    y = ds.train[ds.target]
    out = {"name": ds.target, "n_missing": int(y.isna().sum())}
    y = y.dropna()
    if ds.task == "classification":
        counts = y.value_counts()
        n_classes = int(counts.size)
        out["type"] = "binary" if n_classes == 2 else ("multiclass" if n_classes > 2 else "single_class")
        out["n_classes"] = n_classes
        out["class_counts"] = {str(_label_key(k)): int(v) for k, v in counts.items()}
        if n_classes:
            ratio = float(counts.max() / counts.min())
            out["majority_class"] = str(_label_key(counts.idxmax()))
            out["minority_class"] = str(_label_key(counts.idxmin()))
            out["minority_fraction"] = _r(counts.min() / counts.sum())
            out["imbalance_ratio"] = _r(ratio)
            out["imbalance_level"] = imbalance_level(ratio)
            out["is_imbalanced"] = bool(ratio >= 1.5)
    else:
        yf = pd.to_numeric(y, errors="coerce").dropna()
        out["type"] = "continuous"
        out.update(_numeric_stats(yf))
        vals = yf.to_numpy()
        out["is_integer"] = bool(vals.size and np.all(np.mod(vals, 1) == 0))
        out["is_non_negative"] = bool(vals.size and np.all(vals >= 0))
        out["is_skewed"] = bool(out.get("skewness") is not None and abs(out["skewness"]) > SKEW_THRESHOLD)
    return out


def _label_key(v):
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return int(v)
    return v.item() if hasattr(v, "item") else v


def _duplicate_groups(features) -> list:
    """Sets of identical columns, e.g. [["a", "a_copy"]] - symmetric view of duplicate_of."""
    groups: dict[str, list] = {}
    for i in features:
        if i.duplicate_of:
            groups.setdefault(i.duplicate_of, [i.duplicate_of]).append(i.name)
    return list(groups.values())


def build_profile(ds: Dataset, an: Analysis) -> dict:
    n_train = len(ds.train)
    columns = []
    type_lists = {"numeric": [], "categorical": [], "datetime": [], "text": [], "boolean": [], "id": []}

    for info in an.columns:
        entry = {
            "name": info.name, "role": info.role, "declared_type": info.declared_type,
            "inferred_type": info.inferred_type, "n_missing": info.n_missing,
            "missing_fraction": _r(info.missing_fraction), "n_unique": info.n_unique,
            "is_constant": info.is_constant, "is_id_like": info.is_id_like,
            "duplicate_of": info.duplicate_of, "high_cardinality": info.high_cardinality,
            "used_for_modelling": info.role == "feature" and not info.dropped,
            "drop_reason": info.drop_reason,
        }
        if info.notes:
            entry["notes"] = info.notes
        type_lists.setdefault(info.inferred_type, []).append(info.name)
        if info.role == "feature":
            s = an.train[info.name]
            if info.inferred_type in ("numeric", "boolean"):
                entry["stats"] = _numeric_stats(s)
            elif info.inferred_type == "datetime":
                v = s.dropna()
                entry["stats"] = {"min": v.min() if len(v) else None, "max": v.max() if len(v) else None}
            else:
                vc = s.value_counts().head(5)
                entry["stats"] = {"top_values": [{"value": str(k), "count": int(c)} for k, c in vc.items()]}
        columns.append(entry)

    features = [i for i in an.columns if i.role == "feature"]
    missing_cols = [
        {"name": i.name, "n_missing": i.n_missing, "missing_fraction": _r(i.missing_fraction)}
        for i in features if i.n_missing > 0
    ]
    total_missing = int(sum(i.n_missing for i in features))
    n_cells = n_train * len(features)

    # Scale disparity among numeric features that are used for modelling
    num_cols = [i.name for i in features if i.inferred_type == "numeric" and not i.dropped]
    stds = {}
    for c in num_cols:
        v = an.train[c].dropna()
        if len(v) > 1:
            sd = float(v.std(ddof=1))
            if np.isfinite(sd) and sd > 0:
                stds[c] = sd
    scale = {"numeric_std_min": None, "numeric_std_max": None, "std_ratio": None, "differently_scaled": False}
    if stds:
        lo, hi = min(stds.values()), max(stds.values())
        scale = {
            "numeric_std_min": _r(lo), "numeric_std_max": _r(hi), "std_ratio": _r(hi / lo),
            "differently_scaled": bool(hi / lo >= SCALE_RATIO_THRESHOLD),
            "smallest_scale_column": min(stds, key=stds.get), "largest_scale_column": max(stds, key=stds.get),
        }

    skewed = []
    for c in num_cols:
        v = an.train[c].dropna()
        if v.nunique() > 2 and len(v) > 2:
            sk = v.skew()
            if np.isfinite(sk) and abs(sk) > SKEW_THRESHOLD:
                skewed.append({"name": c, "skewness": _r(sk)})

    corr_pairs = []
    if 1 < len(num_cols) <= 300 and n_train > 2:
        sub = an.train[num_cols].astype(float)
        with np.errstate(all="ignore"):
            corr = sub.corr().to_numpy()
        iu = np.triu_indices(len(num_cols), k=1)
        mask = np.abs(corr[iu]) >= CORRELATION_THRESHOLD
        for a, b, r in zip(iu[0][mask], iu[1][mask], corr[iu][mask]):
            corr_pairs.append({"a": num_cols[a], "b": num_cols[b], "r": _r(r, 4)})
        corr_pairs = sorted(corr_pairs, key=lambda d: -abs(d["r"]))[:50]

    row_cols = [c for c in ds.train.columns if c != ds.id_column]
    try:
        dup_rows = int(ds.train[row_cols].duplicated().sum()) if row_cols else 0
    except TypeError:
        dup_rows = None

    tgt = target_profile(ds)
    issues = {
        "missing_values": {
            "total_missing_cells": total_missing,
            "missing_cell_fraction": _r(total_missing / n_cells) if n_cells else 0.0,
            "columns_with_missing": missing_cols,
        },
        "constant_columns": [i.name for i in features if i.is_constant],
        "duplicate_columns": [{"column": i.name, "duplicate_of": i.duplicate_of} for i in features if i.duplicate_of],
        "duplicate_groups": _duplicate_groups(features),
        "id_like_columns": [i.name for i in an.columns if i.is_id_like],
        "high_cardinality_columns": [i.name for i in features if i.high_cardinality],
        "scale": scale,
        "skewed_columns": skewed,
        "highly_correlated_pairs": corr_pairs,
        "duplicate_rows": dup_rows,
    }
    if ds.task == "classification" and tgt is not None and "imbalance_ratio" in tgt:
        issues["class_imbalance"] = {
            "imbalance_ratio": tgt["imbalance_ratio"], "level": tgt["imbalance_level"],
            "is_imbalanced": tgt["is_imbalanced"], "minority_class": tgt["minority_class"],
            "minority_fraction": tgt["minority_fraction"],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "name": ds.name, "task": ds.task, "target": ds.target, "id_column": ds.id_column,
            "metric": ds.metric, "n_clusters": ds.n_clusters,
            "n_rows_train": n_train, "n_rows_predict": int(len(ds.pred_ids)), "predict_on": ds.predict_on,
            "n_columns": int(ds.train.shape[1]), "n_features": len(features),
            "n_features_used": len(an.kept),
        },
        "feature_types": {k: v for k, v in type_lists.items() if v},
        "target": tgt,
        "issues": issues,
        "columns": columns,
        "notes": list(ds.notes) + list(an.notes),
    }
