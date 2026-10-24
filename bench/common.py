"""Shared helpers for building benchmark datasets in the competition format.

Layout of one benchmark dataset::

    <name>/public/metadata.json   what the solver sees
    <name>/public/train.csv
    <name>/public/test.csv        (supervised only)
    <name>/private/labels.csv     id,label for every row the solver must predict
    <name>/private/truth.json     task, metric and any injected data issues
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def write_dataset(out_root: str, name: str, df: pd.DataFrame, *, task: str, target: str | None = None,
                  id_column: str | None = None, feature_types: dict | None = None, n_clusters: int | None = None,
                  metric: str | None = None, labels: np.ndarray | None = None, truth: dict | None = None,
                  test_size: float = 0.25, seed: int = 0, declare_types: bool = True) -> str:
    """Split (supervised) and write public/private parts. Returns the dataset directory."""
    base = os.path.join(out_root, name)
    pub, priv = os.path.join(base, "public"), os.path.join(base, "private")
    os.makedirs(pub, exist_ok=True)
    os.makedirs(priv, exist_ok=True)
    meta = {"name": name, "task": task}
    if metric:
        meta["metric"] = metric
    if id_column:
        meta["id_column"] = id_column

    if task == "clustering":
        assert labels is not None and len(labels) == len(df)
        meta["n_clusters"] = int(n_clusters if n_clusters else len(np.unique(labels)))
        df.to_csv(os.path.join(pub, "train.csv"), index=False)
        ids = df[id_column].to_numpy() if id_column else np.arange(len(df))
        pd.DataFrame({"id": ids, "label": labels}).to_csv(os.path.join(priv, "labels.csv"), index=False)
    else:
        meta["target"] = target
        y = df[target]
        strat = y if task == "classification" and y.value_counts().min() >= 2 else None
        tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
        tr.to_csv(os.path.join(pub, "train.csv"), index=False)
        te.drop(columns=[target]).to_csv(os.path.join(pub, "test.csv"), index=False)
        ids = te[id_column].to_numpy() if id_column else np.arange(len(te))
        pd.DataFrame({"id": ids, "label": te[target].to_numpy()}).to_csv(os.path.join(priv, "labels.csv"),
                                                                          index=False)
    if declare_types:
        if feature_types is None:
            feature_types = infer_declared_types(df, exclude={target, id_column})
        meta["feature_types"] = feature_types
    with open(os.path.join(pub, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    t = {"task": task, "metric": metric, "n_clusters": meta.get("n_clusters"), "target": target,
         "id_column": id_column}
    t.update(truth or {})
    with open(os.path.join(priv, "truth.json"), "w") as fh:
        json.dump(t, fh, indent=2, default=str)
    return base


def infer_declared_types(df: pd.DataFrame, exclude: set) -> dict:
    """What an organiser would plausibly declare: dtype-based numeric/categorical."""
    out = {}
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            out[c] = "numeric"
        else:
            out[c] = "categorical"
    return out
