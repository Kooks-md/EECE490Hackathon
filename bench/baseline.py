"""Reference baseline with the same input/output contract, for comparison only.

    python bench/baseline.py DATASET_DIR OUTPUT_DIR

What a typical organiser baseline looks like: median imputation, standard
scaling, one-hot categoricals, then logistic regression / ridge / KMeans.
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


def main(data_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(data_dir, "metadata.json")))
    task, target, idc = meta["task"], meta.get("target"), meta.get("id_column")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_path = os.path.join(data_dir, "test.csv")
    test = pd.read_csv(test_path) if os.path.exists(test_path) else None
    pred_df = test if test is not None else train
    ids = pred_df[idc].to_numpy() if idc else np.arange(len(pred_df))
    feats = [c for c in train.columns if c not in (target, idc)]
    types = meta.get("feature_types", {})
    num = [c for c in feats if types.get(c, "numeric" if pd.api.types.is_numeric_dtype(train[c]) else "categorical")
           == "numeric" and pd.api.types.is_numeric_dtype(train[c])]
    cat = [c for c in feats if c not in num]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=5)), cat),
    ])
    for df in (train, pred_df):
        for c in cat:
            df[c] = df[c].astype(str)
    if task == "clustering":
        X = pre.fit_transform(pred_df[feats])
        labels = KMeans(meta["n_clusters"], n_init=10, random_state=0).fit_predict(X)
        preds = labels
    else:
        tr = train.dropna(subset=[target])
        model = LogisticRegression(max_iter=1000) if task == "classification" else Ridge(alpha=1.0)
        pipe = make_pipeline(pre, model)
        pipe.fit(tr[feats], tr[target])
        preds = pipe.predict(pred_df[feats])
    pd.DataFrame({"id": ids, "prediction": preds}).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    json.dump({"schema_version": "1.0", "dataset": {"task": task}}, open(os.path.join(out_dir, "profile.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
