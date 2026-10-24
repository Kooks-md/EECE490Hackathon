"""Input contract (dataset directory) and output contract (predictions + profile).

Dataset directory::

    metadata.json   task, target, id_column, feature_types, n_clusters, metric
    train.csv       supervised: features + target; clustering: rows to cluster
    test.csv        supervised: features of the rows to predict

Outputs written to the output directory::

    predictions.csv     id,prediction   (one row per row to predict, input order)
    profile.json        structured data profile + modelling report
    probabilities.csv   classification only: id + one column per class

The loader is deliberately lenient about metadata spelling (``task_type``,
``label``, ``k`` ...) because a crash on a key name costs a whole dataset.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .jsonutil import atomic_write_text, dumps

# pandas already treats "", "NA", "N/A", "NULL", "NaN", "None", "nan", "null",
# "n/a", "#N/A", "<NA>" ... as missing. "?" is the common UCI-style marker.
EXTRA_NA_VALUES = ["?"]

METADATA_FILES = ("metadata.json", "meta.json", "spec.json", "dataset.json", "info.json", "task.json")
TRAIN_FILES = ("train.csv", "training.csv", "data.csv", "dataset.csv")
TEST_FILES = ("test.csv", "testing.csv", "eval.csv", "evaluation.csv", "predict.csv")

TYPE_ALIASES = {
    "numeric": "numeric", "number": "numeric", "numerical": "numeric", "float": "numeric",
    "int": "numeric", "integer": "numeric", "continuous": "numeric", "real": "numeric",
    "double": "numeric", "num": "numeric", "ordinal_numeric": "numeric",
    "categorical": "categorical", "category": "categorical", "nominal": "categorical",
    "cat": "categorical", "string": "categorical", "object": "categorical", "str": "categorical",
    "ordinal": "categorical", "binary": "categorical", "bool": "categorical", "boolean": "categorical",
    "discrete": "categorical", "label": "categorical",
    "datetime": "datetime", "date": "datetime", "time": "datetime", "timestamp": "datetime",
    "datetime64": "datetime",
    "text": "text", "free_text": "text", "freetext": "text",
    "id": "id", "identifier": "id", "key": "id", "index": "id",
}


def normalize_type(value) -> str | None:
    if value is None:
        return None
    key = re.sub(r"[\s\-]+", "_", str(value).strip().lower())
    if key in TYPE_ALIASES:
        return TYPE_ALIASES[key]
    for alias, canon in TYPE_ALIASES.items():
        if key.startswith(alias):
            return canon
    return None


def normalize_task(value) -> str | None:
    if value is None:
        return None
    t = str(value).strip().lower()
    if "regress" in t:
        return "regression"
    if "cluster" in t or t in ("segmentation", "unsupervised", "grouping"):
        return "clustering"
    if "class" in t or t in ("binary", "multiclass", "multi_class", "multi-class"):
        return "classification"
    return None


def _first_key(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _parse_feature_types(raw) -> dict:
    """Accept {col: type}, [{name, type}], or {type: [cols]} layouts."""
    out: dict[str, str] = {}
    if raw is None:
        return out
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, (list, tuple)):  # {"numeric": ["a", "b"]}
                canon = normalize_type(k)
                for col in v:
                    if canon:
                        out[str(col)] = canon
            elif isinstance(v, dict):  # {"a": {"type": "numeric"}}
                canon = normalize_type(_first_key(v, "type", "dtype", "kind"))
                if canon:
                    out[str(k)] = canon
            else:
                canon = normalize_type(v)
                if canon:
                    out[str(k)] = canon
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                name = _first_key(item, "name", "column", "feature")
                canon = normalize_type(_first_key(item, "type", "dtype", "kind"))
                if name is not None and canon:
                    out[str(name)] = canon
    return out


def load_metadata(data_dir: str) -> dict:
    for name in METADATA_FILES:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig") as fh:
                meta = json.load(fh)
            if isinstance(meta, dict):
                meta["_source_file"] = name
                return meta
    return {}


def _find(data_dir: str, candidates) -> str | None:
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    return None


def _sniff_sep(path: str) -> str:
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()
    counts = {sep: header.count(sep) for sep in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _clean_column_names(columns) -> list:
    names, seen = [], {}
    for i, c in enumerate(columns):
        name = str(c).strip() or f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}__{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    return names


def is_stringlike(series: pd.Series) -> bool:
    return not (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
    )


def read_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=_sniff_sep(path),
        na_values=EXTRA_NA_VALUES,
        keep_default_na=True,
        skipinitialspace=True,
        low_memory=False,
        encoding="utf-8-sig",
        encoding_errors="replace",
    )
    df.columns = _clean_column_names(df.columns)
    for col in df.columns:
        s = df[col]
        if is_stringlike(s):
            stripped = s.astype("str").str.strip() if not pd.api.types.is_string_dtype(s) else s.str.strip()
            stripped = stripped.mask(stripped.isin(["", "nan", "NaN", "None", "?"]))
            df[col] = stripped
    return df


@dataclass
class Dataset:
    name: str
    task: str
    target: str | None
    id_column: str | None
    feature_types: dict
    n_clusters: int | None
    metric: str | None
    positive_label: object
    train: pd.DataFrame
    test: pd.DataFrame | None
    pred_frame: pd.DataFrame          # rows we must output predictions for
    pred_ids: np.ndarray
    predict_on: str                    # "test" or "train"
    notes: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def feature_columns(self) -> list:
        drop = {self.target, self.id_column}
        return [c for c in self.train.columns if c not in drop]


def _infer_task(train: pd.DataFrame, target: str | None) -> str:
    if target is None or target not in train.columns:
        return "clustering"
    y = train[target].dropna()
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    values = y.to_numpy(dtype=float)
    if np.all(np.mod(values, 1) == 0) and y.nunique() <= 10:
        return "classification"
    return "regression"


def load_dataset(data_dir: str) -> Dataset:
    meta = load_metadata(data_dir)
    notes: list[str] = []

    train_path = _find(data_dir, TRAIN_FILES)
    test_path = _find(data_dir, TEST_FILES)
    if train_path is None and test_path is None:
        csvs = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".csv"))
        if not csvs:
            raise FileNotFoundError(f"no CSV files found in {data_dir}")
        train_path = os.path.join(data_dir, csvs[0])
        notes.append(f"no train.csv/test.csv found; using {csvs[0]}")

    train = read_table(train_path) if train_path else None
    test = read_table(test_path) if test_path else None
    if train is None:  # only a test file: treat it as the table to cluster
        train, test = test, None

    task = normalize_task(_first_key(meta, "task", "task_type", "problem_type", "type", "objective"))
    target = _first_key(meta, "target", "target_column", "target_col", "label", "label_column", "y")
    if isinstance(target, (list, tuple)):
        target = target[0] if target else None
    target = str(target).strip() if target is not None else None
    id_column = _first_key(meta, "id_column", "id_col", "id", "index_column", "row_id")
    id_column = str(id_column).strip() if id_column is not None else None
    n_clusters = _first_key(meta, "n_clusters", "num_clusters", "k", "n_groups", "num_groups", "clusters")
    metric = _first_key(meta, "metric", "eval_metric", "evaluation_metric", "scoring")
    positive_label = _first_key(meta, "positive_label", "pos_label", "positive_class")
    feature_types = _parse_feature_types(_first_key(meta, "feature_types", "features", "column_types", "columns", "schema"))

    if task in ("classification", "regression") and (target is None or target not in train.columns):
        guess = None
        if test is not None:
            extra = [c for c in train.columns if c not in test.columns]
            if len(extra) == 1:
                guess = extra[0]
        if guess is None and task is not None:
            guess = train.columns[-1]
        if target is not None and guess is not None:
            notes.append(f"target '{target}' not found; using '{guess}'")
        target = guess
    if task is None:
        if target is None and test is not None:
            extra = [c for c in train.columns if c not in test.columns]
            if len(extra) == 1:
                target = extra[0]
        task = _infer_task(train, target)
        notes.append(f"task not given; inferred '{task}'")
    if task == "clustering":
        if target is not None and target in train.columns:
            notes.append(f"clustering task: ignoring column '{target}' as a label")
            train = train.drop(columns=[target])
            if test is not None and target in test.columns:
                test = test.drop(columns=[target])
        target = None
        if n_clusters is not None:
            try:
                n_clusters = int(n_clusters)
            except (TypeError, ValueError):
                n_clusters = None
    else:
        n_clusters = None

    if id_column is not None and id_column not in train.columns and (test is None or id_column not in test.columns):
        notes.append(f"id_column '{id_column}' not present; using row index")
        id_column = None

    # Rows to predict
    if task == "clustering":
        pred_frame, predict_on = (test, "test") if test is not None else (train, "train")
    elif test is not None:
        pred_frame, predict_on = test, "test"
    else:
        pred_frame, predict_on = train, "train"
        notes.append("no test.csv for a supervised task; predicting training rows out-of-fold")

    if target is not None and pred_frame is not None and target in pred_frame.columns and predict_on == "test":
        pred_frame = pred_frame.drop(columns=[target])
        test = pred_frame

    if id_column is not None and id_column in pred_frame.columns:
        pred_ids = pred_frame[id_column].to_numpy()
    else:
        pred_ids = np.arange(len(pred_frame))

    # Align test columns to train feature columns
    if test is not None and task != "clustering":
        feats = [c for c in train.columns if c not in (target,)]
        missing = [c for c in feats if c not in test.columns]
        if missing:
            notes.append(f"test is missing columns {missing[:5]}; filled with NaN")
        test = test.reindex(columns=feats)
        if predict_on == "test":
            pred_frame = test

    name = str(_first_key(meta, "name", "dataset", "dataset_name", default=os.path.basename(os.path.normpath(data_dir))))
    return Dataset(
        name=name, task=task, target=target, id_column=id_column, feature_types=feature_types,
        n_clusters=n_clusters, metric=str(metric).lower() if metric else None,
        positive_label=positive_label, train=train, test=test, pred_frame=pred_frame,
        pred_ids=pred_ids, predict_on=predict_on, notes=notes, metadata=meta,
    )


def _normalize_label_values(values: np.ndarray) -> np.ndarray:
    """Write integer-valued float labels as ints (e.g. 1.0 -> 1)."""
    if values.dtype.kind == "f" and values.size and np.all(np.isfinite(values)) and np.all(np.mod(values, 1) == 0):
        return values.astype(np.int64)
    return values


class OutputWriter:
    """Writes outputs atomically, so a kill at any moment leaves valid files."""

    def __init__(self, out_dir: str, pred_ids: np.ndarray):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.pred_ids = pred_ids
        self.n_writes = 0

    def _csv(self, df: pd.DataFrame, name: str) -> None:
        atomic_write_text(os.path.join(self.out_dir, name), df.to_csv(index=False, lineterminator="\n"))

    def predictions(self, values, label_values: bool = False) -> None:
        values = np.asarray(values)
        if label_values:
            values = _normalize_label_values(values)
        else:
            values = np.where(np.isfinite(values.astype(float)), values.astype(float), 0.0)
        if len(values) != len(self.pred_ids):
            raise ValueError(f"prediction length {len(values)} != rows to predict {len(self.pred_ids)}")
        self._csv(pd.DataFrame({"id": self.pred_ids, "prediction": values}), "predictions.csv")
        self.n_writes += 1

    def probabilities(self, proba: np.ndarray, classes) -> None:
        cols = {"id": self.pred_ids}
        for j, c in enumerate(classes):
            cols[f"proba_{c}"] = np.round(proba[:, j], 6)
        self._csv(pd.DataFrame(cols), "probabilities.csv")

    def profile(self, profile: dict) -> None:
        atomic_write_text(os.path.join(self.out_dir, "profile.json"), dumps(profile, indent=2))
