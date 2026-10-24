"""Model-ready matrices built once for all rows (training + rows to predict).

Only label-free transforms (imputation values, scaling, category vocabularies)
are fitted here, on all rows at once; that is transductive but leaks nothing
about the labels. Target encoding is the one label-dependent transform and is
cross-fitted per fold inside the supervised engine.

Two views are produced:

* tree view    raw numerics (NaN kept), datetime parts, categorical codes;
               low-cardinality categoricals are flagged for native handling
               by HistGradientBoosting.
* linear view  NaN-free and scaled: log1p for skewed non-negative numerics,
               median imputation + missing indicators, one-hot for
               categoricals and small-integer codes. Used by linear models,
               KNN and SVMs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from .analysis import Analysis

NATIVE_CAT_MAX_LEVELS = 64        # above this, categoricals are frequency/target encoded
ONEHOT_INT_MAX_LEVELS = 12        # small-integer numerics also get one-hot in the linear view
ONEHOT_DATEPART_MAX_LEVELS = 31   # hour / month / weekday parts
DENSE_CELL_LIMIT = 30_000_000     # switch the linear view to sparse above this


@dataclass
class FeatureSpace:
    n_train: int
    tree: np.ndarray
    tree_cat: np.ndarray
    tree_names: list
    tree_imputed: np.ndarray
    linear: object                 # ndarray or scipy.sparse.csr_matrix
    linear_names: list
    highcard: np.ndarray | None    # int codes (-1 = missing) for target encoding
    highcard_names: list
    steps: list = field(default_factory=list)

    @property
    def n_all(self) -> int:
        return self.tree.shape[0]

    def view(self, kind: str):
        return {"tree": self.tree, "tree_imputed": self.tree_imputed, "linear": self.linear}[kind]


def _expand_datetime(s: pd.Series, name: str) -> pd.DataFrame:
    s = pd.to_datetime(s, errors="coerce")
    parts = {
        f"{name}__year": s.dt.year,
        f"{name}__month": s.dt.month,
        f"{name}__day": s.dt.day,
        f"{name}__dayofweek": s.dt.dayofweek,
        f"{name}__dayofyear": s.dt.dayofyear,
    }
    hours = s.dt.hour
    if (hours.fillna(0) != 0).any():
        parts[f"{name}__hour"] = hours
        minutes = s.dt.minute
        if (minutes.fillna(0) != 0).any():
            parts[f"{name}__minute"] = minutes
    parts[f"{name}__is_weekend"] = (s.dt.dayofweek >= 5).astype(float).where(s.notna())
    ref = s.min()
    parts[f"{name}__elapsed_days"] = (s - ref).dt.total_seconds() / 86400.0
    df = pd.DataFrame(parts).astype(float)
    keep = [c for c in df.columns if df[c].nunique(dropna=True) > 1]
    return df[keep]


def _rare_threshold(n_rows: int) -> int:
    return int(max(2, min(10, n_rows // 200)))


def _encode_categorical(s: pd.Series, min_count: int):
    """Frequency-ranked integer codes (0 = most frequent); rare levels merged; NaN kept."""
    s = s.astype("object").where(s.notna(), None)
    counts = pd.Series(s).value_counts(dropna=True)
    rare = counts[counts < min_count].index
    if len(rare) >= 2:
        s = s.where(~pd.Series(s).isin(rare), "__rare__")
        counts = pd.Series(s).value_counts(dropna=True)
    vocab = {lvl: i for i, lvl in enumerate(counts.index)}
    codes = pd.Series(s).map(vocab).to_numpy(dtype=float)  # NaN for missing
    freq = pd.Series(s).map(counts).to_numpy(dtype=float)
    return codes, freq, len(vocab)


def _onehot(codes: np.ndarray, n_levels: int, name: str, include_nan: bool):
    n = len(codes)
    valid = np.isfinite(codes)
    rows = np.nonzero(valid)[0]
    cols = codes[valid].astype(np.int64)
    width = n_levels + (1 if include_nan else 0)
    if include_nan:
        nan_rows = np.nonzero(~valid)[0]
        rows = np.concatenate([rows, nan_rows])
        cols = np.concatenate([cols, np.full(len(nan_rows), n_levels, dtype=np.int64)])
    mat = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, width))
    names = [f"{name}=={j}" for j in range(n_levels)] + ([f"{name}==NaN"] if include_nan else [])
    return mat, names


def build_features(an: Analysis) -> FeatureSpace:
    n_train = len(an.train)
    frames = [an.train[an.kept]]
    if an.pred is not None:
        frames.append(an.pred[an.kept])
    data = pd.concat(frames, axis=0, ignore_index=True) if an.kept else pd.DataFrame(index=range(
        n_train + (len(an.pred) if an.pred is not None else 0)))
    n_all = len(data)
    min_count = _rare_threshold(n_all)
    steps: list[str] = []

    numeric_cols, cat_cols, dt_cols, text_cols = [], [], [], []
    for col in an.kept:
        mt = an.info(col).model_type
        if mt == "numeric":
            numeric_cols.append(col)
        elif mt == "datetime":
            dt_cols.append(col)
        elif mt == "text":
            text_cols.append(col)
        else:
            cat_cols.append(col)

    # ---- numeric block (raw, NaN kept) -----------------------------------------
    num = data[numeric_cols].astype(float) if numeric_cols else pd.DataFrame(index=data.index)
    datepart_cols: list[str] = []
    for col in dt_cols:
        parts = _expand_datetime(data[col], col)
        datepart_cols.extend([c for c in parts.columns if not c.endswith("elapsed_days")])
        num = pd.concat([num, parts], axis=1)
    if dt_cols:
        steps.append(f"expanded datetime columns into calendar parts: {dt_cols}")
    for col in text_cols:
        t = data[col].astype("str").mask(data[col].isna())
        num[f"{col}__n_chars"] = t.str.len().astype(float)
        num[f"{col}__n_words"] = t.str.split().str.len().astype(float)
    if text_cols:
        steps.append(f"text columns reduced to length/word-count features: {text_cols}")

    # ---- categorical block --------------------------------------------------------
    tree_parts, tree_names, tree_cat = [], [], []
    tree_parts.append(num.to_numpy(dtype=float))
    tree_names.extend(num.columns.tolist())
    tree_cat.extend([False] * num.shape[1])

    lin_dense_cols, lin_dense_names = [], []
    lin_sparse_parts, lin_sparse_names = [], []
    highcard_codes, highcard_names = [], []
    native, onehot_cats, freq_cats = [], [], []

    for col in cat_cols + [c for c in text_cols if data[c].nunique() < 0.5 * max(1, data[c].notna().sum())]:
        codes, freq, n_levels = _encode_categorical(data[col], min_count)
        if n_levels == 0:
            continue
        if n_levels <= NATIVE_CAT_MAX_LEVELS:
            tree_parts.append(codes[:, None])
            tree_names.append(col)
            tree_cat.append(True)
            mat, names = _onehot(codes, n_levels, col, include_nan=bool(np.isnan(codes).any()))
            lin_sparse_parts.append(mat)
            lin_sparse_names.extend(names)
            native.append(col)
            onehot_cats.append(col)
        else:
            logf = np.log1p(np.nan_to_num(freq, nan=0.0))
            tree_parts.append(np.column_stack([logf, codes]))
            tree_names.extend([f"{col}__logfreq", f"{col}__code"])
            tree_cat.extend([False, False])
            lin_dense_cols.append(logf)
            lin_dense_names.append(f"{col}__logfreq")
            highcard_codes.append(np.where(np.isfinite(codes), codes, -1).astype(np.int64))
            highcard_names.append(col)
            freq_cats.append(col)
    if native:
        steps.append(f"categoricals with <= {NATIVE_CAT_MAX_LEVELS} levels: native splits in gradient boosting, "
                     f"one-hot in the linear view ({len(native)} columns); levels seen < {min_count} times merged")
    if freq_cats:
        steps.append(f"high-cardinality categoricals frequency-encoded + cross-fitted target encoding: {freq_cats}")

    tree = np.column_stack(tree_parts) if tree_parts and sum(p.shape[1] for p in tree_parts) else np.zeros((n_all, 0))
    tree_cat_arr = np.array(tree_cat, dtype=bool)

    # NaN-free tree view for estimators without missing-value support
    tree_imp = tree.copy()
    if tree_imp.size:
        train_part = tree_imp[:n_train]
        with np.errstate(all="ignore"):
            med = np.nanmedian(train_part, axis=0) if n_train else np.zeros(tree_imp.shape[1])
        med = np.where(np.isfinite(med), med, 0.0)
        for j in range(tree_imp.shape[1]):
            bad = ~np.isfinite(tree_imp[:, j])
            if bad.any():
                tree_imp[bad, j] = -1.0 if tree_cat_arr[j] else med[j]

    # ---- linear view --------------------------------------------------------------
    logged, indicators = [], []
    for col in num.columns:
        x = num[col].to_numpy(dtype=float)
        xtr = x[:n_train]
        finite_tr = xtr[np.isfinite(xtr)]
        if finite_tr.size < 2:
            continue
        if finite_tr.min() >= 0 and pd.Series(finite_tr).skew() > 1.0 and np.unique(finite_tr).size > 2:
            x = np.log1p(np.where(np.isfinite(x), np.maximum(x, 0), np.nan))
            logged.append(col)
        miss = ~np.isfinite(x)
        med = np.nanmedian(x[:n_train]) if np.isfinite(x[:n_train]).any() else 0.0
        x = np.where(miss, med, x)
        sd = x.std()
        if sd <= 1e-12:
            continue
        lin_dense_cols.append(np.clip((x - x.mean()) / sd, -5.0, 5.0))
        lin_dense_names.append(col)
        if miss.any():
            lin_dense_cols.append(miss.astype(float))
            lin_dense_names.append(f"{col}__missing")
            indicators.append(col)
        # small-integer codes (season, weather, hour...) also one-hot for linear models
        raw = num[col].to_numpy(dtype=float)
        finite = raw[np.isfinite(raw)]
        n_lv = np.unique(finite).size
        limit = ONEHOT_DATEPART_MAX_LEVELS if col in datepart_cols else ONEHOT_INT_MAX_LEVELS
        if 3 <= n_lv <= limit and np.all(np.mod(finite, 1) == 0):
            levels = np.unique(finite)
            codes = np.full(len(raw), np.nan)
            ok = np.isfinite(raw)
            codes[ok] = np.searchsorted(levels, raw[ok])
            mat, names = _onehot(codes, len(levels), col, include_nan=False)
            lin_sparse_parts.append(mat)
            lin_sparse_names.extend(names)
    if logged:
        steps.append(f"log1p applied to skewed non-negative numerics (linear view): {logged}")
    if indicators:
        steps.append(f"median imputation + missing-value indicators (linear view): {indicators}")
    steps.append("numeric features standardised and clipped to +-5 sd (linear view)")

    dense = np.column_stack(lin_dense_cols) if lin_dense_cols else np.zeros((n_all, 0))
    names = list(lin_dense_names)
    if lin_sparse_parts:
        sp = sparse.hstack(lin_sparse_parts, format="csr")
        names += lin_sparse_names
        if n_all * (dense.shape[1] + sp.shape[1]) > DENSE_CELL_LIMIT:
            linear = sparse.hstack([sparse.csr_matrix(dense), sp], format="csr")
        else:
            linear = np.hstack([dense, sp.toarray()])
    else:
        linear = dense
    if (linear.shape[1] if hasattr(linear, "shape") else 0) == 0:
        linear = np.zeros((n_all, 1))
        names = ["__bias__"]

    return FeatureSpace(
        n_train=n_train, tree=tree, tree_cat=tree_cat_arr, tree_names=tree_names, tree_imputed=tree_imp,
        linear=linear, linear_names=names,
        highcard=np.column_stack(highcard_codes) if highcard_codes else None,
        highcard_names=highcard_names, steps=steps,
    )
