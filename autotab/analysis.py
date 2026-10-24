"""Column analysis: type inference, ID / constant / duplicate detection.

Every definition used in the profile lives here so the profile and the
modelling pipeline can never disagree. Definitions (also in the README):

* missing        NaN after parsing. Empty / whitespace-only strings and the
                 usual NA tokens ("NA", "null", "?", ...) count as missing.
* constant       at most one distinct non-missing value (all-missing columns
                 are constant).
* duplicate      identical to an earlier column (as read, or after parsing),
                 with missing values in the same rows. The first occurrence is kept.
* id-like        no missing values, a distinct value on every row, and either
                 non-numeric, or integer-valued and monotonic / named like an ID.
* high-card      categorical with more than 50 distinct values.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dataio import Dataset

HIGH_CARDINALITY = 50
ID_NAME_PATTERN = re.compile(r"(^|[^a-z])(id|ids|idx|uuid|guid|key|index|udi|rowid|row_id|unnamed)([^a-z]|$)")
_NUMERIC_JUNK = re.compile(r"[,$€£%\s]")
_NUMBERISH = re.compile(r"^[\s$€£]*[-+]?[\d,]*\.?\d+(e[-+]?\d+)?\s*%?$", re.IGNORECASE)


@dataclass
class ColumnInfo:
    name: str
    declared_type: str | None = None
    inferred_type: str = "numeric"      # numeric | categorical | datetime | text | boolean
    model_type: str = "numeric"         # numeric | categorical | datetime | text
    role: str = "feature"               # feature | id | target
    n_missing: int = 0
    missing_fraction: float = 0.0
    n_unique: int = 0
    is_constant: bool = False
    is_id_like: bool = False
    duplicate_of: str | None = None
    high_cardinality: bool = False
    integer_valued: bool = False
    dropped: bool = False
    drop_reason: str | None = None
    notes: list = field(default_factory=list)


@dataclass
class Analysis:
    columns: list                       # ColumnInfo for every non-target column
    train: pd.DataFrame                 # parsed feature frame (training rows)
    pred: pd.DataFrame | None           # parsed feature frame (rows to predict), None if same as train
    kept: list                          # column names used for modelling
    notes: list = field(default_factory=list)

    def info(self, name: str) -> ColumnInfo:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)


def _coerce_numeric(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    nn = s.notna().sum()
    if nn and num.notna().sum() / nn < 0.95:
        sample = s.dropna().astype(str).head(500)
        if len(sample) and sample.str.match(_NUMBERISH).mean() >= 0.9:
            cleaned = s.astype("str").str.replace(_NUMERIC_JUNK, "", regex=True)
            cleaned = cleaned.mask(s.isna())
            num2 = pd.to_numeric(cleaned, errors="coerce")
            if num2.notna().sum() > num.notna().sum():
                num = num2
    return num.astype(float)


def _try_datetime(s: pd.Series, force: bool = False) -> pd.Series | None:
    values = s.dropna().astype(str)
    if values.empty:
        return None
    sample = values.drop_duplicates().head(300)
    if not force:
        lens = sample.str.len()
        looks = sample.str.contains(r"\d") & sample.str.contains(r"[-/:T]") & lens.between(6, 40)
        if looks.mean() < 0.9:
            return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() < 0.9:
            return None
        full = pd.to_datetime(s.astype("str").mask(s.isna()), errors="coerce", format="mixed")
    nn = s.notna().sum()
    if nn and full.notna().sum() / nn < 0.9:
        return None
    try:
        full = full.dt.tz_localize(None) if getattr(full.dt, "tz", None) is not None else full
    except (TypeError, AttributeError):
        pass
    return full


def _looks_like_text(s: pd.Series) -> bool:
    values = s.dropna().astype(str)
    if len(values) < 20:
        return False
    sample = values.head(1000)
    unique_ratio = values.nunique() / len(values)
    return bool(unique_ratio > 0.5 and sample.str.len().mean() > 25 and sample.str.contains(" ").mean() > 0.5)


def _as_category_strings(s: pd.Series) -> pd.Series:
    """Numeric column declared categorical -> clean string labels (1.0 -> '1')."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        vals = s.to_numpy(dtype=float)
        out = np.empty(len(vals), dtype=object)
        for i, v in enumerate(vals):
            if not np.isfinite(v):
                out[i] = np.nan
            elif float(v).is_integer():
                out[i] = str(int(v))
            else:
                out[i] = repr(float(v))
        return pd.Series(out, index=s.index, dtype="object").astype("str").mask(pd.isna(s))
    return s.astype("str").mask(s.isna())


def infer_column(s: pd.Series, declared: str | None) -> tuple[str, pd.Series]:
    """Return (inferred_type, parsed_series) for one column (train+test values).

    Infinite numeric values are treated as missing.
    """
    kind, parsed = _infer_column(s, declared)
    if kind in ("numeric", "boolean") and pd.api.types.is_float_dtype(parsed):
        parsed = parsed.replace([np.inf, -np.inf], np.nan)
    return kind, parsed


def _infer_column(s: pd.Series, declared: str | None) -> tuple[str, pd.Series]:
    if pd.api.types.is_bool_dtype(s):
        return "boolean", s.astype(float)
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime", s
    if pd.api.types.is_numeric_dtype(s):
        if declared in ("categorical", "text"):
            return "categorical", _as_category_strings(s)
        return "numeric", s.astype(float)
    # string-like column
    if s.notna().sum() == 0:
        return (declared if declared in ("numeric", "categorical", "datetime", "text") else "numeric"), pd.Series(
            np.nan, index=s.index, dtype=float)
    if declared not in ("categorical", "text", "datetime"):
        num = _coerce_numeric(s)
        frac = num.notna().sum() / max(1, s.notna().sum())
        if frac >= 0.95 or (declared == "numeric" and frac >= 0.5):
            return "numeric", num
    mostly_unique = s.nunique(dropna=True) > 0.5 * max(1, s.notna().sum())
    if declared in (None, "datetime") or (declared == "categorical" and mostly_unique):
        # A "categorical" whose values are mostly distinct date strings is really a datetime.
        dt = _try_datetime(s, force=(declared == "datetime"))
        if dt is not None:
            return "datetime", dt
    lowered = s.str.lower()
    uniq = set(lowered.dropna().unique()[:5])
    if declared is None and uniq and uniq <= {"true", "false"}:
        return "boolean", lowered.map({"true": 1.0, "false": 0.0}).astype(float)
    if declared == "text" or (declared is None and _looks_like_text(s)):
        return "text", s
    return "categorical", s


def _is_integer_valued(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return bool(finite.size) and bool(np.all(np.mod(finite, 1) == 0))


def _id_like(s: pd.Series, name: str, inferred: str) -> bool:
    n = len(s)
    if n < 20 or s.isna().any():
        return False
    if s.nunique(dropna=True) != n:
        return False
    if inferred in ("categorical", "text"):
        return True
    if inferred == "numeric":
        vals = s.to_numpy(dtype=float)
        if not _is_integer_valued(vals):
            return False
        diffs = np.diff(vals)
        monotonic = bool(np.all(diffs > 0) or np.all(diffs < 0))
        lname = name.lower().strip()
        named = bool(ID_NAME_PATTERN.search(lname)) or lname.endswith("id")
        return monotonic or named
    return False


def _column_hash(s: pd.Series) -> str:
    h = pd.util.hash_pandas_object(s, index=False).to_numpy()
    return f"{s.dtype}:{hashlib.sha1(h.tobytes()).hexdigest()}"


def analyze(ds: Dataset) -> Analysis:
    """Parse every feature column and flag ID / constant / duplicate columns."""
    feats = ds.feature_columns
    n_train = len(ds.train)
    has_pred = ds.predict_on == "test" and ds.pred_frame is not None
    if has_pred:
        pred_df = ds.pred_frame.reindex(columns=feats)
        combined = pd.concat([ds.train[feats], pred_df], axis=0, ignore_index=True)
    else:
        combined = ds.train[feats].reset_index(drop=True)

    infos: list[ColumnInfo] = []
    parsed: dict[str, pd.Series] = {}
    notes: list[str] = []

    if ds.id_column is not None and ds.id_column in ds.train.columns:
        idc = ds.train[ds.id_column]
        infos.append(ColumnInfo(
            name=ds.id_column, declared_type=ds.feature_types.get(ds.id_column, "id"), inferred_type="id",
            model_type="id", role="id", n_missing=int(idc.isna().sum()),
            missing_fraction=float(idc.isna().mean()) if n_train else 0.0, n_unique=int(idc.nunique()),
            is_id_like=True, dropped=True, drop_reason="id_column",
        ))

    for col in feats:
        declared = ds.feature_types.get(col)
        info = ColumnInfo(name=col, declared_type=declared)
        if declared == "id":
            info.role, info.inferred_type, info.model_type = "id", "id", "id"
            info.is_id_like, info.dropped, info.drop_reason = True, True, "declared_id"
            s_tr = combined[col].iloc[:n_train]
            info.n_missing = int(s_tr.isna().sum())
            info.missing_fraction = float(s_tr.isna().mean()) if n_train else 0.0
            info.n_unique = int(s_tr.nunique())
            infos.append(info)
            continue
        try:
            inferred, series = infer_column(combined[col], declared)
        except Exception as exc:  # noqa: BLE001 - never let one column kill the run
            inferred, series = "categorical", combined[col].astype("str").mask(combined[col].isna())
            info.notes.append(f"type inference failed ({type(exc).__name__}); treated as categorical")
        info.inferred_type = inferred
        if declared in ("numeric", "categorical", "datetime", "text") and declared != inferred:
            if not (declared == "categorical" and inferred == "boolean"):
                info.notes.append(f"declared {declared}, parsed as {inferred}")
        info.model_type = "numeric" if inferred == "boolean" else inferred
        parsed[col] = series

        s_tr = series.iloc[:n_train]
        info.n_missing = int(s_tr.isna().sum())
        info.missing_fraction = float(info.n_missing / n_train) if n_train else 0.0
        info.n_unique = int(s_tr.nunique(dropna=True))
        info.is_constant = info.n_unique <= 1
        if inferred in ("numeric", "boolean"):
            info.integer_valued = _is_integer_valued(s_tr.to_numpy(dtype=float))
        if inferred in ("categorical", "text") and info.n_unique > HIGH_CARDINALITY:
            info.high_cardinality = True
        info.is_id_like = _id_like(s_tr, col, inferred)
        infos.append(info)

    # Duplicate columns, first occurrence kept: identical either as read from the file or
    # after parsing (so a copy is caught even if the two copies were declared different types).
    seen_raw: dict[str, str] = {}
    seen_parsed: dict[str, str] = {}
    for info in infos:
        if info.role != "feature":
            continue
        raw = ds.train[info.name].reset_index(drop=True)
        s_tr = parsed[info.name].iloc[:n_train].reset_index(drop=True)
        k_raw, k_parsed = _column_hash(raw), _column_hash(s_tr)
        original = None
        if k_raw in seen_raw and raw.equals(ds.train[seen_raw[k_raw]].reset_index(drop=True)):
            original = seen_raw[k_raw]
        elif k_parsed in seen_parsed and s_tr.equals(parsed[seen_parsed[k_parsed]].iloc[:n_train].reset_index(drop=True)):
            original = seen_parsed[k_parsed]
        if original is not None:
            info.duplicate_of = original
            continue
        seen_raw.setdefault(k_raw, info.name)
        seen_parsed.setdefault(k_parsed, info.name)

    # Modelling decisions
    for info in infos:
        if info.role != "feature":
            continue
        s_all = parsed[info.name]
        if info.is_id_like:
            info.dropped, info.drop_reason = True, "id_like"
        elif s_all.nunique(dropna=False) <= 1:
            info.dropped, info.drop_reason = True, "constant"
        elif info.is_constant and s_all.iloc[:n_train].isna().sum() == 0:
            info.dropped, info.drop_reason = True, "constant"
        elif info.duplicate_of is not None:
            info.dropped, info.drop_reason = True, f"duplicate_of:{info.duplicate_of}"

    kept = [i.name for i in infos if i.role == "feature" and not i.dropped]
    parsed_df = pd.DataFrame({c: parsed[c] for c in parsed}) if parsed else pd.DataFrame(index=combined.index)
    train_p = parsed_df.iloc[:n_train].reset_index(drop=True)
    pred_p = parsed_df.iloc[n_train:].reset_index(drop=True) if has_pred else None
    if not kept:
        notes.append("no usable feature columns after cleaning")
    return Analysis(columns=infos, train=train_p, pred=pred_p, kept=kept, notes=notes)
