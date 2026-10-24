"""Profiling definitions (see autotab/analysis.py and the README)."""

import numpy as np
import pandas as pd

from autotab.analysis import analyze
from autotab.dataio import load_dataset
from autotab.profile import build_profile
from conftest import write_ds


def _profile(tmp_path, df, **meta):
    d = write_ds(tmp_path / "ds", df, **meta)
    ds = load_dataset(str(d))
    an = analyze(ds)
    return build_profile(ds, an), an


def test_constant_duplicate_id_missing(tmp_path, rng):
    n = 120
    df = pd.DataFrame({
        "cust_id": [f"C{i:04d}" for i in range(n)],          # unique strings -> id-like
        "seq": np.arange(n),                                   # monotonic unique ints -> id-like
        "x": rng.normal(size=n),
        "x_copy": None,
        "const": 5,
        "const_with_nan": [1.0 if i % 3 else np.nan for i in range(n)],
        "all_nan": np.nan,
        "cat": rng.choice(["a", "b"], n),
        "y": rng.integers(0, 2, n),
    })
    df["x_copy"] = df["x"]
    prof, an = _profile(tmp_path, df, task="classification", target="y")
    issues = prof["issues"]
    assert set(issues["constant_columns"]) == {"const", "const_with_nan", "all_nan"}
    assert {d["column"] for d in issues["duplicate_columns"]} == {"x_copy"}
    assert {"cust_id", "seq"} <= set(issues["id_like_columns"])
    cols = {c["name"]: c for c in prof["columns"]}
    assert cols["const_with_nan"]["n_missing"] == n // 3
    assert cols["all_nan"]["n_missing"] == n
    assert "x" in an.kept and "x_copy" not in an.kept and "const" not in an.kept
    # one value + NaN is reported constant but kept: missingness itself may be informative
    assert "const_with_nan" in an.kept


def test_numeric_strings_with_blanks_are_numeric(tmp_path):
    vals = ["12.5", "13", " ", "7.25", "?", "100"] * 10
    df = pd.DataFrame({"charges": vals, "y": list(range(60))})
    prof, an = _profile(tmp_path, df, task="regression", target="y")
    col = {c["name"]: c for c in prof["columns"]}["charges"]
    assert col["inferred_type"] == "numeric"
    assert col["n_missing"] == 20  # blank and '?' are missing


def test_datetime_detected_and_declared_types_respected(tmp_path, rng):
    n = 50
    df = pd.DataFrame({
        "when": pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M:%S"),
        "code": rng.integers(1, 4, n),
        "y": rng.normal(size=n),
    })
    prof, an = _profile(tmp_path, df, task="regression", target="y", feature_types={"code": "categorical"})
    cols = {c["name"]: c for c in prof["columns"]}
    assert cols["when"]["inferred_type"] == "datetime"
    assert cols["code"]["inferred_type"] == "categorical"


def test_class_imbalance_reported(tmp_path, rng):
    y = np.array([0] * 90 + [1] * 10)
    df = pd.DataFrame({"x": rng.normal(size=100), "y": y})
    prof, _ = _profile(tmp_path, df, task="classification", target="y")
    ci = prof["issues"]["class_imbalance"]
    assert ci["imbalance_ratio"] == 9.0 and ci["is_imbalanced"] and ci["level"] == "moderate"
    assert prof["target"]["class_counts"] == {"0": 90, "1": 10}


def test_profile_is_strict_json(tmp_path, rng):
    import json
    df = pd.DataFrame({"x": [np.inf, 1.0, 2.0, np.nan] * 10, "y": rng.normal(size=40)})
    prof, _ = _profile(tmp_path, df, task="regression", target="y")
    from autotab.jsonutil import dumps
    json.loads(dumps(prof))  # no NaN / numpy types leak through


def test_duplicate_detected_across_declared_types(tmp_path, rng):
    n = 60
    ts = pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame({"ts": ts, "ts_copy": ts, "y": rng.normal(size=n)})
    prof, an = _profile(tmp_path, df, task="regression", target="y",
                        feature_types={"ts": "datetime", "ts_copy": "categorical"})
    assert {d["column"] for d in prof["issues"]["duplicate_columns"]} == {"ts_copy"}
    assert "ts" in an.kept and "ts_copy" not in an.kept
