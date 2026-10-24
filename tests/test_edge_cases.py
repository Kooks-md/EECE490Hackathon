"""Inputs that commonly crash AutoML pipelines."""

import numpy as np
import pandas as pd

from conftest import run_main, write_ds


def test_single_class_training_target(tmp_path, rng):
    df = pd.DataFrame({"x": rng.normal(size=50), "y": "only"})
    d = write_ds(tmp_path / "ds", df.iloc[:40], df.iloc[40:].drop(columns=["y"]), task="classification", target="y")
    pred, _ = run_main(d, tmp_path / "out")
    assert (pred["prediction"] == "only").all()


def test_rare_class_and_nan_targets(tmp_path, rng):
    n = 300
    y = rng.choice(["a", "b"], n).astype(object)
    y[0] = "rare"           # a class with a single example
    y[5:15] = np.nan        # unlabeled rows
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n), "y": y})
    d = write_ds(tmp_path / "ds", df.iloc[:250], df.iloc[250:].drop(columns=["y"]),
                 task="classification", target="y")
    pred, prof = run_main(d, tmp_path / "out")
    assert len(pred) == 50 and set(pred["prediction"]) <= {"a", "b", "rare"}
    assert not prof["modelling"].get("error")


def test_unseen_categories_and_missing_test_columns(tmp_path, rng):
    n = 200
    df = pd.DataFrame({"c": rng.choice(["p", "q"], n), "x": rng.normal(size=n), "w": rng.normal(size=n)})
    df["y"] = df["x"] * 2 + (df["c"] == "p")
    test = pd.DataFrame({"c": ["zzz", "p", None], "x": [0.1, 0.2, 0.3]})  # unseen level, column w absent
    d = write_ds(tmp_path / "ds", df, test, task="regression", target="y")
    pred, _ = run_main(d, tmp_path / "out")
    assert len(pred) == 3 and np.isfinite(pred["prediction"].astype(float)).all()


def test_tiny_dataset(tmp_path, rng):
    df = pd.DataFrame({"x": rng.normal(size=12), "y": rng.normal(size=12)})
    d = write_ds(tmp_path / "ds", df.iloc[:8], df.iloc[8:].drop(columns=["y"]), task="regression", target="y")
    pred, _ = run_main(d, tmp_path / "out")
    assert len(pred) == 4


def test_no_usable_features(tmp_path, rng):
    df = pd.DataFrame({"const": 1, "y": rng.integers(0, 2, 40)})
    d = write_ds(tmp_path / "ds", df.iloc[:30], df.iloc[30:].drop(columns=["y"]), task="classification", target="y")
    pred, _ = run_main(d, tmp_path / "out")
    assert len(pred) == 10


def test_clustering_without_k(tmp_path, rng):
    X = np.vstack([rng.normal(0, 0.5, (60, 2)), rng.normal(5, 0.5, (60, 2))])
    d = write_ds(tmp_path / "ds", pd.DataFrame(X, columns=["a", "b"]), task="clustering")
    pred, prof = run_main(d, tmp_path / "out")
    assert pred["prediction"].nunique() == 2


def test_high_cardinality_and_semicolon_csv(tmp_path, rng):
    n = 400
    df = pd.DataFrame({"city": [f"city_{i % 120}" for i in range(n)], "x": rng.normal(size=n)})
    df["y"] = (df["x"] + (df["city"].str[-1].astype(int) % 2) > 0.5).astype(int)
    d = tmp_path / "ds"
    d.mkdir()
    df.iloc[:300].to_csv(d / "train.csv", index=False, sep=";")
    df.iloc[300:].drop(columns=["y"]).to_csv(d / "test.csv", index=False, sep=";")
    (d / "metadata.json").write_text('{"task": "classification", "target": "y"}')
    pred, prof = run_main(d, tmp_path / "out")
    assert len(pred) == 100
    assert "city" in prof["issues"]["high_cardinality_columns"]


def test_broken_input_still_produces_output(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "train.csv").write_text("a,b,y\n1,2,x\n3,4,y\n")
    (d / "test.csv").write_text("a,b\n5,6\n")
    (d / "metadata.json").write_text('{"task": "classification", "target": "not_a_column", "n_clusters": "x"}')
    pred, _ = run_main(d, tmp_path / "out")
    assert len(pred) == 1


def test_boolean_target_round_trips(tmp_path, rng):
    n = 200
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "flag": x + rng.normal(0, 0.3, n) > 0})
    d = write_ds(tmp_path / "ds", df.iloc[:150], df.iloc[150:].drop(columns=["flag"]),
                 task="classification", target="flag")
    pred, _ = run_main(d, tmp_path / "out")
    assert set(pred["prediction"].astype(str)) <= {"True", "False"}


def test_empty_test_file(tmp_path, rng):
    df = pd.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50)})
    d = write_ds(tmp_path / "ds", df, df.iloc[:0].drop(columns=["y"]), task="regression", target="y")
    pred, _ = run_main(d, tmp_path / "out")
    assert len(pred) == 0 and list(pred.columns) == ["id", "prediction"]


def test_unlabelled_target_falls_back(tmp_path, rng):
    df = pd.DataFrame({"x": rng.normal(size=30), "y": np.nan})
    d = write_ds(tmp_path / "ds", df.iloc[:20], df.iloc[20:].drop(columns=["y"]), task="regression", target="y")
    pred, prof = run_main(d, tmp_path / "out")
    assert len(pred) == 10
    assert prof.get("fallback", {}).get("used")
