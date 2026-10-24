"""End-to-end: main.py produces valid outputs for each task family."""

import numpy as np
import pandas as pd

from conftest import run_main, write_ds


def _cls_frame(rng, n):
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    color = rng.choice(["red", "green", "blue"], n)
    y = np.where(x1 + 0.5 * (color == "red") + rng.normal(0, 0.5, n) > 0.3, "yes", "no")
    return pd.DataFrame({"row": np.arange(n), "x1": x1, "x2": x2, "color": color, "label": y})


def test_classification_string_labels(tmp_path, rng):
    df = _cls_frame(rng, 400)
    tr, te = df.iloc[:300], df.iloc[300:].drop(columns=["label"])
    d = write_ds(tmp_path / "ds", tr, te, task="classification", target="label", id_column="row")
    pred, prof = run_main(d, tmp_path / "out")
    assert list(pred.columns) == ["id", "prediction"]
    assert len(pred) == 100
    assert pred["id"].tolist() == te["row"].tolist()
    assert set(pred["prediction"]) <= {"yes", "no"}
    assert prof["dataset"]["task"] == "classification"
    assert prof["target"]["n_classes"] == 2


def test_integer_labels_written_as_integers(tmp_path, rng):
    df = _cls_frame(rng, 300)
    df["label"] = (df["label"] == "yes").astype(int)
    d = write_ds(tmp_path / "ds", df.iloc[:200], df.iloc[200:].drop(columns=["label"]),
                 task="classification", target="label")
    pred, _ = run_main(d, tmp_path / "out")
    assert set(pred["prediction"].astype(str)) <= {"0", "1"}


def test_regression(tmp_path, rng):
    n = 500
    X = rng.normal(size=(n, 3))
    y = 3 * X[:, 0] - 2 * X[:, 1] ** 2 + rng.normal(0, 0.3, n)
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = y
    d = write_ds(tmp_path / "ds", df.iloc[:400], df.iloc[400:].drop(columns=["target"]),
                 task="regression", target="target")
    pred, prof = run_main(d, tmp_path / "out")
    p = pred["prediction"].astype(float).to_numpy()
    assert np.isfinite(p).all() and len(p) == 100
    resid = y[400:] - p
    assert 1 - resid.var() / y[400:].var() > 0.8


def test_clustering(tmp_path, rng):
    centers = np.array([[0, 0], [6, 6], [0, 8]])
    lab = rng.integers(0, 3, 300)
    X = centers[lab] + rng.normal(0, 1, (300, 2))
    df = pd.DataFrame(X, columns=["u", "v"])
    d = write_ds(tmp_path / "ds", df, task="clustering", n_clusters=3)
    pred, prof = run_main(d, tmp_path / "out")
    from sklearn.metrics import adjusted_rand_score
    assert len(pred) == 300
    assert pred["prediction"].nunique() == 3
    assert adjusted_rand_score(lab, pred["prediction"]) > 0.9
    assert prof["dataset"]["n_clusters"] == 3


def test_missing_metadata_infers_task(tmp_path, rng):
    df = _cls_frame(rng, 200)
    d = write_ds(tmp_path / "ds", df.iloc[:150], df.iloc[150:].drop(columns=["label"]))
    pred, prof = run_main(d, tmp_path / "out")
    assert prof["dataset"]["task"] == "classification"
    assert prof["dataset"]["target"] == "label"
    assert len(pred) == 50


def test_time_limit_respected(tmp_path, rng):
    """A short limit on a mid-sized table still yields valid output in time."""
    import time
    n = 20000
    X = rng.normal(size=(n, 20))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    df["y"] = (X[:, 0] * X[:, 1] + X[:, 2] > 0).astype(int)
    d = write_ds(tmp_path / "ds", df.iloc[:15000], df.iloc[15000:].drop(columns=["y"]),
                 task="classification", target="y")
    t = time.time()
    pred, prof = run_main(d, tmp_path / "out", time_limit=12)
    assert time.time() - t < 12
    assert len(pred) == 5000


def test_module_entry_point_and_api(tmp_path, rng):
    import json
    import os
    import subprocess
    import sys

    import autotab
    from conftest import ROOT
    df = _cls_frame(rng, 120)
    d = write_ds(tmp_path / "ds", df.iloc[:90], df.iloc[90:].drop(columns=["label"]),
                 task="classification", target="label")
    proc = subprocess.run([sys.executable, "-m", "autotab", str(d), str(tmp_path / "out1")], cwd=ROOT,
                          capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stderr
    assert os.path.exists(tmp_path / "out1" / "predictions.csv")
    assert autotab.solve(d, tmp_path / "out2") == 0
    assert json.load(open(tmp_path / "out2" / "profile.json"))["dataset"]["task"] == "classification"
