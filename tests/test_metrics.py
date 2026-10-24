import numpy as np
from sklearn import metrics as skm

from autotab.ensemble import caruana, fit_rule
from autotab.metrics import resolve_metric, score_labels


def test_fast_label_metrics_match_sklearn(rng):
    for _ in range(200):
        k, n = rng.integers(2, 6), rng.integers(5, 200)
        y, p = rng.integers(0, k, n), rng.integers(0, k, n)
        assert np.isclose(score_labels("f1_macro", y, p), skm.f1_score(y, p, average="macro", zero_division=0))
        assert np.isclose(score_labels("f1_weighted", y, p),
                          skm.f1_score(y, p, average="weighted", zero_division=0))
        assert np.isclose(score_labels("mcc", y, p), skm.matthews_corrcoef(y, p))


def test_metric_aliases():
    assert resolve_metric("classification", "Macro F1") == "f1_macro"
    assert resolve_metric("regression", "mean_absolute_error") == "mae"
    assert resolve_metric("clustering", None) == "ari"
    assert resolve_metric("regression", "nonsense") == "rmse"


def test_threshold_tuning_helps_imbalanced(rng):
    n = 2000
    y = (rng.random(n) < 0.1).astype(int)
    p1 = np.clip(0.25 * y + rng.normal(0.1, 0.08, n), 0, 1)
    P = np.column_stack([1 - p1, p1])
    w, s = fit_rule(P, y, "f1_macro", 2, pos=1)
    assert s > score_labels("f1_macro", y, np.argmax(P, axis=1))


def test_caruana_prefers_better_model(rng):
    y = rng.normal(size=500)
    preds = {"good": y + rng.normal(0, 0.1, 500), "bad": rng.normal(size=500)}
    w, _ = caruana(preds, lambda p: -np.mean((p - y) ** 2))
    assert w.get("good", 0) > 0.8
