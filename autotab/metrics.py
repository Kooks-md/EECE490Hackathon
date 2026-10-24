"""Metric registry. Every scorer returns "higher is better".

Classification scorers take integer-coded labels; probability metrics take
an (n, K) probability matrix. Regression scorers take raw-scale predictions.
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

CLASSIFICATION_METRICS = {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "f1", "roc_auc", "log_loss", "mcc"}
REGRESSION_METRICS = {"rmse", "mse", "mae", "r2", "rmsle", "mape"}
CLUSTERING_METRICS = {"ari", "nmi", "ami", "silhouette"}

DEFAULTS = {"classification": "f1_macro", "regression": "rmse", "clustering": "ari"}

ALIASES = {
    "acc": "accuracy", "balanced_acc": "balanced_accuracy", "bacc": "balanced_accuracy",
    "macro_f1": "f1_macro", "f1macro": "f1_macro", "f1-macro": "f1_macro", "f1_score": "f1_macro",
    "weighted_f1": "f1_weighted", "f1_binary": "f1", "auc": "roc_auc", "rocauc": "roc_auc",
    "logloss": "log_loss", "cross_entropy": "log_loss", "matthews": "mcc",
    "root_mean_squared_error": "rmse", "mean_squared_error": "mse", "mean_absolute_error": "mae",
    "r_squared": "r2", "r^2": "r2", "neg_rmse": "rmse", "neg_mae": "mae",
    "adjusted_rand": "ari", "adjusted_rand_index": "ari", "adjusted_rand_score": "ari",
    "normalized_mutual_info": "nmi", "nmi_score": "nmi", "adjusted_mutual_info": "ami",
}


def resolve_metric(task: str, name: str | None) -> str:
    if name:
        key = str(name).strip().lower().replace(" ", "_")
        key = ALIASES.get(key, key)
        allowed = {"classification": CLASSIFICATION_METRICS, "regression": REGRESSION_METRICS,
                   "clustering": CLUSTERING_METRICS}[task]
        if key in allowed:
            return key
    return DEFAULTS[task]


def needs_proba(metric: str) -> bool:
    return metric in ("roc_auc", "log_loss")


def score_labels(metric: str, y: np.ndarray, pred: np.ndarray, pos: int = 1) -> float:
    """Label metrics from a confusion matrix (integer-coded labels).

    Equivalent to sklearn's f1/balanced-accuracy/MCC with zero_division=0, but
    ~50x faster, which matters because threshold tuning calls this hundreds of
    times per refit.
    """
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    if metric == "accuracy":
        return float(np.mean(y == pred))
    k = int(max(y.max(initial=0), pred.max(initial=0))) + 1
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)          # true counts per class
    predicted = cm.sum(axis=0)        # predicted counts per class
    if metric == "balanced_accuracy":
        present = support > 0
        return float(np.mean(tp[present] / support[present])) if present.any() else 0.0
    if metric in ("f1_macro", "f1_weighted", "f1"):
        denom = support + predicted
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
        if metric == "f1":
            return float(f1[pos]) if pos < k else 0.0
        present = (support > 0) | (predicted > 0)  # sklearn averages over labels present in y or pred
        if metric == "f1_macro":
            return float(f1[present].mean()) if present.any() else 0.0
        return float((f1 * support).sum() / support.sum()) if support.sum() else 0.0
    if metric == "mcc":
        n = cm.sum()
        cov_ytyp = (tp.sum() * n) - (support * predicted).sum()
        cov_ypyp = n * n - (predicted ** 2).sum()
        cov_ytyt = n * n - (support ** 2).sum()
        denom = np.sqrt(cov_ytyt * cov_ypyp)
        return float(cov_ytyp / denom) if denom > 0 else 0.0
    raise ValueError(metric)


def score_proba(metric: str, y: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    proba = np.clip(proba, 1e-12, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    if metric == "log_loss":
        return -float(skm.log_loss(y, proba, labels=np.arange(n_classes)))
    if metric == "roc_auc":
        if n_classes == 2:
            return float(skm.roc_auc_score(y, proba[:, 1])) if len(np.unique(y)) == 2 else 0.5
        present = np.unique(y)
        if len(present) < 2:
            return 0.5
        try:
            return float(skm.roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=np.arange(n_classes)))
        except ValueError:
            return 0.5
    raise ValueError(metric)


def score_regression(metric: str, y: np.ndarray, pred: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    if metric in ("rmse", "mse"):
        mse = float(np.mean((y - pred) ** 2))
        return -np.sqrt(mse) if metric == "rmse" else -mse
    if metric == "mae":
        return -float(np.mean(np.abs(y - pred)))
    if metric == "r2":
        return float(skm.r2_score(y, pred))
    if metric == "rmsle":
        return -float(np.sqrt(np.mean((np.log1p(np.maximum(pred, 0)) - np.log1p(np.maximum(y, 0))) ** 2)))
    if metric == "mape":
        denom = np.maximum(np.abs(y), 1e-9)
        return -float(np.mean(np.abs(y - pred) / denom))
    raise ValueError(metric)


def score_clustering(metric: str, labels_true: np.ndarray, labels_pred: np.ndarray, X=None) -> float:
    if metric == "ari":
        return float(skm.adjusted_rand_score(labels_true, labels_pred))
    if metric == "nmi":
        return float(skm.normalized_mutual_info_score(labels_true, labels_pred))
    if metric == "ami":
        return float(skm.adjusted_mutual_info_score(labels_true, labels_pred))
    if metric == "silhouette":
        return float(skm.silhouette_score(X, labels_pred)) if X is not None else 0.0
    raise ValueError(metric)
