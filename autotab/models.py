"""Candidate model portfolio for classification and regression.

The portfolio is fixed and small on purpose: a few configurations that were
chosen offline on the benchmark (bench/), rather than a live hyperparameter
search that would mostly fit noise inside a 60-second budget. Diversity comes
from model families (boosted trees, randomised trees, linear, KNN, SVM) and
target transforms; the ensemble selector decides how to combine them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


@dataclass
class FitContext:
    task: str                   # classification | regression
    n_classes: int
    n_rows: int                 # rows in the training fold
    n_threads: int
    seed: int
    cat_mask: np.ndarray        # native-categorical mask for the tree view (incl. appended columns)
    min_class_count: int = 0    # smallest class count in the training fold
    iter_budget: int | None = None  # boosting rounds affordable within the time budget


@dataclass
class Candidate:
    name: str
    view: str                                  # tree | tree_imputed | linear
    build: Callable[[FitContext], object]
    target_transform: str | None = None        # "log1p" for regression
    follow_transform: bool = False             # adopt log1p when it won for gradient boosting
    max_rows: int = 10 ** 9
    max_cols: int = 10 ** 9
    cost: float = 1.0                          # prior cost relative to the first HGB fold
    tags: list = field(default_factory=list)


def _iteration_cap(n: int) -> int:
    """Early stopping on validation loss rarely triggers on small, clean data (the loss keeps
    creeping down as probabilities sharpen), so cap the rounds by size. Measured on the
    benchmark: 300 rounds at n<=2000 cost no accuracy and cut boosting time 3-5x."""
    if n <= 2000:
        return 300
    if n <= 10_000:
        return 600
    if n <= 50_000:
        return 1000
    return 1500


def _hgb_params(ctx: FitContext, lr: float, leaves: int, l2: float, max_features: float, min_leaf: int | None):
    n = ctx.n_rows
    min_leaf = min_leaf if min_leaf is not None else int(np.clip(n // 40, 3, 20))
    max_iter = _iteration_cap(n)
    if ctx.task == "classification" and ctx.n_classes >= 5:  # one tree per class per round
        max_iter, lr = min(max_iter, 250), max(lr, 0.1)
    if ctx.iter_budget and ctx.iter_budget < max_iter:
        # Fewer rounds affordable: take proportionally bigger steps (bounded).
        lr = min(0.3, lr * float(np.sqrt(max_iter / ctx.iter_budget)))
        max_iter = ctx.iter_budget
    params = dict(
        learning_rate=lr, max_iter=max_iter, max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
        l2_regularization=l2, max_features=max_features, random_state=ctx.seed,
        categorical_features=ctx.cat_mask if ctx.cat_mask is not None and ctx.cat_mask.any() else None,
        early_stopping=True, validation_fraction=0.1 if n >= 2000 else 0.15, n_iter_no_change=30,
    )
    small = n < 120 or (ctx.task == "classification" and ctx.min_class_count < 3)
    if small:  # internal validation split would be unreliable (or impossible to stratify)
        params.update(early_stopping=False, max_iter=150 if n >= 60 else 80, learning_rate=0.08)
    return params


def make_hgb(lr=0.06, leaves=31, l2=1.0, max_features=1.0, min_leaf=None, loss=None):
    def build(ctx: FitContext):
        params = _hgb_params(ctx, lr, leaves, l2, max_features, min_leaf)
        if ctx.task == "classification":
            return HistGradientBoostingClassifier(**params)
        return HistGradientBoostingRegressor(loss=loss or "squared_error", **params)
    return build


def make_linear(C: float = 1.0):
    def build(ctx: FitContext):
        if ctx.task == "classification":
            return LogisticRegression(C=C, max_iter=1000, tol=1e-4)
        return RidgeCV(alphas=np.logspace(-3, 3, 13))
    return build


def make_extra_trees():
    def build(ctx: FitContext):
        n = ctx.n_rows
        n_est = 200 if n <= 20000 else 100
        common = dict(n_estimators=n_est, n_jobs=ctx.n_threads, random_state=ctx.seed)
        if ctx.task == "classification":
            return ExtraTreesClassifier(max_features="sqrt", min_samples_leaf=1, **common)
        return ExtraTreesRegressor(max_features=0.8, min_samples_leaf=2, **common)
    return build


def make_knn():
    def build(ctx: FitContext):
        k = int(np.clip(round(np.sqrt(ctx.n_rows) / 2), 5, 25))
        k = max(1, min(k, ctx.n_rows - 1))
        if ctx.task == "classification":
            return KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=ctx.n_threads)
        return KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=ctx.n_threads)
    return build


def make_svm():
    def build(ctx: FitContext):
        if ctx.task == "classification":
            cv = int(min(3, max(2, ctx.min_class_count)))
            return CalibratedClassifierCV(SVC(C=1.0, gamma="scale"), cv=cv, method="sigmoid", ensemble=False)
        return TransformedTargetRegressor(regressor=SVR(C=1.0, epsilon=0.1, gamma="scale"),
                                          transformer=StandardScaler())
    return build


def portfolio(task: str, y: np.ndarray, metric: str) -> list:
    """Ordered candidate list. The linear model goes first: it is nearly free and gets a
    real model (not just the majority-class baseline) written within a second or two."""
    linear = Candidate("linear", "linear", make_linear(), cost=0.1, tags=["linear"])
    hgb = Candidate("hgb", "tree", make_hgb(), cost=1.0, tags=["gbm"])
    hgb_log = Candidate("hgb_log", "tree", make_hgb(), target_transform="log1p", cost=1.0, tags=["gbm"])
    linear_log = Candidate("linear_log", "linear", make_linear(), target_transform="log1p", cost=0.1,
                           tags=["linear"])
    cands = [linear, hgb]
    if task == "regression":
        y = np.asarray(y, dtype=float)
        nonneg = bool(np.all(y >= 0))
        skew = float(((y - y.mean()) ** 3).mean() / (y.std() ** 3 + 1e-12)) if y.size > 2 else 0.0
        counts = nonneg and bool(np.all(np.mod(y, 1) == 0)) and np.unique(y).size > 2
        if metric == "rmsle" and nonneg:
            cands = [linear_log, hgb_log]
        else:
            if nonneg and skew > 0.75:
                cands = [linear, hgb, hgb_log, linear_log]
            if metric == "mae":
                cands.insert(2, Candidate("hgb_l1", "tree", make_hgb(loss="absolute_error"), cost=1.2, tags=["gbm"]))
            if counts:
                cands.append(Candidate("hgb_poisson", "tree", make_hgb(loss="poisson"), cost=1.0, tags=["gbm"]))
    follow = task == "regression"
    cands += [
        Candidate("extra_trees", "tree_imputed", make_extra_trees(), cost=1.3, max_rows=300_000, tags=["trees"],
                  follow_transform=follow),
        Candidate("knn", "linear", make_knn(), cost=0.4, max_rows=30_000, max_cols=400, tags=["knn"],
                  follow_transform=follow),
        Candidate("svm", "linear", make_svm(), cost=1.0, max_rows=4_000, max_cols=600, tags=["svm"],
                  follow_transform=follow),
        Candidate("hgb_b", "tree", make_hgb(lr=0.05, leaves=15, l2=0.5, max_features=0.6, min_leaf=None),
                  cost=1.3, tags=["gbm"], follow_transform=follow),
    ]
    return cands


def fit_predict(cand: Candidate, ctx: FitContext, X_tr, y_tr: np.ndarray, X_eval: list) -> list:
    """Fit on (X_tr, y_tr) and return predictions for every matrix in X_eval.

    Classification returns (n, K) probabilities with columns for all K classes,
    even when the training fold lacks some of them.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ctx.task == "classification":
            present = np.unique(y_tr)
            if present.size < 2:
                out = []
                for X in X_eval:
                    P = np.zeros((X.shape[0], ctx.n_classes))
                    P[:, present[0]] = 1.0
                    out.append(P)
                return out
            model = cand.build(ctx)
            model.fit(X_tr, y_tr)
            cols = np.asarray(model.classes_, dtype=int)
            out = []
            for X in X_eval:
                p = model.predict_proba(X)
                P = np.zeros((X.shape[0], ctx.n_classes))
                P[:, cols] = p
                out.append(P)
            return out
        y_fit = np.log1p(np.maximum(y_tr, 0)) if cand.target_transform == "log1p" else y_tr
        model = cand.build(ctx)
        model.fit(X_tr, y_fit)
        out = []
        for X in X_eval:
            p = np.asarray(model.predict(X), dtype=float)
            if cand.target_transform == "log1p":
                p = np.expm1(np.clip(p, -50, 50))
            out.append(np.where(np.isfinite(p), p, np.nanmean(y_tr)))
        return out
