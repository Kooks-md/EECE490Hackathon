"""Classification / regression engine.

Protocol
1. Write a baseline prediction immediately (majority class / mean), so a valid
   output exists even if everything after this point fails.
2. K-fold cross-validation (stratified for classification). Every candidate is
   fitted on the same folds; each fold model also predicts the rows we must
   output, and those predictions are averaged across folds ("fold bagging").
   No refit on the full data is needed, which saves the most expensive step.
3. After each candidate finishes, the out-of-fold predictions are re-blended
   (Caruana selection), the decision rule is re-tuned, and predictions are
   rewritten atomically. Running out of time therefore never loses work.
4. Time control. A single fit cannot be interrupted, so cost is bounded before
   it is paid: a short timing probe caps the number of boosting rounds so one
   CV pass fits the budget; the number of folds adapts to the measured cost of
   the first boosting fold; and any candidate whose projected cost exceeds the
   remaining budget is skipped.
"""

from __future__ import annotations

import dataclasses
import time
import traceback

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from . import ensemble as ens
from .budget import Budget
from .dataio import Dataset, OutputWriter
from .features import FeatureSpace
from .metrics import needs_proba, score_labels, score_proba, score_regression
from .models import Candidate, FitContext, fit_predict, portfolio

MAX_TRAIN_ROWS = 150_000


def _make_folds(y: np.ndarray, task: str, seed: int) -> list:
    n = len(y)
    if n <= 600:
        n_splits, repeats = 5, 2      # small data: repeat CV to steady model selection
    elif n <= 50_000:
        n_splits, repeats = 5, 1
    else:
        n_splits, repeats = 4, 1
    n_splits = int(max(2, min(n_splits, n // 2 if n >= 4 else 2)))
    folds = []
    for r in range(repeats):
        if task == "classification":
            counts = np.bincount(y)
            if counts[counts > 0].max() >= n_splits:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
            else:
                splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        folds.extend(list(splitter.split(np.zeros(n), y)))
    return folds


def _te_block(fs: FeatureSpace, rows_fit: np.ndarray, y_fit: np.ndarray, others: list, task: str,
              n_classes: int, seed: int):
    if fs.highcard is None:
        return None, [None] * len(others)
    if task == "classification":
        target_type = "binary" if n_classes == 2 else "multiclass"
    else:
        target_type = "continuous"
    enc = TargetEncoder(target_type=target_type, smooth="auto", cv=5, shuffle=True, random_state=seed)
    tr = enc.fit_transform(fs.highcard[rows_fit], y_fit)
    oth = [enc.transform(fs.highcard[r]) for r in others]
    return tr, oth


def _stack(base, extra):
    if extra is None:
        return base
    if sparse.issparse(base):
        return sparse.hstack([base, sparse.csr_matrix(extra)], format="csr")
    return np.hstack([base, extra])


def _fold_matrices(fs: FeatureSpace, cand: Candidate, rows_fit, y_fit, others, task, n_classes, seed):
    base = fs.view(cand.view)
    te_tr, te_oth = _te_block(fs, rows_fit, y_fit, others, task, n_classes, seed)
    if te_tr is not None and cand.view == "linear":
        mu, sd = te_tr.mean(axis=0), te_tr.std(axis=0) + 1e-9
        te_tr = (te_tr - mu) / sd
        te_oth = [(t - mu) / sd for t in te_oth]
    X_fit = _stack(base[rows_fit], te_tr)
    X_oth = [_stack(base[r], t) for r, t in zip(others, te_oth)]
    if cand.view == "tree":
        n_extra = 0 if te_tr is None else te_tr.shape[1]
        mask = np.concatenate([fs.tree_cat, np.zeros(n_extra, dtype=bool)])
    else:
        mask = None
    return X_fit, X_oth, mask


class SupervisedRun:
    def __init__(self, ds: Dataset, fs: FeatureSpace, metric: str, budget: Budget, writer: OutputWriter,
                 seed: int, n_threads: int, log):
        self.ds, self.fs, self.metric, self.budget, self.writer = ds, fs, metric, budget, writer
        self.seed, self.n_threads, self.log = seed, n_threads, log
        self.task = ds.task
        self.report: dict = {"metric": metric, "candidates": [], "errors": []}

    # ------------------------------------------------------------------ setup
    def _prepare_target(self):
        y_raw = self.ds.train[self.ds.target]
        if self.task == "regression":
            y_num = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
            lab = np.isfinite(y_num)
            self.idx_lab = np.nonzero(lab)[0]
            self.y = y_num[lab]
            self.classes = None
            self.n_classes = 0
        else:
            lab = y_raw.notna().to_numpy()
            self.idx_lab = np.nonzero(lab)[0]
            values = y_raw[lab].to_numpy()
            if values.dtype == object:
                values = values.astype(str)
            self.classes, self.y = np.unique(values, return_inverse=True)
            self.n_classes = len(self.classes)
            counts = np.bincount(self.y, minlength=self.n_classes)
            self.pos = int(np.argmin(counts)) if self.n_classes == 2 else 1
            if self.ds.positive_label is not None and self.n_classes == 2:
                for j, c in enumerate(self.classes):
                    if str(c) == str(self.ds.positive_label):
                        self.pos = j
        if len(self.idx_lab) > MAX_TRAIN_ROWS:
            rng = np.random.default_rng(self.seed)
            keep = np.sort(rng.choice(len(self.idx_lab), MAX_TRAIN_ROWS, replace=False))
            self.idx_lab, self.y = self.idx_lab[keep], self.y[keep]
            self.report["subsampled_training_rows"] = MAX_TRAIN_ROWS
        if self.ds.predict_on == "test":
            self.idx_pred = np.arange(self.fs.n_train, self.fs.n_all)
        else:
            self.idx_pred = np.arange(self.fs.n_train)

    def _baseline(self):
        n = len(self.idx_pred)
        if self.task == "classification":
            counts = np.bincount(self.y, minlength=self.n_classes)
            P = np.tile(counts / counts.sum(), (n, 1))
            self._write_classification(P, np.ones(self.n_classes), tag="baseline")
        else:
            value = float(np.median(self.y)) if self.metric == "mae" else float(np.mean(self.y))
            self.writer.predictions(np.full(n, value))
        self.report["baseline_written_at_s"] = round(self.budget.elapsed(), 2)

    def _write_classification(self, P_pred, w, tag=""):
        labels = self.classes[ens.apply_rule(P_pred, w)]
        self.writer.predictions(labels, label_values=True)
        try:
            self.writer.probabilities(P_pred, self.classes)
        except Exception:  # noqa: BLE001 - probabilities are optional
            pass

    # ------------------------------------------------------------ scoring
    def _score_oof(self, oof, valid):
        y = self.y[valid]
        if self.task == "classification":
            P = oof[valid]
            if needs_proba(self.metric):
                return score_proba(self.metric, y, P, self.n_classes)
            return score_labels(self.metric, y, np.argmax(P, axis=1), pos=self.pos)
        return score_regression(self.metric, y, oof[valid])

    # ------------------------------------------------------------ main loop
    def run(self) -> dict:
        self._prepare_target()
        if len(self.y) == 0:
            raise ValueError("no labelled training rows")  # the watchdog's fallback takes over
        if len(self.idx_pred) == 0:
            self.writer.predictions(np.array([], dtype=float))
            self.report["note"] = "no rows to predict"
            return self.report
        self._baseline()
        if self.task == "classification" and self.n_classes < 2:
            self.report["note"] = "single class in training data; predicting it everywhere"
            return self.report
        if len(self.y) < 4 or self.fs.tree.shape[1] == 0:
            self.report["note"] = "too little data/features for model selection; baseline kept"
            return self.report

        folds = _make_folds(self.y, self.task, self.seed)
        self.report["cv"] = {"planned_folds": len(folds)}
        cands = portfolio(self.task, self.y, self.metric)
        results: dict[str, dict] = {}
        gbm_fold_time = None        # measured cost of one boosting fold; prices every later candidate
        self.iter_budget = None     # boosting rounds affordable per fit (None = not probed yet)
        n_lab = len(self.y)
        K = max(self.n_classes, 1)

        prefer_log = self.metric == "rmsle"
        for cand in cands:
            if cand.follow_transform and prefer_log and cand.target_transform is None:
                cand = dataclasses.replace(cand, name=f"{cand.name}_log", target_transform="log1p")
            is_gbm = "gbm" in cand.tags
            first_gbm = is_gbm and gbm_fold_time is None
            n_cols = self.fs.view(cand.view).shape[1]
            if n_lab > cand.max_rows or n_cols > cand.max_cols:
                self.report["candidates"].append({"name": cand.name, "status": "skipped (size)"})
                continue
            est = None if gbm_fold_time is None else cand.cost * gbm_fold_time * len(folds)
            if est is not None and not self.budget.can_afford(est):
                self.report["candidates"].append({"name": cand.name, "status": "skipped (time)",
                                                  "estimated_s": round(est, 2)})
                continue
            if self.budget.remaining() < 0.5:
                break
            shape = (n_lab, K) if self.task == "classification" else (n_lab,)
            oof_sum, oof_cnt = np.zeros(shape), np.zeros(n_lab)
            pred_shape = (len(self.idx_pred), K) if self.task == "classification" else (len(self.idx_pred),)
            pred_sum, n_done = np.zeros(pred_shape), 0
            t_start, aborted, err = time.time(), False, None
            fold_i = 0
            while fold_i < len(folds):
                tr_pos, va_pos = folds[fold_i]
                t_f = time.time()
                try:
                    rows_tr = self.idx_lab[tr_pos]
                    others = [self.idx_lab[va_pos], self.idx_pred]
                    X_tr, (X_va, X_pr), mask = _fold_matrices(
                        self.fs, cand, rows_tr, self.y[tr_pos], others, self.task, self.n_classes,
                        self.seed + fold_i)
                    counts = np.bincount(self.y[tr_pos], minlength=K) if self.task == "classification" else None
                    ctx = FitContext(task=self.task, n_classes=self.n_classes, n_rows=len(tr_pos),
                                     n_threads=self.n_threads, seed=self.seed + fold_i,
                                     cat_mask=mask if mask is not None else np.zeros(0, bool),
                                     min_class_count=int(counts[counts > 0].min()) if counts is not None else 0)
                    if is_gbm:
                        if self.iter_budget is None:
                            self.iter_budget = self._probe_rounds(cand, ctx, X_tr, self.y[tr_pos], len(folds))
                            t_f = time.time()  # the probe is not part of the fold's cost
                        ctx.iter_budget = self.iter_budget or None
                    p_va, p_pr = fit_predict(cand, ctx, X_tr, self.y[tr_pos], [X_va, X_pr])
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"
                    self.report["errors"].append({"candidate": cand.name, "fold": fold_i, "error": err,
                                                  "trace": traceback.format_exc(limit=3)})
                    aborted = True
                    break
                oof_sum[va_pos] += p_va
                oof_cnt[va_pos] += 1
                pred_sum += p_pr
                n_done += 1
                dt = time.time() - t_f
                if first_gbm and fold_i == 0:
                    gbm_fold_time = max(dt, 1e-3)
                    # Cap boosting's CV at ~40% of what is left; every later model shares these folds.
                    affordable = 1 + int(0.40 * max(self.budget.remaining(), 0) / gbm_fold_time)
                    if affordable < len(folds):
                        folds = folds[:max(1, affordable)]
                        self.report["cv"]["folds_reduced_to"] = len(folds)
                left = len(folds) - (fold_i + 1)
                if left and dt * left * 1.1 > self.budget.remaining():
                    aborted, err = True, "time"
                    break
                fold_i += 1
            elapsed = time.time() - t_start
            if aborted or n_done < len(folds):
                self.report["candidates"].append({"name": cand.name, "status": f"aborted ({err})",
                                                  "seconds": round(elapsed, 2)})
                if first_gbm and gbm_fold_time is None:
                    gbm_fold_time = max(elapsed, 0.05)
                continue
            valid = oof_cnt > 0
            oof = oof_sum / np.maximum(oof_cnt, 1)[:, None] if oof_sum.ndim == 2 else oof_sum / np.maximum(oof_cnt, 1)
            results[cand.name] = {"oof": oof, "pred": pred_sum / n_done, "valid": valid}
            score = self._score_oof(oof, valid)
            self.report["candidates"].append({"name": cand.name, "status": "ok", "cv_score": round(float(score), 6),
                                              "seconds": round(elapsed, 2), "folds": n_done})
            self.log(f"{cand.name}: cv {self.metric}={score:.5f} ({elapsed:.1f}s)")
            if "hgb" in results and "hgb_log" in results and cand.name in ("hgb", "hgb_log"):
                prefer_log = self._score_oof(results["hgb_log"]["oof"], results["hgb_log"]["valid"]) > \
                    self._score_oof(results["hgb"]["oof"], results["hgb"]["valid"])
                self.report["log_target_preferred"] = bool(prefer_log)
            try:
                self._finalize(results)
            except Exception as exc:  # noqa: BLE001
                self.report["errors"].append({"candidate": "ensemble", "error": f"{type(exc).__name__}: {exc}",
                                              "trace": traceback.format_exc(limit=3)})
        self.report["cv"]["folds_used"] = len(folds)
        return self.report

    def _probe_rounds(self, cand: Candidate, ctx: FitContext, X_tr, y_tr, n_folds: int) -> int:
        """Time boosting rounds, then cap them so one CV pass uses <= 50% of the budget left.

        Returns 0 when no cap is needed. A single fit cannot be interrupted once
        started, so this is what keeps e.g. 50-class problems inside the limit.
        """
        import warnings
        y_fit = np.log1p(np.maximum(y_tr, 0)) if cand.target_transform == "log1p" else y_tr
        try:
            secs = []
            for rounds in (2, 10):  # difference removes the fixed binning cost
                model = cand.build(ctx)
                model.set_params(max_iter=rounds, early_stopping=False)
                t = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_tr, y_fit)
                secs.append(time.time() - t)
            per_round = max((secs[1] - secs[0]) / 8.0, secs[1] / 40.0, 1e-5)
        except Exception:  # noqa: BLE001
            return 0
        allowed = 0.50 * max(self.budget.remaining(), 0.0) / max(n_folds, 1)
        rounds = int(allowed / per_round)
        self.report["boosting_probe"] = {"seconds_per_round": round(per_round, 5), "round_cap": rounds}
        return max(rounds, 10)

    # ------------------------------------------------------------ blending
    def _finalize(self, results: dict):
        names = list(results)
        valid = np.logical_and.reduce([results[n]["valid"] for n in names])
        y = self.y[valid]
        oofs = {n: results[n]["oof"][valid] for n in names}
        preds = {n: results[n]["pred"] for n in names}
        if self.task == "classification":
            weights, _ = ens.caruana(oofs, lambda P: ens.log_loss_score(y, P))
            P_oof = ens.blend(oofs, weights)
            w, s_ens = ens.fit_rule(P_oof, y, self.metric, self.n_classes, self.pos)
            if needs_proba(self.metric):
                s_ens = score_proba(self.metric, y, P_oof, self.n_classes)
            chosen, rule = weights, w
            # Compare with the best single model (with its own tuned rule).
            for n in names:
                w_n, s_n = ens.fit_rule(oofs[n], y, self.metric, self.n_classes, self.pos)
                if needs_proba(self.metric):
                    s_n = score_proba(self.metric, y, oofs[n], self.n_classes)
                if s_n > s_ens + 1e-4:
                    chosen, rule, s_ens = {n: 1.0}, w_n, s_n
            P_pred = ens.blend(preds, chosen)
            if self.ds.predict_on == "train":
                P_pred = self._merge_oof(P_pred, ens.blend({n: results[n]["oof"] for n in names}, chosen),
                                         valid)
            self._write_classification(P_pred, rule)
            self.report["ensemble"] = {"weights": {k: round(v, 4) for k, v in chosen.items()},
                                       "cv_score": round(float(s_ens), 6),
                                       "decision_weights": [round(float(x), 5) for x in rule]}
        else:
            weights, s_ens = ens.caruana(oofs, lambda p: score_regression(self.metric, y, p))
            pred = ens.blend(preds, weights)
            if self.ds.predict_on == "train":
                pred = self._merge_oof(pred, ens.blend({n: results[n]["oof"] for n in names}, weights), valid)
            lo, hi = float(self.y.min()), float(self.y.max())
            span = hi - lo
            lo_c, hi_c = lo - 0.05 * span, hi + 0.05 * span
            if lo >= 0:
                lo_c = max(0.0, lo_c)
            pred = np.clip(pred, lo_c, hi_c)
            self.writer.predictions(pred)
            self.report["ensemble"] = {"weights": {k: round(v, 4) for k, v in weights.items()},
                                       "cv_score": round(float(s_ens), 6),
                                       "clip": [round(lo_c, 6), round(hi_c, 6)]}

    def _merge_oof(self, fold_avg, oof_full, valid):
        """predict_on == train: use out-of-fold predictions for labelled rows."""
        out = np.array(fold_avg, copy=True)
        pos_in_pred = self.idx_lab[valid]
        out[pos_in_pred] = oof_full[valid]
        return out


def run_supervised(ds, fs, metric, budget, writer, seed, n_threads, log) -> dict:
    return SupervisedRun(ds, fs, metric, budget, writer, seed, n_threads, log).run()
