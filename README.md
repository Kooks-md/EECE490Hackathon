# Unknown Dataset Hackathon: `autotab`

One program that takes an unfamiliar table, works out what it is, and solves the task that comes with it: **classification**, **regression** or **clustering**. It returns predictions plus a structured data profile, within a 60-second limit per dataset on CPU.

**Team:** Karim Sadek · Mohamad Hajj Chehade · Mahdi al Zein · Mohamad Al Zein (EECE 490, Introduction to Machine Learning)

```bash
pip install -r requirements.txt          # Python 3.13, pinned versions (requirements-lock.txt = full transitive pins)
python main.py DATASET_DIR OUTPUT_DIR    # the standard entry point
python main.py examples/classification_churn/public out/   # try it
```

`python -m autotab DATASET_DIR OUTPUT_DIR` and `autotab.solve(DATASET_DIR, OUTPUT_DIR)` do the same thing. Add `--verbose` to see progress, or `--time-limit S` to change the 60 s default.

---

## Contents

1. [Input / output contract](#1-input--output-contract)
2. [How it works](#2-how-it-works)
3. [Data profile](#3-data-profile)
4. [Results on the local benchmark](#4-results-on-the-local-benchmark)
5. [Reproducing everything](#5-reproducing-everything)
6. [Assumptions](#6-assumptions)
7. [Repository layout](#7-repository-layout)
8. [Limitations](#8-limitations)

---

## 1. Input / output contract

The organisers' contract, practice datasets and checker were released at kickoff and are not in this repository. We therefore defined the contract below. The loader is deliberately lenient: it accepts alternative file and key names, infers what is missing, and records every inference in the profile under `notes`. Adapting to a different contract only touches `autotab/dataio.py`.

### Input: `DATASET_DIR/`

| File | Content |
|---|---|
| `metadata.json` | `task` (`classification` / `regression` / `clustering`), `target`, optional `id_column`, `feature_types` (`{column: numeric \| categorical \| datetime \| text \| id}`), `n_clusters` (clustering), optional `metric`, optional `positive_label` |
| `train.csv` | Supervised: features and target. Clustering: the rows to cluster. |
| `test.csv` | Supervised: features of the rows to predict. Clustering: optional; if present, these are the rows to label. |

Accepted spellings include `task_type`, `label`, `target_column`, `k`, `num_clusters`, and `feature_types` given as a dict, as a list of `{name, type}`, or as `{type: [columns]}`. Missing pieces are inferred:

- **Target.** The one column in `train.csv` that is absent from `test.csv`.
- **Task.** From the target's dtype and cardinality.
- **Number of clusters.** By silhouette over k = 2..10.

### Output: `OUTPUT_DIR/`

| File | Content |
|---|---|
| `predictions.csv` | `id,prediction`, one row per row to predict, in input order. `id` is the `id_column` value, or the 0-based row index if there is none. Class labels are written exactly as they appear in the training data (`1`, not `1.0`). |
| `profile.json` | The structured data profile (§3) plus a modelling report. |
| `probabilities.csv` | Classification only: `id` plus one `proba_<class>` column per class. |

Metrics:

- **Defaults.** `f1_macro` for classification, `rmse` for regression, adjusted Rand index (`ari`) for clustering.
- **Overriding.** Set `metric` in `metadata.json`. Supported values:
  - Classification: `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`, `f1`, `mcc`, `roc_auc`, `log_loss`.
  - Regression: `rmse`, `mse`, `mae`, `r2`, `rmsle`, `mape`.
  - Clustering: `ari`, `nmi`, `ami`.
- **Effect.** The metric changes what is optimised: model selection, ensemble weights, decision thresholds and target transforms.

---

## 2. How it works

```
main.py (watchdog, stdlib only)
 └─ autotab.worker (child process)
     ├─ dataio      load, align columns, infer what metadata leaves out
     ├─ analysis    parse types; flag id-like / constant / duplicate columns
     ├─ profile     write profile.json (within ~1.5 s)
     ├─ features    tree view + linear view, built once for all rows
     ├─ supervised  CV portfolio -> ensemble -> metric-specific decision rule
     └─ clustering  noise-feature removal -> KMeans / Gaussian mixtures -> label-free selection
```

### 2.1 Never lose a dataset

A crash or timeout on one dataset costs about 4.7 of the 85 model-quality points. That is more than almost any modelling improvement is worth, so robustness comes first.

- **Watchdog process.** `main.py` imports only the standard library and runs the pipeline in a child process. If the child crashes, hangs inside native code, or is still running near the limit, the watchdog kills it at `limit - 1.5 s`.
- **Baseline written first, then improved.** The worker writes a baseline prediction within about 2 seconds (majority class, mean, or a quick KMeans). It then overwrites that file each time a better model finishes. Every write goes to a temp file followed by an atomic rename, so a kill at any instant leaves a valid file.
- **Stdlib fallback.** If no valid output exists at all (for example, an unreadable CSV), the watchdog writes a majority-class / mean prediction using only the standard library.
- **Isolated failures.** Every model, fold and column is guarded. One failure is logged in the profile under `modelling.errors` and the run continues.
- **Output hygiene.**
  - numpy types are converted before JSON; NaN and Inf become `null` (strict JSON).
  - Label types round-trip unchanged.
  - Regression predictions are always finite.

### 2.2 Time budget

All deadlines are measured from the moment `main.py` starts, so interpreter start-up and imports count, exactly as they do for the grader.

- **Deadlines.**
  - Model fitting may *start* until 70% of the limit (42 s of 60).
  - Final writes finish by about 55 s.
  - The watchdog kills the worker at 58.5 s.
- **Bounding each fit.** A running fit cannot be interrupted, so its cost is bounded before it is paid:
  - *Round caps by size.* Boosting rounds are capped by data size: 300 up to 2k rows, 600 up to 10k, 1000 up to 50k. Early stopping on validation loss rarely triggers on small, clean data, because the loss keeps creeping down as probabilities sharpen. On small tables the cap cut boosting time 3–5x, at a cost of at most 0.2 accuracy points in our tests.
  - *Timing probe.* A probe fits 2 and 10 boosting rounds; the difference cancels the fixed binning cost. It then caps rounds so that one CV pass of boosting uses at most half of the remaining budget. When the cap binds, the learning rate is raised by the square root of the cut, up to 0.3.
  - *Adaptive folds.* The number of folds adapts to the measured cost of the first boosting fold.
  - *Pricing other candidates.* Every other candidate is priced from that fold and skipped if unaffordable.
- **Threads.** The thread count comes from the container's cgroup CPU quota, not `os.cpu_count()`. `n_jobs=-1` inside a CPU-limited container oversubscribes and can multiply runtime.

Measured on 2 CPU cores over the 72 benchmark tables (whole process, including start-up):

- **Wall time.** Median 6.0 s, mean 8.5 s, maximum 37.0 s.
- **Stress tables.** All finish within 24 s:
  - 150k rows;
  - 50 classes;
  - 500 features;
  - a 10,000-level categorical;
  - 60k-row clustering.

### 2.3 Data preparation (`analysis.py`, `features.py`)

- **Type parsing.** Declared feature types are used but verified.
  - Numbers stored as text are converted, for example Telco's `TotalCharges` with blank strings.
  - Date strings become calendar parts: year, month, day, weekday, day of year, hour, weekend flag, elapsed days. In Bike Sharing the hour carries much of the signal. A column declared categorical whose values are mostly distinct date strings is treated as a datetime.
  - Long free text becomes length and word-count features.
- **Dropped columns.** ID-like columns, constant columns and exact duplicates are dropped before modelling. Definitions are in §3.
- **Two views, built once for all rows.** Only label-free statistics (medians, scales, category vocabularies) are fitted on all rows. That is transductive but leaks no labels.
  - *Tree view.*
    - Raw numerics, with missing values kept as NaN.
    - Categoricals with up to 64 levels as integer codes, split natively by histogram gradient boosting.
    - Higher-cardinality categoricals as frequency encoding plus target encoding.
  - *Linear view.*
    - Numerics: `log1p` for skewed non-negative columns, median imputation plus missing indicators, standardisation, clipping at ±5 sd.
    - One-hot encoding for categoricals and for small-integer codes such as season, weather or hour.
- **Target encoding.** The one label-dependent transform. It is cross-fitted inside each training fold with scikit-learn's `TargetEncoder`, so it never sees the rows it encodes.

### 2.4 Classification and regression (`supervised.py`, `models.py`, `ensemble.py`)

**Portfolio.** A small, fixed set of configurations. The linear model runs first: it costs well under a second and replaces the majority-class baseline almost immediately.

| Candidate (run order) | Model | View |
|---|---|---|
| `linear` | Logistic regression / RidgeCV | linear |
| `hgb` | HistGradientBoosting, early stopping, native categoricals | tree |
| `extra_trees` | ExtraTrees | tree (imputed) |
| `knn` | k-NN, distance-weighted (n ≤ 30k) | linear |
| `svm` | RBF SVM, sigmoid-calibrated (n ≤ 4k) | linear |
| `hgb_b` | second boosting config: shallower, feature subsampling | tree |
| regression extras | `hgb_log` / `linear_log` (log1p target when non-negative and skewed), `hgb_poisson` (counts), `hgb_l1` (when the metric is MAE) | |

Hyperparameters are fixed, conservative defaults checked against the benchmark, not searched live. Inside 60 seconds a random search mostly fits noise, and the time is better spent on model diversity.

**Protocol.**

1. **Folds.** Stratified K-fold: 5 folds, repeated twice when n ≤ 600 to steady model selection on small data; 4 folds above 50k rows.
2. **Fold bagging.** Every candidate uses the same folds. Each fold model also predicts the test rows, and those predictions are averaged. No final refit is needed, which removes the most expensive step and gives a bagged ensemble for free.
3. **Ensembling.** After each candidate finishes, the out-of-fold predictions are re-blended with greedy ensemble selection (Caruana et al., 2004) and the predictions are rewritten.
   - Classification blends are selected on log-loss, a proper scoring rule, so the blend stays calibrated.
   - Regression blends are selected on the task metric.
   - The best single model is kept instead if it beats the blend.
4. **Decision rule.** This is where much of the gain on imbalanced data comes from.
   - Accuracy: argmax.
   - F1, balanced accuracy, MCC: a binary threshold, or per-class probability weights for multiclass, tuned on out-of-fold predictions. The threshold is the middle of the near-optimal plateau, not its noisiest peak.
5. **Regression specifics.**
   - Whether a log target helps is decided by comparing `hgb` against `hgb_log`. The later candidates then follow the winner.
   - Predictions are clipped to the training range ±5%, and to ≥ 0 for non-negative targets.

### 2.5 Clustering (`clustering.py`)

Clustering is a third of model quality and the least standard part. The work went into two places: removing irrelevant features, and choosing an algorithm without labels.

**Representation.** Median-imputed, standardised numerics. Each categorical becomes a one-hot block scaled to unit total variance, so a five-level categorical weighs as much as one numeric column.

**Irrelevant-feature removal, two stages.**

1. *Dependence screen.* Cluster structure makes the features that carry it statistically dependent on each other. A column generated independently of everything else is not. Pairwise chi-square tests on discretised features (Bonferroni over all pairs, plus a Cramér's V ≥ 0.1 floor) drop columns that depend on nothing.
   - This needs no clustering first, so strong noise columns cannot mask the signal.
   - The earlier design, which clustered first and then tested features, failed exactly that way: a random binary column dominated KMeans and every feature looked irrelevant.
2. *Leave-one-out relevance.* For each remaining feature, cluster on the other features. Then test with chi-square whether the feature's *distribution*, not just its mean, differs across those clusters.
   - This drops features related to other columns but not to the cluster structure, such as `sex` in Palmer Penguins.
   - Testing means only failed when clusters sit symmetrically, for example high and low spenders at the same income.

If a stage finds nothing clearly relevant, it keeps everything.

**Candidates.** KMeans (10 inits) and Gaussian mixtures with spherical, diagonal, tied and full covariance. Mixtures are fitted on ≤ 20k rows, then predict all rows.

**Selection rule.** Take the mixture with the lowest BIC. Fall back to KMeans if the mixture's partition is less reproducible on 75% subsamples than KMeans' (stability lower by more than 0.05). BIC alone picks degenerate solutions when categorical one-hot columns let a component collapse; the stability check catches those.

Rules compared (mean ARI over 35 labelled clustering tables; see `bench/` and §4):

| Rule | Mean ARI |
|---|---|
| KMeans only | 0.790 |
| Ward only | 0.776 |
| Silhouette-best | 0.785 |
| Lowest BIC | 0.809 |
| **Lowest BIC + stability guard (used)** | **0.813** |
| Oracle (best candidate per dataset) | 0.848 |

### 2.6 What the profile is for

The profile is written before any model is fitted, so it is correct even if modelling later fails. A `modelling` section is added afterwards. It covers:

- every candidate's CV score and time;
- the ensemble weights and decision rule;
- the clustering features used and dropped;
- the chosen algorithm, BIC and stability values.

---

## 3. Data profile

`profile.json` has a fixed top-level schema (`schema_version: "1.0"`):

| Key | Content |
|---|---|
| `dataset` | name, task, target, id column, metric, n_clusters, row counts, column counts, features used |
| `feature_types` | column lists per inferred type (`numeric`, `categorical`, `datetime`, `text`, `boolean`, `id`) |
| `target` | classification: classes, counts, majority/minority, imbalance ratio and level. Regression: min/max/mean/std/median/skewness, integer and non-negative flags. |
| `issues` | missing values (cells, fraction, per column), constant columns, duplicate columns and groups, id-like columns, high-cardinality columns, scale disparity, skewed columns, highly correlated pairs, duplicate rows, class imbalance |
| `columns` | per column: role, declared vs inferred type, missing count and fraction, distinct values, flags, whether used for modelling (and why not), summary statistics |
| `preprocessing`, `modelling`, `timings`, `notes` | what was done and why |

**Definitions.** These are stated precisely because a profile is only "correct" against a definition. The profile describes the training table after parsing (for clustering, the table being clustered).

| Term | Definition |
|---|---|
| missing | NaN after parsing. Empty and whitespace-only strings, `NA`, `N/A`, `null`, `None`, `nan`, `?` and infinite numbers count as missing. |
| constant | at most one distinct non-missing value; all-missing columns are constant. A column with one value plus missing cells is reported constant but kept for modelling, because its missingness can be informative. |
| duplicate | identical to an earlier column, either as read from the file or after parsing, with missing cells in the same rows. The first occurrence is kept. `duplicate_groups` lists each set of identical columns. |
| id-like | no missing values and a distinct value on every row, and either non-numeric, or integer-valued and (monotonic, or named like an ID). Floats are never id-like. |
| high cardinality | categorical with more than 50 distinct values |
| differently scaled | max/min standard deviation of the numeric features used is ≥ 10 |
| skewed | \|skewness\| > 1 |
| class imbalance | ratio = majority count / minority count. Level: balanced < 1.5, mild < 3, moderate < 10, severe ≥ 10. `is_imbalanced` means ratio ≥ 1.5. |
| highly correlated | \|Pearson r\| ≥ 0.95 among numeric features |

---

## 4. Results on the local benchmark

The official practice and assessment datasets were not available to us, so we built our own benchmark of **72 tables** (generators and builders in `bench/`):

- **26 real tables.** The five Kaggle examples from the brief, plus Titanic, Adult, California housing, Palmer Penguins, Pima, Ionosphere, Glass, Abalone, wine quality, seeds, and scikit-learn's bundled sets.
- **46 synthetic tables.** Each has a random mix of the issues listed in the brief injected and recorded as ground truth:
  - missing values;
  - rescaled columns;
  - numerics turned into categoricals;
  - constant columns, duplicate columns and irrelevant noise columns;
  - ID columns;
  - class imbalance;
  - nonlinear targets, count targets with a datetime column, and anisotropic clusters.

Each table is solved in a fresh process with the 60 s limit on 2 CPU cores. It is compared with a typical organiser-style baseline (`bench/baseline.py`): median imputation, scaling, one-hot encoding, then logistic regression, ridge or KMeans.

| Task | Tables | autotab | Baseline | Gain | Wall time, median / max |
|---|---|---|---|---|---|
| Classification (declared metric: macro-F1, accuracy or balanced accuracy) | 20 | **0.829** | 0.704 | +0.125 | 10.8 s / 37.0 s |
| Regression (R² on held-out rows) | 17 | **0.829** | 0.599 | +0.229 | 15.0 s / 34.2 s |
| Clustering (ARI vs hidden labels) | 35 | **0.822** | 0.623 | +0.199 | 1.9 s / 3.7 s |

- **Profile checks.** 2038 of 2038 pass, against an independent oracle and the injected ground truth (`bench/check.py`).
- **Contract failures.** 0 output-contract failures and 0 fallbacks used.

<details>
<summary>Per-table results on the real datasets</summary>

| Dataset | Task | Metric | autotab | Baseline | Wall (s) |
|---|---|---|---|---|---|
| adult_income | classification | f1_macro | 0.826 | 0.787 | 30.1 |
| ai4i_maintenance | classification | f1_macro | 0.865 | 0.689 | 9.3 |
| breast_cancer | classification | f1_macro | 0.963 | 0.985 | 11.3 |
| digits_cls | classification | accuracy | 0.982 | 0.969 | 37.0 |
| glass | classification | f1_macro | 0.630 | 0.532 | 13.5 |
| ionosphere | classification | accuracy | 0.932 | 0.864 | 8.6 |
| penguins_cls | classification | f1_macro | 0.990 | 0.990 | 13.0 |
| pima_diabetes | classification | balanced_accuracy | 0.785 | 0.735 | 8.7 |
| telco_churn | classification | f1_macro | 0.730 | 0.723 | 9.0 |
| titanic | classification | accuracy | 0.812 | 0.807 | 5.6 |
| abalone | regression | R² | 0.569 | 0.543 | 15.9 |
| auto_insurance_tiny (63 rows) | regression | R² | 0.719 | 0.691 | 6.5 |
| bike_sharing | regression | R² | 0.952 | 0.276 | 21.5 |
| california_housing | regression | R² | 0.847 | 0.638 | 34.2 |
| diabetes | regression | R² | 0.361 | 0.358 | 8.8 |
| house_prices | regression | R² | 0.879 | 0.719 | 18.7 |
| wine_quality | regression | R² | 0.440 | 0.345 | 7.5 |
| iris | clustering | ARI | 0.904 | 0.620 | 1.8 |
| wine | clustering | ARI | 0.847 | 0.897 | 1.9 |
| seeds | clustering | ARI | 0.838 | 0.773 | 1.8 |
| penguins | clustering | ARI | 0.898 | 0.590 | 2.1 |
| mall_customers | clustering | ARI | 0.955 | 0.608 | 2.0 |
| digits | clustering | ARI | 0.534 | 0.468 | 3.7 |
| breast_cancer | clustering | ARI | 0.671 | 0.654 | 2.3 |
| glass | clustering | ARI | 0.234 | 0.172 | 1.6 |
| abalone (sex) | clustering | ARI | 0.135 | 0.135 | 2.4 |

</details>

Stress tables, testing only that the time limit holds:

| Stress table | Task | Score | Wall (s) |
|---|---|---|---|
| 150,000 rows × 35 columns | classification | 0.979 | 22.6 |
| 50 classes, 20k rows | classification | 0.574 | 23.3 |
| 500 features, 2k rows | regression (R²) | 0.819 | 23.2 |
| 10,000-level categorical, 30k rows | classification | 0.651 | 15.3 |
| free text, `$1,234.56` strings, dd/mm/yyyy dates | regression (R²) | 0.948 | 12.1 |
| 60,000 rows + 5 noise columns | clustering (ARI) | 0.999 | 4.6 |
| 150 columns, 130 of them noise | clustering (ARI) | 1.000 | 2.9 |

Time-adaptive decisions (fold counts, boosting-round caps) depend on the measured speed of the machine, so scores can move in the third decimal between runs.

Where the gains come from:

- **Regression.** Datetime expansion and count/log targets: Bike Sharing goes from 0.28 to 0.95.
- **Clustering.** Removing irrelevant features: the seven customer-segment tables, including the real Mall data, go from 0.05–0.71 ARI to 0.85–1.00.
- **Imbalanced classification.** Metric-aware thresholds: AI4I goes from 0.69 to 0.87 macro-F1.

Where it loses:

- **Small clean tables** (breast cancer test set of 143 rows, wine clustering). A plain linear model or KMeans is already near-optimal there, and CV-based selection on a few hundred rows is noisy.
- **This benchmark is ours.** The synthetic generators encode our guesses about how the hidden tables were made, and the clustering selection rule was chosen on 35 of these tables. The real tables are the less biased evidence.

## 5. Reproducing everything

```bash
pip install -r requirements.txt pytest
python -m pytest -q tests                       # unit + end-to-end tests

# Synthetic benchmark with the issues from the brief injected (no download needed)
python bench/make_synthetic.py bench_data/syn --per-task 10 --seed 0          # 30 tables
python bench/make_synthetic.py bench_data/cluval --clustering-validation       # 16 clustering tables, seed 1

# Real tables: the five Kaggle examples and others (raw CSVs not committed; sources below)
python bench/make_real.py RAW_DIR bench_data/real                              # 26 tables

# Run and compare against the reference baseline (about 15 minutes on 2 cores)
python bench/run.py --data bench_data/real bench_data/syn bench_data/cluval --solver baseline --out baseline.json
python bench/run.py --data bench_data/real bench_data/syn bench_data/cluval --out ours.json --compare baseline.json

# Try the bundled examples (public/ = solver input, private/ = labels for checking)
python main.py examples/classification_churn/public out/churn --verbose
python bench/check.py examples/classification_churn out/churn
```

`make_real.py` expects this layout under `RAW_DIR`:

| Path | Source |
|---|---|
| `telco/telco.csv` | Telco Customer Churn (IBM sample data) |
| `ai4i/ai4i2020.csv` | AI4I 2020 Predictive Maintenance (UCI) |
| `house/train.csv` | Kaggle House Prices `train.csv` |
| `bike/train.csv` | Kaggle Bike Sharing Demand `train.csv` |
| `mall/mall.csv` | Mall Customer Segmentation |
| `penguins/`, `titanic/`, `calhousing/`, `adult/`, `abalone/`, `glass/`, `winered/`, `pima/`, `ionosphere/`, `seeds/`, `autoins/` | public mirrors of the standard tables |

Iris, wine, breast cancer, digits and diabetes come bundled with scikit-learn.

How the benchmark is built:

- **Splits.** 75/25; labels stay in `private/`, so the solver sees exactly what the grader's solver would.
- **Leaking columns removed.**
  - AI4I failure-mode flags `TWF`, `HDF`, `PWF`, `OSF`, `RNF` encode the target.
  - Bike Sharing `casual` and `registered` sum to the target.
- **Mall labels.** Mall Customers has no labels. Its reference labels are KMeans(5) on standardised income × spending score, which is the segmentation the brief describes.

---

## 6. Assumptions

1. **Contract.** The input/output contract in §1, since the official one was not available to us.
2. **Default metrics.**
   - Classification: macro-F1, which is robust to imbalance. The decision rule is re-tuned for whatever `metric` is declared.
   - Regression: RMSE.
   - Clustering: ARI against hidden labels. "Evaluation labels remain private" implies external labels rather than an internal index such as silhouette.
3. **Positive class.** For binary `f1`, it is `positive_label` if given, else the minority class.
4. **Clustering rows.** If `test.csv` exists, the train and test rows are clustered together and labels are returned for the test rows.
5. **Data size.** Datasets are assumed to fit in memory.
   - Training rows above 150k are subsampled.
   - Clustering mixtures are fitted on at most 20k rows, then predict all rows.
6. **Assessment environment.** 2 or more CPU cores; we developed and measured on 2. Python 3.13 with the pinned versions in `requirements.txt`; no network access needed at run time.
7. **Same feature space.** Test rows are assumed to come from the same feature space as train. Columns missing from `test.csv` are filled with NaN, extra columns are ignored, and unseen categories map to "missing" or "rare".
8. **Infinite values.** Treated as missing.

---

## 7. Repository layout

```
main.py                 entry point: watchdog + stdlib fallback
requirements.txt        pinned direct dependencies (requirements-lock.txt: full environment)
autotab/
  worker.py             pipeline orchestration (child process)
  dataio.py             input contract, lenient metadata, atomic output writers
  analysis.py           type inference; id / constant / duplicate detection (profile definitions)
  profile.py            profile.json
  features.py           tree and linear views, datetime expansion, encodings
  supervised.py         CV engine, fold bagging, time guards, blending, writes
  models.py             candidate portfolio
  ensemble.py           Caruana selection, threshold / class-weight tuning
  metrics.py            metric registry (fast confusion-matrix metrics)
  clustering.py         feature screening, candidates, label-free selection
  budget.py             wall-clock budget from process start
  resources.py          cgroup-aware CPU count
  jsonutil.py           strict JSON, atomic writes
bench/
  make_synthetic.py     synthetic tables with injected issues + ground truth
  make_real.py          real tables (Kaggle examples and others) in contract format
  baseline.py           reference baseline (impute/scale/one-hot + LR/Ridge/KMeans)
  run.py                runs a solver over datasets, scores, times, compares
  check.py              contract validation, scoring, profile checks vs an independent oracle
examples/               three small datasets (one per task) in the contract format, with labels
tests/                  pytest suite: contract, profile definitions, edge cases, metrics
.github/workflows/ci.yml   tests + smoke benchmark on every push
```

---

## 8. Limitations

- **Contract.** The input/output contract is our assumption (§1). A different official contract only requires changes in `autotab/dataio.py` and the watchdog's output check in `main.py`.
- **Non-convex clusters.** Clustering assumes roughly convex or elliptical groups (KMeans and Gaussian mixtures). Rings or moons would need spectral or density-based methods, which we left out: they have no reliable label-free way to choose their parameters inside the time budget.
- **Text columns.** Reduced to length and word counts, with no text features.
- **No deep learning.** Importing PyTorch alone costs seconds, and on tables of this size on CPU it rarely beats boosted trees within 60 s.
- **Fixed hyperparameters.** The portfolio was tuned offline on our benchmark. A different family of hidden tables could favour other settings.
- **CPU speed.** Wall times were measured on 2 cores. The probe and fold adaptation measure the actual machine, so a slower grader reduces work rather than overrunning.
