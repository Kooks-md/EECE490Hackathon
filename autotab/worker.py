"""The actual pipeline, run as a child process of main.py (the watchdog).

    python -m autotab.worker DATASET_DIR OUTPUT_DIR [--t0 T] [--time-limit 60]

Stages: load -> analyse columns -> write profile -> model -> rewrite profile
with the modelling report. Each stage is guarded; predictions are written
early (baseline) and rewritten whenever a better model is available.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings


def run(data_dir: str, out_dir: str, t0: float, time_limit: float, seed: int = 42,
        threads: int | None = None, verbose: bool = False) -> int:
    warnings.filterwarnings("ignore")
    from .analysis import analyze
    from .budget import Budget
    from .dataio import OutputWriter, load_dataset
    from .metrics import resolve_metric
    from .profile import build_profile
    from .resources import effective_cpu_count

    budget = Budget(t0, time_limit)
    n_threads = threads or effective_cpu_count()

    def log(msg: str) -> None:
        if verbose:
            print(f"[{budget.elapsed():5.1f}s] {msg}", file=sys.stderr, flush=True)

    timings: dict[str, float] = {"start_s": round(budget.elapsed(), 3)}
    ds = load_dataset(data_dir)
    writer = OutputWriter(out_dir, ds.pred_ids)
    metric = resolve_metric(ds.task, ds.metric)
    timings["loaded_s"] = round(budget.elapsed(), 3)
    log(f"loaded {ds.name}: task={ds.task} metric={metric} train={ds.train.shape} predict={len(ds.pred_ids)}")

    an = analyze(ds)
    profile = build_profile(ds, an)
    profile["run"] = {"metric": metric, "seed": seed, "threads": n_threads, "time_limit_s": time_limit}
    timings["profiled_s"] = round(budget.elapsed(), 3)
    writer.profile(profile)
    log(f"profiled: {len(an.kept)} usable features")

    report: dict = {}
    try:
        if ds.task == "clustering":
            from .clustering import run_clustering
            report = run_clustering(ds, an, budget, writer, seed, n_threads, log)
        else:
            from .features import build_features
            from .supervised import run_supervised
            fs = build_features(an)
            profile["preprocessing"] = fs.steps
            timings["features_s"] = round(budget.elapsed(), 3)
            report = run_supervised(ds, fs, metric, budget, writer, seed, n_threads, log)
    except Exception as exc:  # noqa: BLE001 - keep whatever predictions were already written
        report = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=5)}
        log(f"modelling failed: {report['error']}")
        if writer.n_writes == 0:
            raise
    timings["finished_s"] = round(budget.elapsed(), 3)
    profile["modelling"] = report
    profile["timings"] = timings
    if "preprocessing" not in profile and isinstance(report, dict) and "preprocessing" in report:
        profile["preprocessing"] = report.pop("preprocessing")
    writer.profile(profile)
    log(f"done in {budget.elapsed():.1f}s")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir")
    p.add_argument("out_dir")
    p.add_argument("--t0", type=float, default=None, help="process start time (epoch seconds)")
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    t0 = a.t0 if a.t0 is not None else time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    return run(a.data_dir, a.out_dir, t0, a.time_limit, a.seed, a.threads, a.verbose)


if __name__ == "__main__":
    sys.exit(main())
