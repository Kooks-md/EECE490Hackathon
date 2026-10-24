#!/usr/bin/env python3
"""Unknown Dataset Hackathon - standard entry point.

    python main.py DATASET_DIR OUTPUT_DIR [--time-limit 60] [--seed 42] [--verbose]

Runs the pipeline (autotab.worker) in a child process under a watchdog:

* the child writes a baseline prediction within seconds and keeps overwriting
  it (atomically) with better ones;
* if the child crashes, hangs inside native code, or runs out of time, the
  watchdog kills it and keeps the last valid output;
* if no valid output exists at all, a standard-library fallback writes one.

The watchdog itself imports only the standard library, so it starts in
milliseconds and cannot be taken down by a broken numerical stack.
"""

import time

T0 = time.time()

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from autotab.resources import effective_cpu_count, thread_env  # noqa: E402

TRAIN_FILES = ("train.csv", "training.csv", "data.csv", "dataset.csv")
TEST_FILES = ("test.csv", "testing.csv", "eval.csv", "evaluation.csv", "predict.csv")
META_FILES = ("metadata.json", "meta.json", "spec.json", "dataset.json", "info.json", "task.json")


# --------------------------------------------------------------------------- helpers (stdlib only)
def _find(d, names):
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    return None


def _read_meta(d):
    p = _find(d, META_FILES)
    if not p:
        return {}
    try:
        with open(p, encoding="utf-8-sig") as fh:
            m = json.load(fh)
        return m if isinstance(m, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_csv_rows(path):
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        sample = fh.readline()
        fh.seek(0)
        counts = {s: sample.count(s) for s in (",", ";", "\t", "|")}
        sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ","
        reader = csv.reader(fh, delimiter=sep)
        header = [h.strip() for h in next(reader, [])]
        rows = [r for r in reader if r]
    return header, rows


def _task_from_meta(meta):
    t = str(meta.get("task") or meta.get("task_type") or meta.get("problem_type") or "").lower()
    if "regress" in t:
        return "regression"
    if "cluster" in t:
        return "clustering"
    if "class" in t:
        return "classification"
    return None


def _prediction_rows(data_dir, meta):
    """(header, rows) of the table we must predict, as the worker defines it."""
    test = _find(data_dir, TEST_FILES)
    train = _find(data_dir, TRAIN_FILES)
    if test:
        return _read_csv_rows(test)
    if train:
        return _read_csv_rows(train)
    return [], []


def outputs_valid(data_dir, out_dir):
    pred = os.path.join(out_dir, "predictions.csv")
    prof = os.path.join(out_dir, "profile.json")
    try:
        with open(prof, encoding="utf-8") as fh:
            json.load(fh)
        header, rows = _read_csv_rows(pred)
        if header[:2] != ["id", "prediction"]:
            return False
        _, expected = _prediction_rows(data_dir, _read_meta(data_dir))
        return len(rows) == len(expected) and all(len(r) >= 2 and r[1] != "" for r in rows)
    except (OSError, ValueError, StopIteration):
        return False


def emergency_fallback(data_dir, out_dir, reason):
    """Last resort: majority class / mean / single cluster, standard library only."""
    os.makedirs(out_dir, exist_ok=True)
    meta = _read_meta(data_dir)
    task = _task_from_meta(meta) or "classification"
    target = meta.get("target") or meta.get("target_column") or meta.get("label")
    id_col = meta.get("id_column") or meta.get("id")
    p_header, p_rows = _prediction_rows(data_dir, meta)
    id_idx = p_header.index(id_col) if id_col in p_header else None
    ids = [r[id_idx] if id_idx is not None and id_idx < len(r) else str(i) for i, r in enumerate(p_rows)]
    value = "0"
    train = _find(data_dir, TRAIN_FILES)
    if task != "clustering" and train:
        header, rows = _read_csv_rows(train)
        if target not in header and header:
            target = header[-1]
        if target in header:
            j = header.index(target)
            ys = [r[j].strip() for r in rows if j < len(r) and r[j].strip() not in ("", "NA", "nan", "?")]
            if ys:
                if task == "regression":
                    nums = []
                    for v in ys:
                        try:
                            nums.append(float(v))
                        except ValueError:
                            pass
                    value = repr(statistics.fmean(nums)) if nums else "0"
                else:
                    value = Counter(ys).most_common(1)[0][0]
    with open(os.path.join(out_dir, "predictions.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "prediction"])
        for i in ids:
            w.writerow([i, value])
    prof_path = os.path.join(out_dir, "profile.json")
    profile = None
    try:
        with open(prof_path, encoding="utf-8") as fh:
            profile = json.load(fh)
    except (OSError, ValueError):
        profile = {"schema_version": "1.0", "dataset": {"task": task, "n_rows_predict": len(ids)}}
    profile["fallback"] = {"used": True, "reason": reason}
    with open(prof_path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Unknown Dataset Hackathon solver")
    ap.add_argument("data_dir", help="dataset directory (metadata.json, train.csv, test.csv)")
    ap.add_argument("out_dir", help="where predictions.csv and profile.json are written")
    ap.add_argument("--time-limit", type=float, default=60.0, help="wall-clock limit per dataset (s)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    n_threads = effective_cpu_count()
    env = thread_env(n_threads)
    env["PYTHONWARNINGS"] = "ignore"
    cmd = [sys.executable, "-m", "autotab.worker", a.data_dir, a.out_dir, "--t0", repr(T0),
           "--time-limit", str(a.time_limit), "--seed", str(a.seed), "--threads", str(n_threads)]
    if a.verbose:
        cmd.append("--verbose")
    kill_at = T0 + a.time_limit - max(1.5, 0.025 * a.time_limit)
    reason = None
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env)
    try:
        rc = proc.wait(timeout=max(0.5, kill_at - time.time()))
        if rc != 0:
            reason = f"worker exited with code {rc}"
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        reason = "worker killed at the time limit"
    if reason and a.verbose:
        print(f"[watchdog] {reason}", file=sys.stderr)
    if not outputs_valid(a.data_dir, a.out_dir):
        emergency_fallback(a.data_dir, a.out_dir, reason or "worker produced invalid output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
