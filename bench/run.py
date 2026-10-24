"""Run a solver over benchmark datasets and report quality, profile accuracy and time.

    python bench/run.py --data BENCH_DIR [BENCH_DIR ...] --out RESULTS.json
                        [--solver main|baseline] [--filter SUBSTR] [--time-limit 60]
                        [--compare OTHER_RESULTS.json]

Each dataset is solved in a fresh process, exactly like the grader would, with
an external hard timeout of time-limit + 10 s. Results are written as JSON and
summarised as a table.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from check import check_profile, score, validate_outputs  # noqa: E402


def find_datasets(roots, flt=None):
    out = []
    for root in roots:
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isdir(os.path.join(d, "public")) and os.path.isdir(os.path.join(d, "private")):
                if not flt or any(f in name for f in flt.split(",")):
                    out.append(d)
    return out


def solver_cmd(kind, pub, out, limit):
    if kind == "baseline":
        return [sys.executable, os.path.join(HERE, "baseline.py"), pub, out]
    return [sys.executable, os.path.join(ROOT, "main.py"), pub, out, "--time-limit", str(limit)]


def run_one(ds_dir, kind, limit, keep_dir=None):
    name = os.path.basename(ds_dir)
    out = os.path.join(keep_dir, name) if keep_dir else tempfile.mkdtemp(prefix="bench_")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    t = time.time()
    try:
        proc = subprocess.run(solver_cmd(kind, os.path.join(ds_dir, "public"), out, limit),
                              capture_output=True, text=True, timeout=limit + 10)
        rc, stderr = proc.returncode, proc.stderr[-2000:]
    except subprocess.TimeoutExpired:
        rc, stderr = "timeout", ""
    wall = time.time() - t
    res = {"dataset": name, "solver": kind, "wall_s": round(wall, 2), "returncode": rc}
    errs, pred = validate_outputs(ds_dir, out)
    res["contract_errors"] = errs
    if pred is not None and not errs:
        res.update(score(ds_dir, pred))
    else:
        truth = json.load(open(os.path.join(ds_dir, "private", "truth.json")))
        res.update({"task": truth["task"], "quality": 0.0})
    if kind != "baseline":
        res["profile"] = check_profile(ds_dir, out)
        try:
            prof = json.load(open(os.path.join(out, "profile.json")))
            m = prof.get("modelling", {})
            res["ensemble"] = m.get("ensemble", {}).get("weights") or m.get("algorithm")
            res["model_errors"] = len(m.get("errors", []) or [])
            if prof.get("fallback"):
                res["fallback"] = prof["fallback"]
        except (OSError, ValueError):
            pass
    if rc not in (0,) and stderr:
        res["stderr_tail"] = stderr[-500:]
    if not keep_dir:
        shutil.rmtree(out, ignore_errors=True)
    return res


def summarise(results, compare=None):
    by_task = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)
    cmp = {r["dataset"]: r for r in compare} if compare else {}
    lines = []
    hdr = f"{'dataset':34s} {'task':14s} {'metric':18s} {'quality':>8s} {'base':>8s} {'delta':>7s} {'wall_s':>7s} {'profile':>8s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in results:
        b = cmp.get(r["dataset"], {})
        bq = b.get("quality")
        q = r.get("quality", 0.0)
        prof = r.get("profile", {})
        ptxt = f"{prof.get('passed', 0)}/{prof.get('total', 0)}" if prof else ""
        metric = r.get("metric") or ""
        lines.append(f"{r['dataset'][:34]:34s} {r['task']:14s} {metric[:18]:18s} {q:8.4f} "
                     f"{(f'{bq:8.4f}' if bq is not None else '        ')} "
                     f"{(f'{q - bq:+7.4f}' if bq is not None else '       ')} {r['wall_s']:7.2f} {ptxt:>8s}"
                     + ("  !" + ";".join(r["contract_errors"]) if r.get("contract_errors") else ""))
    lines.append("")
    for task, rs in sorted(by_task.items()):
        q = sum(r.get("quality", 0) for r in rs) / len(rs)
        msg = f"{task:14s} n={len(rs):2d}  mean quality={q:.4f}"
        if cmp:
            bs = [cmp[r["dataset"]].get("quality", 0) for r in rs if r["dataset"] in cmp]
            if bs:
                msg += f"  baseline={sum(bs) / len(bs):.4f}"
        walls = [r["wall_s"] for r in rs]
        msg += f"  wall mean={sum(walls) / len(walls):.1f}s max={max(walls):.1f}s"
        lines.append(msg)
    profs = [r["profile"] for r in results if r.get("profile")]
    if profs:
        p = sum(x["passed"] for x in profs)
        t = sum(x["total"] for x in profs)
        lines.append(f"profile checks passed: {p}/{t}")
    fails = [r["dataset"] for r in results if r.get("contract_errors")]
    lines.append(f"contract failures: {len(fails)} {fails if fails else ''}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--solver", default="main", choices=["main", "baseline"])
    ap.add_argument("--filter", default=None)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--keep-outputs", default=None)
    a = ap.parse_args()
    datasets = find_datasets(a.data, a.filter)
    results = []
    for d in datasets:
        r = run_one(d, a.solver, a.time_limit, a.keep_outputs)
        results.append(r)
        print(f"{r['dataset']:34s} q={r.get('quality', 0):.4f} wall={r['wall_s']:.1f}s"
              + (f" errors={r['contract_errors']}" if r.get("contract_errors") else "")
              + (f" profile={r['profile']['passed']}/{r['profile']['total']}" if r.get("profile") else ""),
              flush=True)
    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=1)
    compare = json.load(open(a.compare)) if a.compare and os.path.exists(a.compare) else None
    print()
    print(summarise(results, compare))


if __name__ == "__main__":
    main()
