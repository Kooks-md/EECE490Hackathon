"""CPU detection that respects container limits.

``os.cpu_count()`` reports the host's cores inside a container. If the grader
runs us with ``--cpus=2`` on a 32-core host, ``n_jobs=-1`` would start 32
threads on 2 cores and multiply runtime. We take the minimum of the affinity
mask and the cgroup CPU quota instead. Standard library only, because the
watchdog in ``main.py`` imports this before numpy is loaded.
"""

from __future__ import annotations

import os

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
)


def _cgroup_quota() -> float | None:
    """Return the CPU quota in cores, or None when unlimited / unknown."""
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max", encoding="ascii") as fh:
            quota, period = fh.read().split()[:2]
        if quota != "max" and float(period) > 0:
            return float(quota) / float(period)
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="ascii") as fh:
            quota_v1 = float(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="ascii") as fh:
            period_v1 = float(fh.read().strip())
        if quota_v1 > 0 and period_v1 > 0:
            return quota_v1 / period_v1
    except (OSError, ValueError):
        pass
    return None


def effective_cpu_count() -> int:
    """Number of CPUs this process can actually use (at least 1)."""
    n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    quota = _cgroup_quota()
    if quota is not None:
        n = min(n, max(1, int(round(quota))))
    return max(1, n)


def thread_env(n_threads: int, base: dict | None = None) -> dict:
    """Environment with thread-pool variables pinned, unless already set by the user."""
    env = dict(os.environ if base is None else base)
    for var in THREAD_ENV_VARS:
        env.setdefault(var, str(n_threads))
    return env
