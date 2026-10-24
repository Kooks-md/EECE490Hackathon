"""autotab: a time-budgeted AutoML engine for unfamiliar tabular datasets.

One program handles classification, regression and clustering. Entry points:

    python main.py DATASET_DIR OUTPUT_DIR [--time-limit 60]
    python -m autotab DATASET_DIR OUTPUT_DIR
    autotab.solve(DATASET_DIR, OUTPUT_DIR)

All three run the pipeline (autotab.worker) under the watchdog in main.py, so
a valid output always exists before the time limit.
"""

__version__ = "1.0.0"


def solve(data_dir: str, out_dir: str, time_limit: float = 60.0, seed: int = 42) -> int:
    """Run the full pipeline on one dataset directory; returns the process exit code (0)."""
    import os
    import subprocess
    import sys

    main = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
    cmd = [sys.executable, main, str(data_dir), str(out_dir), "--time-limit", str(time_limit), "--seed", str(seed)]
    return subprocess.run(cmd, check=False).returncode
