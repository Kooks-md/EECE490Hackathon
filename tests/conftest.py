import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def write_ds(path, train, test=None, **meta):
    os.makedirs(path, exist_ok=True)
    train.to_csv(os.path.join(path, "train.csv"), index=False)
    if test is not None:
        test.to_csv(os.path.join(path, "test.csv"), index=False)
    if meta:
        with open(os.path.join(path, "metadata.json"), "w") as fh:
            json.dump(meta, fh)
    return path


def run_main(data_dir, out_dir, time_limit=60, extra=()):
    cmd = [sys.executable, os.path.join(ROOT, "main.py"), str(data_dir), str(out_dir),
           "--time-limit", str(time_limit), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit + 30)
    assert proc.returncode == 0, proc.stderr
    pred = pd.read_csv(os.path.join(out_dir, "predictions.csv"), keep_default_na=False)
    with open(os.path.join(out_dir, "profile.json")) as fh:
        profile = json.load(fh)
    return pred, profile


@pytest.fixture
def rng():
    return np.random.default_rng(0)
