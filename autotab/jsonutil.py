"""Strict-JSON serialisation helpers.

numpy integers and booleans are not JSON serialisable, and Python writes NaN as
the bare token ``NaN``, which strict parsers reject. Everything that leaves the
program goes through :func:`to_jsonable` first.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os

import numpy as np


def to_jsonable(obj):
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):  # pandas Timestamp
        try:
            return obj.isoformat()
        except Exception:  # noqa: BLE001
            pass
    try:
        if obj != obj:  # NaN-like scalars (pd.NA, NaT)
            return None
    except Exception:  # noqa: BLE001
        pass
    return str(obj)


def dumps(obj, **kwargs) -> str:
    return json.dumps(to_jsonable(obj), allow_nan=False, **kwargs)


def atomic_write_text(path: str, text: str) -> None:
    """Write via a temp file + rename so readers never see a half-written file."""
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    os.replace(tmp, path)
