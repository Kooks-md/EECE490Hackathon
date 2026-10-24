"""Wall-clock budget shared by every stage of a run.

All deadlines are measured from ``t0``, the moment the watchdog process
started, so interpreter start-up and imports count against the limit exactly
as they do for the grader.
"""

from __future__ import annotations

import time


class Budget:
    """Soft deadline for starting model work, hard deadline for finishing writes."""

    def __init__(self, t0: float, time_limit: float = 60.0, work_fraction: float = 0.70):
        self.t0 = float(t0)
        self.time_limit = float(time_limit)
        # Model fitting must stop by soft_end; final ensembling and writing by hard_end.
        self.soft_end = self.t0 + max(4.0, self.time_limit * work_fraction)
        self.hard_end = self.t0 + max(5.0, self.time_limit - max(3.0, 0.08 * self.time_limit))

    def elapsed(self) -> float:
        return time.time() - self.t0

    def remaining(self) -> float:
        """Seconds left for model work (may be negative)."""
        return self.soft_end - time.time()

    def remaining_hard(self) -> float:
        return self.hard_end - time.time()

    def can_afford(self, seconds: float, factor: float = 1.3) -> bool:
        return self.remaining() > seconds * factor

    def expired(self) -> bool:
        return self.remaining() <= 0
