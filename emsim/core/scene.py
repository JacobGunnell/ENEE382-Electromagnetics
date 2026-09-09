"""The simulation scene: entities + solvers + change notification.

Deliberately free of Qt so the model stays usable headless (batch figures,
tests, notebooks).  Observers are plain callables.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .entities import PointCharge
from .solvers import CoulombSolver, FieldSample, Solver


class Scene:
    def __init__(self) -> None:
        self.charges: list[PointCharge] = []
        self.solvers: list[Solver] = [CoulombSolver()]
        self.time: float = 0.0            # for future time-dependent solvers
        self._observers: list[Callable[[str], None]] = []

    # -- observation -------------------------------------------------------
    def subscribe(self, cb: Callable[[str], None]) -> None:
        self._observers.append(cb)

    def notify(self, reason: str = "changed") -> None:
        for cb in list(self._observers):
            cb(reason)

    # -- entity management -------------------------------------------------
    def add_charge(self, charge: PointCharge) -> PointCharge:
        self.charges.append(charge)
        self.notify("charges")
        return charge

    def remove_charge(self, uid: int) -> None:
        self.charges = [c for c in self.charges if c.uid != uid]
        self.notify("charges")

    def clear(self) -> None:
        self.charges.clear()
        self.notify("charges")

    def by_uid(self, uid: int) -> PointCharge | None:
        for c in self.charges:
            if c.uid == uid:
                return c
        return None

    # -- field evaluation --------------------------------------------------
    def evaluate(self, points: np.ndarray,
                 exclude_uid: int | None = None) -> FieldSample:
        """Merge every solver's contribution at ``points``."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        out: FieldSample = {}
        for solver in self.solvers:
            for key, val in solver.evaluate(self, points, exclude_uid).items():
                out[key] = val if key not in out else out[key] + val
        return out

    def positions(self) -> np.ndarray:
        if not self.charges:
            return np.zeros((0, 3))
        return np.array([c.position for c in self.charges])

    def forces(self) -> np.ndarray:
        """Force on each charge from every *other* charge, shape (N, 3) [N]."""
        n = len(self.charges)
        if n == 0:
            return np.zeros((0, 3))
        F = np.zeros((n, 3))
        for i, c in enumerate(self.charges):
            sample = self.evaluate(c.position[None, :], exclude_uid=c.uid)
            F[i] = c.q * sample["E"][0]
        return F
