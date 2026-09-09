"""Field solvers.

A solver maps (scene, sample points) -> named field arrays.  The scene owns a
list of them and merges their output, so adding magnetostatics or radiation
means writing one more class with the same two methods -- nothing else in the
app needs to know.

Naming convention for the returned dict:
    "E" : (N, 3) electric field      [V/m]
    "V" : (N,)   electric potential  [V]
    "B" : (N, 3) magnetic field      [T]     (future)
    "A" : (N, 3) vector potential    [T m]   (future)
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..units import K_COULOMB

FieldSample = dict[str, np.ndarray]


class Solver(Protocol):
    """Contributes one or more named fields at arbitrary sample points."""

    name: str
    provides: tuple[str, ...]

    def evaluate(self, scene, points: np.ndarray,
                 exclude_uid: int | None = None) -> FieldSample:
        ...


class CoulombSolver:
    """Electrostatic field and potential of a set of finite-radius charges."""

    name = "coulomb"
    provides = ("E", "V")

    def evaluate(self, scene, points: np.ndarray,
                 exclude_uid: int | None = None) -> FieldSample:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        n = len(points)
        E = np.zeros((n, 3))
        V = np.zeros(n)

        charges = [c for c in scene.charges if c.uid != exclude_uid]
        if not charges:
            return {"E": E, "V": V}

        pos = np.array([c.position for c in charges])          # (M, 3)
        q = np.array([c.q for c in charges])                   # (M,)
        a = np.array([max(c.radius, 1e-9) for c in charges])   # (M,)

        # r[i, j] = points[i] - pos[j]
        r = points[:, None, :] - pos[None, :, :]               # (N, M, 3)
        d = np.linalg.norm(r, axis=2)                          # (N, M)

        inside = d < a[None, :]
        d_safe = np.where(d > 0.0, d, 1.0)

        # Outside:  E = k q / d^2,   V = k q / d
        e_mag = K_COULOMB * q[None, :] / d_safe**2
        v_val = K_COULOMB * q[None, :] / d_safe
        # Inside a uniformly charged ball:
        #   E = k q d / a^3,  V = k q (3 a^2 - d^2) / (2 a^3)
        a3 = a[None, :] ** 3
        e_mag = np.where(inside, K_COULOMB * q[None, :] * d / a3, e_mag)
        v_val = np.where(
            inside,
            K_COULOMB * q[None, :] * (3.0 * a[None, :] ** 2 - d**2) / (2.0 * a3),
            v_val,
        )

        rhat = r / d_safe[:, :, None]
        rhat[d == 0.0] = 0.0

        E = np.einsum("nm,nmk->nk", e_mag, rhat)
        V = v_val.sum(axis=1)
        return {"E": E, "V": V}
