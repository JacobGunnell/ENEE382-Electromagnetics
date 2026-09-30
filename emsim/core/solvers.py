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

from .kernels import coulomb_sum

FieldSample = dict[str, np.ndarray]


class Solver(Protocol):
    """Contributes one or more named fields at arbitrary sample points."""

    name: str
    provides: tuple[str, ...]

    def evaluate(self, scene, points: np.ndarray,
                 exclude_uid: int | None = None) -> FieldSample:
        ...


class CoulombSolver:
    """Electrostatic field and potential of every charge element in the scene.

    The scene has already solved for conductor equilibrium and flattened
    everything -- free charges, insulator sites, conductor surface sites --
    into one list, so this is a plain superposition with no special cases and
    no idealised geometry.
    """

    name = "coulomb"
    provides = ("E", "V")

    def evaluate(self, scene, points: np.ndarray,
                 exclude_uid: int | None = None) -> FieldSample:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        pos, q, rad = scene.state().exclude(exclude_uid)
        E, V = coulomb_sum(pos, q, rad, points)
        return {"E": E, "V": V}
