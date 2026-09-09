"""Simulation entities.

All quantities are SI.  ``velocity`` is carried on the charge even though the
electrostatic solver ignores it: the retarded-potential (Lienard-Wiechert)
solver needed for radiation reads it directly, so entities do not have to
change when that lands.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

_uid_counter = itertools.count(1)


@dataclass
class PointCharge:
    """A charge modelled as a uniformly-charged ball of radius ``radius``.

    Outside ``radius`` this is exactly a point charge; inside, the field and
    potential follow the solid-sphere solution.  That removes the 1/r^2
    singularity without introducing unphysical behaviour anywhere a user can
    actually look.
    """

    q: float                                   # coulombs
    position: np.ndarray                       # metres, shape (3,)
    velocity: np.ndarray = field(              # m/s, unused by electrostatics
        default_factory=lambda: np.zeros(3))
    mass: float = 1e-9                         # kg
    radius: float = 0.03                       # m, softening / render radius
    label: str = ""
    uid: int = field(default_factory=lambda: next(_uid_counter))

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(3)
        if not self.label:
            self.label = f"q{self.uid}"


#: Reference charge whose ball has radius :data:`BASE_RADIUS`.
REFERENCE_CHARGE = 10e-9      # C
BASE_RADIUS = 0.030           # m


def radius_for_charge(q: float, domain: float = 1.0) -> float:
    """Ball radius for a charge of magnitude ``q``.

    Volume scales with |q| (constant charge density) so the radius goes as
    |q|^(1/3), clamped so extreme magnitudes stay both visible and small
    compared with the region of interest.
    """
    base = BASE_RADIUS * domain
    if q == 0.0:
        return 0.6 * base
    r = base * (abs(q) / REFERENCE_CHARGE) ** (1.0 / 3.0)
    return float(np.clip(r, 0.35 * base, 2.6 * base))
