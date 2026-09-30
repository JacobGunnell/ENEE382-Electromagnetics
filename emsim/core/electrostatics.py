"""Electrostatic equilibrium of a set of conductors (boundary element method).

A conductor in equilibrium satisfies three conditions, all of which fall out
of one linear system rather than being imposed by hand:

* its surface is an equipotential,
* the field inside it vanishes,
* its charge lives on the surface, piling up wherever the surface curves
  sharply -- the edge effect.

With one unknown charge per surface site and one unknown potential per
conductor, the system is the saddle-point problem

    [ P   -B ] [q]   [ -V_ext ]
    [ B^T  0 ] [V] = [   Q    ]

where ``P`` is the potential-coefficient matrix over sites, ``B`` marks which
conductor owns each site, ``V_ext`` is the potential from everything that is
*not* a conductor (free charges and insulators), and ``Q`` is each conductor's
total charge.  The first block row says every site of a conductor sits at that
conductor's potential; the second says charge is conserved on each body.

Inverting once buys more than a solve.  Writing the inverse in blocks,

    q = A_qv (-V_ext) + A_qQ Q
    V = A_Vv (-V_ext) + A_VQ Q,

the block ``A_VQ`` *is* the conductor potential-coefficient matrix: the
potentials produced by unit charges, i.e. exactly what capacitance is defined
from.  So the capacitance matrix costs nothing beyond the factorisation, and
re-solving while a point charge is dragged costs one matrix-vector product.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..units import K_COULOMB
from .bodies import Body, Sites
from .kernels import potential_matrix

#: Refuse to build a denser system than this; the inverse is O(N^3).
MAX_SITES = 2400


def self_potential(sites: Sites, dim: int, wire_radius: float) -> np.ndarray:
    """Diagonal of ``P``: a site's own potential at its collocation point.

    ``dim`` 2 treats the site as a disc of equal area evaluated at its centre,
    ``V = q / (2 pi eps0 a)``.  ``dim`` 1 treats it as a segment of wire
    evaluated on the wire surface, ``V = 2 k q asinh(L / 2 r_w) / L`` -- which,
    unlike the usual ``log(L / r_w)`` form, stays valid when the segments get
    shorter than the wire is thick.
    """
    m = np.maximum(sites.measure, 1e-300)
    if dim == 2:
        a = np.sqrt(m / np.pi)
        return 2.0 * K_COULOMB / a
    r_w = max(wire_radius, 1e-9)
    return (2.0 * K_COULOMB / m) * np.arcsinh(m / (2.0 * r_w))


@dataclass
class ConductorSystem:
    """A factorised set of conductors, reusable until the geometry changes."""

    uids: list[int] = field(default_factory=list)
    pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    radius: np.ndarray = field(default_factory=lambda: np.zeros(0))
    owner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    #: Blocks of the inverse of the augmented matrix.
    a_qv: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    a_qQ: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    a_Vv: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    #: Conductor potential coefficients: ``V_i = sum_j p_cond[i, j] Q_j``.
    p_cond: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    truncated: bool = False

    @property
    def n_sites(self) -> int:
        return len(self.pos)

    @property
    def n_conductors(self) -> int:
        return len(self.uids)

    def index_of(self, uid: int) -> int | None:
        return self.uids.index(uid) if uid in self.uids else None

    # ------------------------------------------------------------------
    def solve(self, charges: np.ndarray, v_ext: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
        """Site charges and conductor potentials for the given totals.

        ``v_ext`` is the potential at each site from every non-conductor
        source.  Passing ``None`` means an isolated set of conductors.
        """
        Q = np.asarray(charges, dtype=float).reshape(self.n_conductors)
        if self.n_sites == 0:
            return np.zeros(0), np.zeros(self.n_conductors)
        q = self.a_qQ @ Q
        V = self.p_cond @ Q
        if v_ext is not None and len(v_ext):
            rhs = -np.asarray(v_ext, dtype=float)
            q = q + self.a_qv @ rhs
            V = V + self.a_Vv @ rhs
        return q, V

    def capacitance(self, uid_a: int, uid_b: int) -> float | None:
        """Two-conductor capacitance ``C = 1 / (P_aa + P_bb - 2 P_ab)``.

        This is the geometric capacitance -- what you would measure by putting
        +Q on one and -Q on the other -- so it does not depend on whatever
        charge happens to be on them now.  Any other conductors present are
        included as floating, uncharged bodies, which is what they physically
        are.
        """
        i, j = self.index_of(uid_a), self.index_of(uid_b)
        if i is None or j is None or i == j:
            return None
        denom = self.p_cond[i, i] + self.p_cond[j, j] - 2.0 * self.p_cond[i, j]
        return 1.0 / denom if denom > 0 else None


def build_system(conductors: list[Body],
                 site_scale: float = 1.0) -> ConductorSystem:
    """Discretise, assemble and invert.  Cost is O(N^3) in the site count."""
    sys = ConductorSystem()
    if not conductors:
        return sys

    # Share the site budget out by surface area so a large plate is not
    # resolved more coarsely than a small one next to it.
    requested = [max(int(b.n_sites * site_scale), 8) for b in conductors]
    total = sum(requested)
    if total > MAX_SITES:
        scale = MAX_SITES / total
        requested = [max(8, int(r * scale)) for r in requested]
        sys.truncated = True

    pos, rad, owner, diag = [], [], [], []
    for k, (body, n) in enumerate(zip(conductors, requested)):
        s = body.surface_sites(n)
        dim = 1 if body.kind in ("line", "loop") else 2
        wire = getattr(body, "wire_radius", 1e-3)
        pos.append(s.pos)
        rad.append(s.radius)
        owner.append(np.full(len(s), k))
        diag.append(self_potential(s, dim, wire))
        sys.uids.append(body.uid)

    sys.pos = np.vstack(pos)
    sys.radius = np.concatenate(rad)
    sys.owner = np.concatenate(owner)
    n, m = len(sys.pos), len(sys.uids)

    P = potential_matrix(sys.pos, np.concatenate(diag))
    B = np.zeros((n, m))
    B[np.arange(n), sys.owner] = 1.0

    A = np.zeros((n + m, n + m))
    A[:n, :n] = P
    A[:n, n:] = -B
    A[n:, :n] = B.T
    inv = np.linalg.inv(A)

    sys.a_qv = inv[:n, :n]
    sys.a_qQ = inv[:n, n:]
    sys.a_Vv = inv[n:, :n]
    sys.p_cond = inv[n:, n:]
    # The augmented inverse is symmetric up to round-off; symmetrising keeps
    # capacitance from picking up an asymmetry that is pure numerical noise.
    sys.p_cond = 0.5 * (sys.p_cond + sys.p_cond.T)
    return sys


def geometry_key(bodies: list[Body]) -> tuple:
    """Signature that changes exactly when the conductor system must be rebuilt."""
    parts = []
    for b in bodies:
        if not b.is_conductor:
            continue
        geo = tuple(round(float(getattr(b, name)), 12)
                    for name in ("length", "radius", "width", "height",
                                 "wire_radius")
                    if hasattr(b, name))
        parts.append((b.uid, b.kind, b.n_sites, geo,
                      tuple(np.round(b.position, 12)),
                      tuple(np.round(b.axis, 12))))
    return tuple(parts)
