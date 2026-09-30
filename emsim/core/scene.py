"""The simulation scene: entities, bodies, solvers and change notification.

Deliberately free of Qt so the model stays usable headless (batch figures,
tests, notebooks).  Observers are plain callables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .bodies import Body, Material
from .electrostatics import ConductorSystem, build_system, geometry_key
from .entities import PointCharge
from .kernels import coulomb_sum
from .solvers import CoulombSolver, FieldSample, Solver


@dataclass
class SolvedState:
    """Everything the renderers and field evaluators need, solved once."""

    pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    radius: np.ndarray = field(default_factory=lambda: np.zeros(0))
    owner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    #: uid -> (positions, charges) of that body's rendered/relaxing sites.
    body_sites: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    #: uid -> conductor potential [V].
    potentials: dict[int, float] = field(default_factory=dict)
    system: ConductorSystem = field(default_factory=ConductorSystem)

    def exclude(self, uid: int | None):
        if uid is None:
            return self.pos, self.q, self.radius
        keep = self.owner != uid
        return self.pos[keep], self.q[keep], self.radius[keep]


class Scene:
    def __init__(self) -> None:
        self.charges: list[PointCharge] = []
        self.bodies: list[Body] = []
        self.solvers: list[Solver] = [CoulombSolver()]
        self.time: float = 0.0            # for future time-dependent solvers
        self._observers: list[Callable[[str], None]] = []
        #: Scaled below 1 while a body is being dragged: rebuilding the
        #: factorisation is O(N^3), so coarsening it keeps the drag fluid.
        self.site_scale = 1.0
        self._system = ConductorSystem()
        self._system_key: tuple | None = None
        self._state: SolvedState | None = None

    # -- observation -------------------------------------------------------
    def subscribe(self, cb: Callable[[str], None]) -> None:
        self._observers.append(cb)

    def notify(self, reason: str = "changed") -> None:
        self.invalidate()
        for cb in list(self._observers):
            cb(reason)

    def invalidate(self) -> None:
        """Drop the memoised solve.  Cheap: the factorisation is kept."""
        self._state = None

    # -- entity management -------------------------------------------------
    def add_charge(self, charge: PointCharge) -> PointCharge:
        self.charges.append(charge)
        self.notify("charges")
        return charge

    def remove_charge(self, uid: int) -> None:
        self.charges = [c for c in self.charges if c.uid != uid]
        self.notify("charges")

    def add_body(self, body: Body) -> Body:
        self.bodies.append(body)
        self.notify("bodies")
        return body

    def remove_body(self, uid: int) -> None:
        self.bodies = [b for b in self.bodies if b.uid != uid]
        self.notify("bodies")

    def clear(self) -> None:
        self.charges.clear()
        self.bodies.clear()
        self.notify("cleared")

    def objects(self) -> list:
        """Point charges and bodies as one list, in creation order.

        The UI shows a single object list, so the model offers one too rather
        than making every caller stitch the two together.
        """
        return sorted(self.charges + self.bodies, key=lambda o: o.seq)

    def by_uid(self, uid: int):
        for c in self.charges:
            if c.uid == uid:
                return c
        for b in self.bodies:
            if b.uid == uid:
                return b
        return None

    def body(self, uid: int) -> Body | None:
        for b in self.bodies:
            if b.uid == uid:
                return b
        return None

    def conductors(self) -> list[Body]:
        return [b for b in self.bodies if b.is_conductor]

    def deposit_charge(self, uid: int, dq: float) -> None:
        b = self.body(uid)
        if b is None:
            return
        b.charge += dq
        if b.is_conductor:
            b.relax_t = 0.0          # replay the migration to the surface
        self.notify("charge-deposited")

    # -- the solve ---------------------------------------------------------
    def state(self) -> SolvedState:
        """Solve for equilibrium and assemble every charge element.

        Memoised until :meth:`invalidate`.  The O(N^3) conductor factorisation
        is reused for as long as the conductor geometry is unchanged, so a
        point charge being dragged past a plate costs one matrix-vector
        product per frame, not a re-inversion.
        """
        if self._state is not None:
            return self._state

        st = SolvedState()
        pos, q, rad, owner = [], [], [], []

        # 1. Fixed sources: free point charges and insulators.
        for c in self.charges:
            pos.append(c.position[None, :])
            q.append(np.array([c.q]))
            rad.append(np.array([max(c.radius, 1e-9)]))
            owner.append(np.array([c.uid]))
        for b in self.bodies:
            if b.is_conductor:
                continue
            s = b.sites()
            n = max(len(s), 1)
            qi = np.full(len(s), b.charge / n)
            pos.append(s.pos)
            q.append(qi)
            rad.append(s.radius)
            owner.append(np.full(len(s), b.uid))
            st.body_sites[b.uid] = (s.pos, qi)

        fixed_pos = np.vstack(pos) if pos else np.zeros((0, 3))
        fixed_q = np.concatenate(q) if q else np.zeros(0)
        fixed_rad = np.concatenate(rad) if rad else np.zeros(0)

        # 2. Conductors: rebuild the factorisation only if geometry moved.
        conductors = self.conductors()
        key = (geometry_key(self.bodies), round(self.site_scale, 4))
        if conductors and key != self._system_key:
            self._system = build_system(conductors, self.site_scale)
            self._system_key = key
        elif not conductors:
            self._system, self._system_key = ConductorSystem(), None
        sysm = self._system
        st.system = sysm

        if sysm.n_sites:
            _, v_ext = coulomb_sum(fixed_pos, fixed_q, fixed_rad, sysm.pos,
                                   want_E=False)
            Q = np.array([self.body(u).charge for u in sysm.uids])
            q_sites, V = sysm.solve(Q, v_ext)
            st.potentials = {u: float(v) for u, v in zip(sysm.uids, V)}

            for k, uid in enumerate(sysm.uids):
                body = self.body(uid)
                sel = sysm.owner == k
                p_eq, q_eq = sysm.pos[sel], q_sites[sel]
                r = sysm.radius[sel]
                t = float(np.clip(getattr(body, "relax_t", 1.0), 0.0, 1.0))
                if t < 1.0:
                    p_eq, q_eq = self._relaxing(body, p_eq, q_eq, t)
                pos.append(p_eq)
                q.append(q_eq)
                rad.append(r)
                owner.append(np.full(len(p_eq), uid))
                st.body_sites[uid] = (p_eq, q_eq)

        st.pos = np.vstack(pos) if pos else np.zeros((0, 3))
        st.q = np.concatenate(q) if q else np.zeros(0)
        st.radius = np.concatenate(rad) if rad else np.zeros(0)
        st.owner = np.concatenate(owner).astype(int) if owner else np.zeros(0, int)
        self._state = st
        return st

    @staticmethod
    def _relaxing(body: Body, p_eq: np.ndarray, q_eq: np.ndarray, t: float):
        """Interpolate between 'just deposited' and equilibrium.

        Purely a visual relaxation, not time-accurate dynamics: the starting
        state is charge spread uniformly through the body, the end state is
        the solved equilibrium, and ``t`` sweeps between them.  For a solid
        body that shows charge streaming radially out to the surface; for a
        plate it shows it piling up along the edges.
        """
        start = body.interior_sites(len(p_eq))
        p0 = start.pos if len(start.pos) == len(p_eq) else p_eq
        q0 = np.full(len(q_eq), body.charge / max(len(q_eq), 1))
        # Ease out, so the rush away from the centre is the visible part.
        e = 1.0 - (1.0 - t) ** 3
        return p0 + (p_eq - p0) * e, q0 + (q_eq - q0) * e

    # -- field evaluation --------------------------------------------------
    def evaluate(self, points: np.ndarray,
                 exclude_uid: int | None = None) -> FieldSample:
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
        """Force on each point charge from every *other* source, ``(N, 3)`` [N]."""
        n = len(self.charges)
        if n == 0:
            return np.zeros((0, 3))
        F = np.zeros((n, 3))
        for i, c in enumerate(self.charges):
            F[i] = c.q * self.evaluate(c.position[None, :],
                                       exclude_uid=c.uid)["E"][0]
        return F

    def body_forces(self) -> dict[int, np.ndarray]:
        """Net force on each charged body, summed over its own sites."""
        st = self.state()
        out: dict[int, np.ndarray] = {}
        for b in self.bodies:
            sites = st.body_sites.get(b.uid)
            if sites is None or not len(sites[0]):
                continue
            p, qi = sites
            src = st.exclude(b.uid)
            E, _ = coulomb_sum(*src, p, want_V=False)
            out[b.uid] = (qi[:, None] * E).sum(axis=0)
        return out

    # -- measurement -------------------------------------------------------
    def measure(self, uid_a: int, uid_b: int) -> dict | None:
        """Potential difference and capacitance between two conductors."""
        st = self.state()
        a, b = self.body(uid_a), self.body(uid_b)
        if uid_a == uid_b or a is None or b is None:
            return None
        if not (a.is_conductor and b.is_conductor):
            return None
        if uid_a not in st.potentials or uid_b not in st.potentials:
            return None
        va, vb = st.potentials[uid_a], st.potentials[uid_b]
        dv = va - vb
        cap = st.system.capacitance(uid_a, uid_b)
        # The charge "on the capacitor": exact when the two carry +-Q, which
        # is the configuration capacitance is defined for.
        q_cap = 0.5 * (a.charge - b.charge)
        return {
            "uid_a": uid_a, "uid_b": uid_b,
            "V_a": va, "V_b": vb, "dV": dv,
            "Q_a": a.charge, "Q_b": b.charge, "Q_cap": q_cap,
            "C": cap,
            "C_measured": (q_cap / dv) if dv != 0.0 else None,
            "balanced": abs(a.charge + b.charge) <= 1e-12 * max(
                abs(a.charge), abs(b.charge), 1e-30),
            "energy": 0.5 * cap * dv * dv if cap else None,
        }
