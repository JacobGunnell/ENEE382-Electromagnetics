"""Charged bodies: lines, loops, sheets, disks and spheres.

Every body is discretised into equal-measure *sites*.  Those sites are the
only thing the rest of the program knows about: the boundary-element solver
puts unknown charges on a conductor's sites, an insulator gets a fixed equal
charge on each of its sites, and the field evaluator sums over all of them.

Because fields come from that explicit sum and nothing else, there is no
infinite-sheet or infinite-wire idealisation anywhere -- edge and fringing
effects fall out on their own.

Conventions
-----------
* ``axis`` is the plate/disk/loop normal, or the direction of a line.
* Sites carry a ``measure``: a length for 1-D bodies, an area for 2-D ones,
  and a volume for the interior of an insulating sphere.
* All lengths are metres, charges coulombs.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
_uid_counter = itertools.count(1000)


class Material(str, Enum):
    CONDUCTOR = "conductor"
    INSULATOR = "insulator"


def _ball_radius(volume: float):
    """Radius of a ball of the given volume.

    Using this as the regularisation radius makes the smoothed sources tile
    the body's volume exactly, which is what lets the summed interior field of
    a uniformly charged ball come out at the Gauss-law value instead of a
    fraction of it.
    """
    return np.cbrt(3.0 * volume / (4.0 * math.pi))


def _disc_radius(area: float):
    """Radius of a disc of the given area, for surface patches."""
    return np.sqrt(area / math.pi)


@dataclass
class Sites:
    """A body's discretisation."""

    pos: np.ndarray          # (N, 3) metres
    measure: np.ndarray      # (N,) length / area / volume, equal per site
    normal: np.ndarray       # (N, 3) surface normal, or the body axis for 1-D
    #: Regularisation radius: half the site spacing.  Fields are evaluated
    #: treating each site as a uniformly charged ball of this radius, which
    #: keeps the visualisation finite on top of a body without altering
    #: anything at a distance.
    radius: np.ndarray       # (N,)

    def __len__(self) -> int:
        return len(self.pos)


def orthonormal_frame(axis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed ``(u, v, w)`` with ``w`` along ``axis``.

    The reference vector is chosen so that the overwhelmingly common case --
    a plate whose normal is +z -- comes out as ``(x, y, z)``.  Otherwise a
    sheet's ``width`` would silently land on a different world axis than the
    user expects when they type it in.
    """
    w = np.asarray(axis, dtype=float)
    n = np.linalg.norm(w)
    w = w / n if n > 0 else np.array([0.0, 0.0, 1.0])
    ref = (np.array([0.0, 1.0, 0.0]) if abs(w[1]) < 0.9
           else np.array([0.0, 0.0, 1.0]))
    u = np.cross(ref, w)
    u /= np.linalg.norm(u)
    return u, np.cross(w, u), w


def _van_der_corput(n: int, base: int = 2) -> np.ndarray:
    """Low-discrepancy sequence in [0, 1), used to decorrelate two samplings."""
    out = np.zeros(n)
    i = np.arange(1, n + 1, dtype=np.int64)
    f, denom = 1.0, base
    while np.any(i > 0):
        f = 1.0 / denom
        out += f * (i % base)
        i //= base
        denom *= base
        if denom > 2**40:
            break
    return out


def _fibonacci_directions(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    th = GOLDEN_ANGLE * np.arange(n)
    return np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)


@dataclass
class Body:
    """Base class. Subclasses supply geometry, sampling and ray casting."""

    material: Material = Material.CONDUCTOR
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    charge: float = 0.0                   # total deposited charge, coulombs
    n_sites: int = 220
    label: str = ""
    uid: int = field(default_factory=lambda: next(_uid_counter))

    #: Relaxation progress after charge is deposited: 0 = just placed and
    #: still spread uniformly, 1 = settled at equilibrium.  Only meaningful
    #: for conductors; insulators stay at 1.
    relax_t: float = 1.0

    kind: str = "body"
    #: ``(attribute, label)`` pairs the UI turns into size fields.
    params: tuple = ()
    #: Flat bodies are meshed two-sided -- each triangle appears with both
    #: windings -- so averaged *vertex* normals cancel to zero and come out
    #: NaN.  They must be shaded from face normals instead.
    smooth_shading: bool = True
    #: A zero-thickness body carries charge on both faces, so field lines
    #: must be seeded off either side of it.
    two_sided: bool = False
    #: True when the interior is a meaningful place for charge to sit, i.e.
    #: when "migrates to the surface" is a visible statement about the body.
    has_interior: bool = False

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.axis = np.asarray(self.axis, dtype=float).reshape(3)
        n = np.linalg.norm(self.axis)
        self.axis = self.axis / n if n > 0 else np.array([0.0, 0.0, 1.0])
        if not self.label:
            self.label = f"{self.kind}{self.uid - 999}"

    # -- to be provided by subclasses -------------------------------------
    def surface_sites(self, n: int) -> Sites:
        """Sites on the conducting surface (where a conductor's charge lives)."""
        raise NotImplementedError

    def interior_sites(self, n: int) -> Sites:
        """Sites filling the body, for uniformly charged insulators."""
        return self.surface_sites(n)

    def raycast(self, origin: np.ndarray, direction: np.ndarray) -> float | None:
        raise NotImplementedError

    def contains(self, points: np.ndarray, pad: float = 0.0) -> np.ndarray:
        """Which of ``points`` lie inside the body, thickened by ``pad``.

        Used to keep field glyphs out of solid geometry and off the surface of
        a plate, where the regularised field is a smoothing artefact rather
        than anything physical.
        """
        return np.zeros(len(np.atleast_2d(points)), dtype=bool)

    def _local(self, points: np.ndarray) -> np.ndarray:
        u, v, w = self.frame()
        return (np.atleast_2d(points) - self.position) @ np.stack([u, v, w]).T

    def seed_normals(self, pos: np.ndarray) -> np.ndarray:
        """Direction to step a field-line seed clear of the surface.

        The default -- radially outward from the centroid -- is right for a
        sphere.  Flat and wire-like bodies override it, because "away from the
        centre" does not leave their surface.
        """
        d = np.atleast_2d(pos) - self.position
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return np.divide(d, n, out=np.tile([0.0, 0.0, 1.0], (len(d), 1)),
                         where=n > 0)

    def _swirl(self, count: int) -> np.ndarray:
        """Golden-angle sequence, so successive seeds spray around a wire."""
        return GOLDEN_ANGLE * np.arange(count)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """``(verts, faces)`` for rendering."""
        raise NotImplementedError

    def size(self) -> float:
        """Characteristic radius, used for view framing and hit slop."""
        raise NotImplementedError

    # -- shared -----------------------------------------------------------
    @property
    def is_conductor(self) -> bool:
        return self.material is Material.CONDUCTOR

    def sites(self, n: int | None = None) -> Sites:
        """The sites that actually carry this body's charge."""
        n = self.n_sites if n is None else n
        n = max(int(n), 4)
        return (self.surface_sites(n) if self.is_conductor
                else self.interior_sites(n))

    def frame(self):
        return orthonormal_frame(self.axis)

    def _sites(self, local: np.ndarray, measure: float, normal: np.ndarray,
               radius: float) -> Sites:
        n = len(local)
        u, v, w = self.frame()
        world = self.position + local @ np.stack([u, v, w])
        nrm = np.atleast_2d(normal)
        if len(nrm) == 1:
            nrm = np.repeat(nrm, n, axis=0)
        return Sites(world, np.full(n, measure), nrm, np.full(n, radius))


# --------------------------------------------------------------------------
@dataclass
class Line(Body):
    """A straight thin wire of length ``length`` along ``axis``."""

    length: float = 0.8
    wire_radius: float = 0.012
    kind: str = "line"
    params: tuple = (("length", "Length"), ("wire_radius", "Wire radius"))

    def size(self) -> float:
        return 0.5 * self.length

    def surface_sites(self, n: int) -> Sites:
        s = (np.arange(n) + 0.5) / n - 0.5            # cell centres in [-1/2, 1/2]
        local = np.stack([np.zeros(n), np.zeros(n), s * self.length], axis=1)
        step = self.length / n
        return self._sites(local, step, self.axis,
                           max(0.5 * step, self.wire_radius))

    def raycast(self, origin, direction):
        u, v, w = self.frame()
        a = self.position - w * (0.5 * self.length)
        b = self.position + w * (0.5 * self.length)
        t, dist = _ray_segment(origin, direction, a, b)
        return t if dist <= self.wire_radius * 2.5 else None

    def seed_normals(self, pos):
        u, v, w = self.frame()
        psi = self._swirl(len(np.atleast_2d(pos)))
        return np.outer(np.cos(psi), u) + np.outer(np.sin(psi), v)

    def contains(self, points, pad=0.0):
        d = self._local(points)
        z = np.clip(d[:, 2], -self.length / 2, self.length / 2)
        radial = np.hypot(d[:, 0], d[:, 1])
        return np.hypot(radial, d[:, 2] - z) <= self.wire_radius + pad

    def mesh(self):
        u, v, w = self.frame()
        return _tube(self.position - w * 0.5 * self.length,
                     self.position + w * 0.5 * self.length, self.wire_radius)


@dataclass
class Loop(Body):
    """A circular ring of radius ``radius`` with ``axis`` as its normal."""

    radius: float = 0.4
    wire_radius: float = 0.012
    kind: str = "loop"
    params: tuple = (("radius", "Radius"), ("wire_radius", "Wire radius"))

    def size(self) -> float:
        return self.radius

    def surface_sites(self, n: int) -> Sites:
        th = 2.0 * math.pi * (np.arange(n) + 0.5) / n
        local = np.stack([self.radius * np.cos(th), self.radius * np.sin(th),
                          np.zeros(n)], axis=1)
        step = 2.0 * math.pi * self.radius / n
        tangent = np.stack([-np.sin(th), np.cos(th), np.zeros(n)], axis=1)
        u, v, w = self.frame()
        world_t = tangent @ np.stack([u, v, w])
        s = self._sites(local, step, self.axis, max(0.5 * step, self.wire_radius))
        s.normal = world_t
        return s

    def raycast(self, origin, direction):
        u, v, w = self.frame()
        th = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
        pts = (self.position
               + np.outer(self.radius * np.cos(th), u)
               + np.outer(self.radius * np.sin(th), v))
        best = None
        for a, b in zip(pts, np.roll(pts, -1, axis=0)):
            t, dist = _ray_segment(origin, direction, a, b)
            if dist <= self.wire_radius * 2.5 and (best is None or t < best):
                best = t
        return best

    def seed_normals(self, pos):
        u, v, w = self.frame()
        d = self._local(pos)
        phi = np.arctan2(d[:, 1], d[:, 0])
        r_hat = np.outer(np.cos(phi), u) + np.outer(np.sin(phi), v)
        psi = self._swirl(len(d))
        return r_hat * np.cos(psi)[:, None] + w[None, :] * np.sin(psi)[:, None]

    def contains(self, points, pad=0.0):
        d = self._local(points)
        radial = np.hypot(d[:, 0], d[:, 1]) - self.radius
        return np.hypot(radial, d[:, 2]) <= self.wire_radius + pad

    def mesh(self):
        return _torus(self.position, *self.frame(), self.radius, self.wire_radius)


@dataclass
class Sheet(Body):
    """A flat rectangular plate, ``axis`` normal, of size ``width`` x ``height``."""

    width: float = 0.8
    height: float = 0.8
    kind: str = "sheet"
    two_sided: bool = True
    params: tuple = (("width", "Width"), ("height", "Height"))
    smooth_shading: bool = False

    def size(self) -> float:
        return 0.5 * math.hypot(self.width, self.height)

    def _grid(self, n: int) -> tuple[int, int]:
        aspect = self.width / max(self.height, 1e-12)
        nx = max(2, int(round(math.sqrt(n * aspect))))
        ny = max(2, int(round(n / nx)))
        return nx, ny

    def surface_sites(self, n: int) -> Sites:
        nx, ny = self._grid(n)
        x = ((np.arange(nx) + 0.5) / nx - 0.5) * self.width
        y = ((np.arange(ny) + 0.5) / ny - 0.5) * self.height
        gx, gy = np.meshgrid(x, y, indexing="ij")
        local = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
        cell = (self.width / nx) * (self.height / ny)
        return self._sites(local, cell, self.axis, _disc_radius(cell))

    def raycast(self, origin, direction):
        u, v, w = self.frame()
        t = _ray_plane(origin, direction, self.position, w)
        if t is None:
            return None
        d = origin + t * direction - self.position
        if abs(d @ u) <= self.width / 2 and abs(d @ v) <= self.height / 2:
            return t
        return None

    def seed_normals(self, pos):
        return np.tile(self.axis, (len(np.atleast_2d(pos)), 1))

    def contains(self, points, pad=0.0):
        d = self._local(points)
        return ((np.abs(d[:, 2]) <= pad)
                & (np.abs(d[:, 0]) <= self.width / 2 + pad)
                & (np.abs(d[:, 1]) <= self.height / 2 + pad))

    def mesh(self):
        u, v, w = self.frame()
        hw, hh = self.width / 2, self.height / 2
        c = np.array([self.position - hw * u - hh * v,
                      self.position + hw * u - hh * v,
                      self.position + hw * u + hh * v,
                      self.position - hw * u + hh * v])
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]])
        return c.astype(np.float32), faces.astype(np.int32)


@dataclass
class Disk(Body):
    """A flat circular plate of radius ``radius``, ``axis`` normal."""

    radius: float = 0.45
    kind: str = "disk"
    two_sided: bool = True
    params: tuple = (("radius", "Radius"),)
    smooth_shading: bool = False

    def size(self) -> float:
        return self.radius

    def surface_sites(self, n: int) -> Sites:
        # Vogel spiral: equal-area cells, no ring artefacts at the rim.
        i = np.arange(n)
        r = self.radius * np.sqrt((i + 0.5) / n)
        th = GOLDEN_ANGLE * i
        local = np.stack([r * np.cos(th), r * np.sin(th), np.zeros(n)], axis=1)
        cell = math.pi * self.radius**2 / n
        return self._sites(local, cell, self.axis, _disc_radius(cell))

    def raycast(self, origin, direction):
        u, v, w = self.frame()
        t = _ray_plane(origin, direction, self.position, w)
        if t is None:
            return None
        d = origin + t * direction - self.position
        return t if np.linalg.norm(d) <= self.radius else None

    def seed_normals(self, pos):
        return np.tile(self.axis, (len(np.atleast_2d(pos)), 1))

    def contains(self, points, pad=0.0):
        d = self._local(points)
        return ((np.abs(d[:, 2]) <= pad)
                & (np.hypot(d[:, 0], d[:, 1]) <= self.radius + pad))

    def mesh(self):
        u, v, w = self.frame()
        m = 64
        th = np.linspace(0.0, 2.0 * math.pi, m, endpoint=False)
        rim = (self.position + np.outer(self.radius * np.cos(th), u)
               + np.outer(self.radius * np.sin(th), v))
        verts = np.vstack([self.position[None, :], rim])
        faces = []
        for i in range(m):
            j = (i + 1) % m
            faces.append((0, i + 1, j + 1))
            faces.append((0, j + 1, i + 1))       # both faces of a thin plate
        return verts.astype(np.float32), np.array(faces, dtype=np.int32)


@dataclass
class Sphere(Body):
    """A solid ball of radius ``radius``."""

    radius: float = 0.35
    kind: str = "sphere"
    params: tuple = (("radius", "Radius"),)
    has_interior: bool = True

    def size(self) -> float:
        return self.radius

    def surface_sites(self, n: int) -> Sites:
        d = _fibonacci_directions(n)
        pos = self.position + d * self.radius
        area = 4.0 * math.pi * self.radius**2 / n
        return Sites(pos, np.full(n, area), d, np.full(n, _disc_radius(area)))

    def interior_sites(self, n: int) -> Sites:
        d = _fibonacci_directions(n)
        # The Fibonacci direction sequence sweeps z monotonically with the
        # index, so driving the radius off the same index would thread the
        # sites along a single spiral curve rather than filling the ball.  A
        # van der Corput sequence hops between subintervals and decorrelates
        # the two.
        r = self.radius * np.cbrt(_van_der_corput(n))
        pos = self.position + d * r[:, None]
        vol = (4.0 / 3.0) * math.pi * self.radius**3 / n
        return Sites(pos, np.full(n, vol), d, np.full(n, _ball_radius(vol)))

    def raycast(self, origin, direction):
        oc = origin - self.position
        b = 2.0 * float(direction @ oc)
        c = float(oc @ oc) - self.radius**2
        disc = b * b - 4.0 * c
        if disc < 0:
            return None
        t = (-b - math.sqrt(disc)) / 2.0
        if t < 0:
            t = (-b + math.sqrt(disc)) / 2.0
        return t if t >= 0 else None

    def contains(self, points, pad=0.0):
        d = np.atleast_2d(points) - self.position
        return np.linalg.norm(d, axis=1) <= self.radius + pad

    def mesh(self):
        import pyqtgraph.opengl as gl
        md = gl.MeshData.sphere(rows=24, cols=36, radius=self.radius)
        return (md.vertexes().astype(np.float32) + self.position,
                md.faces().astype(np.int32))


BODY_TYPES = {
    "Line": Line, "Loop": Loop, "Sheet": Sheet, "Disk": Disk, "Sphere": Sphere,
}


# --------------------------------------------------------------------------
# Ray helpers
# --------------------------------------------------------------------------
def _ray_plane(origin, direction, point, normal) -> float | None:
    denom = float(direction @ normal)
    if abs(denom) < 1e-12:
        return None
    t = float((point - origin) @ normal) / denom
    return t if t >= 0 else None


def _ray_segment(origin, direction, a, b) -> tuple[float, float]:
    """Closest approach between a ray and a segment: ``(t_along_ray, distance)``."""
    d1, d2 = direction, b - a
    r = origin - a
    a11 = float(d1 @ d1)
    a12 = -float(d1 @ d2)
    a22 = float(d2 @ d2)
    b1 = -float(d1 @ r)
    b2 = float(d2 @ r)
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-18:
        s = 0.0
        t = max(0.0, b1 / a11) if a11 > 0 else 0.0
    else:
        t = (b1 * a22 - a12 * b2) / det
        s = (a11 * b2 - a12 * b1) / det
        t, s = max(t, 0.0), float(np.clip(s, 0.0, 1.0))
        t = max(0.0, float((a + s * d2 - origin) @ d1) / a11)
    p1 = origin + t * d1
    p2 = a + float(np.clip(s, 0.0, 1.0)) * d2
    return t, float(np.linalg.norm(p1 - p2))


def _tube(a, b, radius, n=16):
    axis = b - a
    L = np.linalg.norm(axis)
    u, v, w = orthonormal_frame(axis if L > 0 else np.array([0.0, 0.0, 1.0]))
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ring = np.outer(np.cos(th), u) * radius + np.outer(np.sin(th), v) * radius
    verts = np.vstack([a + ring, b + ring, a[None, :], b[None, :]])
    ca, cb = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i),
                  (ca, j, i), (cb, n + i, n + j)]
    return verts.astype(np.float32), np.array(faces, dtype=np.int32)


def _torus(centre, u, v, w, R, r, nu=48, nv=12):
    a = np.linspace(0.0, 2.0 * math.pi, nu, endpoint=False)
    b = np.linspace(0.0, 2.0 * math.pi, nv, endpoint=False)
    A, B = np.meshgrid(a, b, indexing="ij")
    rad = R + r * np.cos(B)
    local = np.stack([rad * np.cos(A), rad * np.sin(A), r * np.sin(B)], axis=-1)
    verts = centre + local.reshape(-1, 3) @ np.stack([u, v, w])
    faces = []
    for i in range(nu):
        for j in range(nv):
            i2, j2 = (i + 1) % nu, (j + 1) % nv
            p, q = i * nv + j, i * nv + j2
            s, t = i2 * nv + j, i2 * nv + j2
            faces += [(p, s, t), (p, t, q)]
    return verts.astype(np.float32), np.array(faces, dtype=np.int32)
