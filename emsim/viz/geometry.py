"""Vectorised mesh construction for arrow glyphs.

Every arrow in a field is packed into a *single* triangle soup so the whole
vector field is one GL draw call.  Rebuilding a few hundred arrows each frame
is then a handful of numpy ops rather than hundreds of Qt objects.
"""

from __future__ import annotations

import numpy as np

# Canonical arrow geometry, pointing along +z from 0 to 1, of unit "width".
_SHAFT_R = 0.045
_HEAD_R = 0.13
_HEAD_FRAC = 0.30


def unit_arrow(n_sides: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Vertices ``(V, 3)`` and faces ``(F, 3)`` of a unit arrow along +z.

    A closed, consistently outward-wound shell: an ``n``-gon shaft capped at
    the bottom, an annulus forming the underside of the head, and a cone to
    the tip.  Winding matters the moment a caller enables face culling or a
    lighting shader, so it is worth getting right in the primitive.
    """
    n = n_sides
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cs, sn = np.cos(th), np.sin(th)
    z_neck = 1.0 - _HEAD_FRAC

    verts = [np.zeros(3)]                                   # 0: base centre
    verts += [np.array([_SHAFT_R * cs[i], _SHAFT_R * sn[i], 0.0])
              for i in range(n)]                            # b: shaft bottom
    verts += [np.array([_SHAFT_R * cs[i], _SHAFT_R * sn[i], z_neck])
              for i in range(n)]                            # t: shaft top
    verts += [np.array([_HEAD_R * cs[i], _HEAD_R * sn[i], z_neck])
              for i in range(n)]                            # h: head rim
    verts.append(np.array([0.0, 0.0, 1.0]))                 # tip

    b0, t0, h0, tip = 1, 1 + n, 1 + 2 * n, 1 + 3 * n
    faces = []
    for i in range(n):
        j = (i + 1) % n
        bi, bj, ti, tj = b0 + i, b0 + j, t0 + i, t0 + j
        hi, hj = h0 + i, h0 + j
        faces.append((0, bj, bi))                           # base cap  (-z)
        faces.append((bi, bj, tj))                          # shaft wall
        faces.append((bi, tj, ti))
        faces.append((ti, hj, hi))                          # head underside (-z)
        faces.append((ti, tj, hj))
        faces.append((tip, hi, hj))                         # cone
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def rotations_to(dirs: np.ndarray) -> np.ndarray:
    """``(N, 3, 3)`` rotations taking +z onto each unit vector in ``dirs``."""
    d = np.asarray(dirs, dtype=float).reshape(-1, 3)
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    d = np.divide(d, nrm, out=np.tile([0.0, 0.0, 1.0], (len(d), 1)),
                  where=nrm > 0)

    # Rodrigues about v = z x d, with c = z . d
    vx, vy = -d[:, 1], d[:, 0]
    c = d[:, 2]
    flipped = c < -0.999999
    k = np.where(flipped, 0.0, 1.0 / np.where(flipped, 1.0, 1.0 + c))

    K = np.zeros((len(d), 3, 3))
    K[:, 0, 2] = vy
    K[:, 1, 2] = -vx
    K[:, 2, 0] = -vy
    K[:, 2, 1] = vx
    R = np.eye(3)[None] + K + np.einsum("nij,njk->nik", K, K) * k[:, None, None]
    R[flipped] = np.diag([1.0, -1.0, -1.0])
    return R


#: Fixed world-space light used by :func:`arrow_soup`'s built-in shading.
LIGHT_DIR = np.array([0.42, 0.50, 0.76])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


def face_shade(tris: np.ndarray, ambient: float) -> np.ndarray:
    """Per-face brightness in ``[ambient, 1]`` from a fixed *world* light.

    pyqtgraph's built-in 'shaded' shader lights from a direction fixed in eye
    space and floors unlit faces at 0.2, which turns a plate whose normal runs
    across the view almost black and swings a colour-coded glyph across the
    whole colormap.  Shading against a world-space light with a raised floor
    keeps both readable.
    """
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    mag = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, mag, out=np.zeros_like(nrm), where=mag > 0)
    lam = np.abs(nrm @ LIGHT_DIR)
    return (ambient + (1.0 - ambient) * lam).astype(np.float32)


def shade_mesh(verts: np.ndarray, faces: np.ndarray, ambient: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Expand an indexed mesh to a shaded triangle soup.

    Returns ``(tris, shade)``; multiply ``shade`` by an RGBA to get the vertex
    colours, which is what lets hover recolour a body without re-meshing it.
    """
    tris = verts[faces].astype(np.float32)
    return tris, face_shade(tris, ambient)


def arrow_soup(origins: np.ndarray, directions: np.ndarray,
               lengths: np.ndarray, widths: np.ndarray,
               colors: np.ndarray, n_sides: int = 10,
               ambient: float | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Build one triangle soup for N arrows.

    Returns ``(F, 3, 3)`` triangle vertices and ``(F, 3, 4)`` vertex colours,
    the un-indexed form :class:`GLMeshItem` consumes directly.

    ``ambient`` bakes a mild diffuse term into the vertex colours instead of
    letting the GL 'shaded' shader do it.  That matters when colour *encodes*
    a value: full Lambertian shading swings a face's brightness over the whole
    range of the colormap, so a glyph's colour no longer matches its colorbar
    entry.  Clamping the shading to ``[ambient, 1]`` keeps the form readable
    while holding the colour close to the true one.
    """
    origins = np.asarray(origins, dtype=float).reshape(-1, 3)
    n = len(origins)
    if n == 0:
        return np.zeros((0, 3, 3), np.float32), np.zeros((0, 3, 4), np.float32)

    lengths = np.broadcast_to(np.asarray(lengths, dtype=float), (n,))
    widths = np.broadcast_to(np.asarray(widths, dtype=float), (n,))
    colors = np.asarray(colors, dtype=np.float32).reshape(n, 4)

    base_v, base_f = unit_arrow(n_sides)
    scale = np.stack([widths, widths, lengths], axis=1)              # (n, 3)
    local = base_v[None, :, :] * scale[:, None, :]                   # (n, V, 3)
    R = rotations_to(directions)
    world = np.einsum("nij,nvj->nvi", R, local) + origins[:, None, :]

    tris = world[:, base_f, :].reshape(-1, 3, 3)                     # (n*F, 3, 3)
    cols = np.repeat(colors, len(base_f) * 3, axis=0).reshape(-1, 3, 4)

    if ambient is not None:
        cols = cols.copy()
        cols[:, :, :3] *= face_shade(tris, ambient)[:, None, None]
    return tris.astype(np.float32), cols


def box_lines(half: float) -> np.ndarray:
    """Line-segment vertex array (``mode='lines'``) for a cube of half-width."""
    h = half
    c = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    return np.array([c[i] for e in edges for i in e], dtype=np.float32)
