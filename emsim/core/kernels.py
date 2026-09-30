"""The Coulomb summation kernel.

Everything in the program -- field glyphs, the potential volume, forces, and
the right-hand side of the conductor solve -- ultimately goes through here.
There is no closed-form idealisation for any geometry: a plate is a sum over
its sites, so fringing at its edge is simply what the sum gives.

Each source is regularised as a uniformly charged ball of radius ``a``: exact
Coulomb outside, the solid-sphere solution inside.  That keeps the field finite
where a sample point lands on top of a source without perturbing anything at a
distance.
"""

from __future__ import annotations

import numpy as np

from ..units import K_COULOMB

#: Cap on the (points x sources x 3) temporary, in bytes.
_CHUNK_BYTES = 48 << 20


def coulomb_sum(src_pos: np.ndarray, src_q: np.ndarray, src_radius: np.ndarray,
                points: np.ndarray, want_E: bool = True, want_V: bool = True
                ) -> tuple[np.ndarray, np.ndarray]:
    """Field and potential at ``points`` from a set of regularised charges.

    Returns ``(E, V)`` with shapes ``(P, 3)`` and ``(P,)``.  Evaluation is
    chunked over sample points so that memory stays bounded no matter how many
    sites the bodies have been discretised into.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    P = len(points)
    E = np.zeros((P, 3))
    V = np.zeros(P)
    K = len(src_pos)
    if K == 0 or P == 0:
        return E, V

    src_pos = np.asarray(src_pos, dtype=float)
    q = np.asarray(src_q, dtype=float)
    a = np.maximum(np.asarray(src_radius, dtype=float), 1e-12)
    kq = K_COULOMB * q
    a3 = a**3

    step = max(1, int(_CHUNK_BYTES / max(K * 24, 1)))
    for lo in range(0, P, step):
        hi = min(lo + step, P)
        diff = points[lo:hi, None, :] - src_pos[None, :, :]      # (C, K, 3)
        d2 = np.einsum("ckx,ckx->ck", diff, diff)
        d = np.sqrt(d2)
        inside = d < a
        d_safe = np.where(d > 0.0, d, 1.0)

        if want_V:
            v_out = kq / d_safe
            v_in = kq * (3.0 * a**2 - d2) / (2.0 * a3)
            V[lo:hi] = np.where(inside, v_in, v_out).sum(axis=1)
        if want_E:
            # E = sum f * diff, with f chosen so no unit vector is needed.
            f_out = kq / (d_safe**3)
            f_in = kq / a3
            f = np.where(inside, f_in, f_out)
            E[lo:hi] = np.einsum("ck,ckx->cx", f, diff)
    return E, V


def potential_matrix(pos: np.ndarray, self_potential: np.ndarray) -> np.ndarray:
    """Dense potential-coefficient matrix ``P`` with ``V_i = sum_j P_ij q_j``.

    Off-diagonal entries use exact point collocation, ``K / |r_i - r_j|``.  The
    diagonal is supplied by the caller, because the self-potential of a site
    depends on the shape it stands for (a disc of equal area for a surface
    patch, a segment of wire for a line element).
    """
    n = len(pos)
    if n == 0:
        return np.zeros((0, 0))
    diff = pos[:, None, :] - pos[None, :, :]
    d = np.sqrt(np.einsum("ijx,ijx->ij", diff, diff))
    np.fill_diagonal(d, 1.0)
    P = K_COULOMB / d
    np.fill_diagonal(P, self_potential)
    return P
