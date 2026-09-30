"""Field-line seeding and tracing.

Field lines carry information that a grid of glyphs does not: where lines
begin and end, and how densely they crowd together.  Both only mean something
if the seeding is done properly, so lines are allocated in proportion to
charge -- a body with twice the charge gets twice the lines, and within a body
the seeds are drawn in proportion to the *local* surface charge, which makes
the pile-up at a plate's rim show up as a visible thickening of lines.

Tracing runs every line in lockstep, one integrator step at a time, so each
step is a single batched field evaluation rather than one per line.  Lines
that leave the region or terminate on a charge drop out of the batch, so the
cost falls away as they finish.
"""

from __future__ import annotations

import math

import numpy as np

from ..units import K_COULOMB

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
#: Target turn per integrator step.  The step length is adapted to hold this,
#: so a line takes long strides where it runs straight and short ones where it
#: bends -- which is where the steps are actually needed.
TARGET_TURN_DEG = 4.0


class _Field:
    """Field-only evaluator with the source terms hoisted out of the loop.

    The tracer calls this tens of thousands of times on small batches, so the
    per-call setup in the general kernel dominates.  Same regularised
    sources and same arithmetic, just precomputed.
    """

    def __init__(self, pos, q, radius):
        self.pos = np.ascontiguousarray(pos, dtype=float)
        self.kq = K_COULOMB * np.asarray(q, dtype=float)
        a = np.maximum(np.asarray(radius, dtype=float), 1e-12)
        self.a2 = a**2
        self.kq_over_a3 = self.kq / a**3

    def __call__(self, points: np.ndarray) -> np.ndarray:
        if not len(self.pos):
            return np.zeros_like(points)
        diff = points[:, None, :] - self.pos[None, :, :]
        d2 = np.einsum("ckx,ckx->ck", diff, diff)
        d = np.sqrt(d2)
        f = self.kq / (d * d2 + 1e-300)
        np.copyto(f, self.kq_over_a3, where=d2 < self.a2)
        return np.einsum("ck,ckx->cx", f, diff)


def _fibonacci_directions(n: int, offset: int = 0) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    th = GOLDEN_ANGLE * (np.arange(n) + offset)
    return np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)


def _stratified(weights: np.ndarray, k: int) -> np.ndarray:
    """Pick ``k`` indices with probability proportional to ``weights``.

    Deterministic, and stable under small changes to the weights, so lines do
    not flicker between frames while something is being dragged.
    """
    total = weights.sum()
    if k <= 0 or total <= 0:
        return np.zeros(0, dtype=int)
    cdf = np.cumsum(weights) / total
    return np.searchsorted(cdf, (np.arange(k) + 0.5) / k)


def collect_seeds(scene, n_lines: int, domain: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Seed points and integration signs, allocated in proportion to charge.

    The sign is +1 where a line leaves positive charge and -1 where it runs
    backwards into negative charge, so every line is traced away from the
    surface it is attached to.
    """
    state = scene.state()
    if not len(state.q):
        return np.zeros((0, 3)), np.zeros(0)

    weights, emitters = [], []
    for c in scene.charges:
        if c.q != 0.0:
            weights.append(abs(c.q))
            emitters.append(("charge", c))
    for b in scene.bodies:
        sites = state.body_sites.get(b.uid)
        if sites is None or not len(sites[0]):
            continue
        # Total charge *magnitude*, not net: a neutral conductor with induced
        # charge on it should still grow field lines.
        w = float(np.abs(sites[1]).sum())
        if w > 0:
            weights.append(w)
            emitters.append(("body", b))
    if not weights:
        return np.zeros((0, 3)), np.zeros(0)

    weights = np.array(weights)
    share = weights / weights.sum()
    counts = np.maximum(1, np.round(share * n_lines).astype(int))

    seeds, signs = [], []
    for (kind, obj), k in zip(emitters, counts):
        if kind == "charge":
            off = max(1.6 * obj.radius, 0.004 * domain)
            d = _fibonacci_directions(k, offset=obj.uid)
            seeds.append(obj.position + d * off)
            signs.append(np.full(k, math.copysign(1.0, obj.q)))
        else:
            pos, q = state.body_sites[obj.uid]
            pick = _stratified(np.abs(q), k)
            if not len(pick):
                continue
            n_hat = obj.seed_normals(pos)[pick]
            if obj.two_sided:
                # A zero-thickness plate carries charge on both faces, so
                # alternate seeds are pushed off opposite sides.
                flip = np.where(np.arange(len(pick)) % 2 == 0, 1.0, -1.0)
                n_hat = n_hat * flip[:, None]
            off = max(0.012 * domain, 1.5 * float(state.radius.mean()))
            seeds.append(pos[pick] + n_hat * off)
            signs.append(np.sign(q[pick]) + (q[pick] == 0))
    if not seeds:
        return np.zeros((0, 3)), np.zeros(0)
    return np.vstack(seeds), np.concatenate(signs)


def trace(scene, seeds: np.ndarray, signs: np.ndarray, *, domain: float,
          step: float, max_steps: int = 400
          ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Integrate field lines from ``seeds``.

    Returns ``(points, |E| along them, sign)`` for each line that survived,
    dropping any that stopped immediately.  Integration is a
    midpoint (RK2) step along the unit field direction, so the step length is
    an arc length and is independent of field strength.
    """
    n = len(seeds)
    if n == 0:
        return [], [], np.zeros(0)

    state = scene.state()
    src_pos, src_q, src_rad = state.exclude(None)
    # Stop a line once it reaches a body's regularisation shell.  Inside that
    # shell the sources are smoothed, so the direction stops being meaningful
    # and a line will otherwise crawl around in the slab instead of landing.
    bodies = []
    for b in scene.bodies:
        own = state.owner == b.uid
        shell = float(state.radius[own].max()) if own.any() else 0.0
        bodies.append((b, max(0.25 * step, 1.05 * shell)))
    charges = list(scene.charges)
    c_pos = (np.array([c.position for c in charges]) if charges
             else np.zeros((0, 3)))
    c_rad = (np.array([max(c.radius, 1e-9) for c in charges]) if charges
             else np.zeros(0))

    field = _Field(src_pos, src_q, src_rad)

    pts = np.full((n, max_steps + 1, 3), np.nan)
    mags = np.full((n, max_steps + 1), np.nan)
    count = np.zeros(n, dtype=int)

    pos = seeds.copy()
    live = np.arange(n)
    limit = domain
    h_min, h_max = 0.35 * step, 3.0 * step
    h = np.full(n, step)
    prev_dir = np.zeros((n, 3))
    target = math.radians(TARGET_TURN_DEG)
    # Every this many steps, drop any line that has barely moved -- it is
    # circling a null point and will never terminate on its own.
    STALL_EVERY, STALL_DIST = 24, 6.0
    anchor = seeds.copy()
    floor = 0.0

    for it in range(max_steps):
        if not len(live):
            break
        p = pos[live]
        E = field(p)
        mag = np.linalg.norm(E, axis=1)
        if floor == 0.0:
            floor = max(float(np.median(mag)) * 1e-4, 1e-30)

        pts[live, count[live]] = p
        mags[live, count[live]] = mag
        count[live] += 1

        sg = signs[live][:, None]
        ok = mag > floor
        d = np.zeros_like(E)
        d[ok] = E[ok] / mag[ok, None] * sg[ok]

        hl = h[live]
        E2 = field(p + d * (0.5 * hl)[:, None])
        m2 = np.linalg.norm(E2, axis=1)
        ok2 = m2 > floor
        d2v = d.copy()
        d2v[ok2] = E2[ok2] / m2[ok2, None] * sg[ok2]
        nxt = p + d2v * hl[:, None]

        # Adapt the step towards a fixed turn per step.
        if it:
            cos = np.clip(np.einsum("ij,ij->i", d2v, prev_dir[live]), -1.0, 1.0)
            turn = np.arccos(cos)
            scale = np.clip(target / np.maximum(turn, 1e-6), 0.6, 1.6)
            h[live] = np.clip(hl * scale, h_min, h_max)
        prev_dir[live] = d2v

        stop = ~ok
        stop |= (np.abs(nxt) > limit).any(axis=1)          # left the region
        if len(c_pos):
            dist = np.linalg.norm(nxt[:, None, :] - c_pos[None, :, :], axis=2)
            stop |= (dist <= 1.25 * c_rad[None, :]).any(axis=1)   # hit a charge
        for b, pad in bodies:
            stop |= b.contains(nxt, pad=pad)                       # hit a body

        pos[live] = nxt
        if it and it % STALL_EVERY == 0:
            crawled = (np.linalg.norm(nxt - anchor[live], axis=1)
                       < STALL_DIST * step)
            stop |= crawled
            anchor[live] = nxt

        done = live[stop]
        if len(done):
            # Pull the final point back onto the box wall first -- a line is
            # stopped on the step that takes it outside, so without this it
            # pokes up to one step length through.  Then evaluate there:
            # `m2` is the magnitude at the *midpoint* of the step, so reusing
            # it would mis-colour the last segment of every line.
            end = _clip_segments(p[stop], nxt[stop], limit)
            pts[done, count[done]] = end
            mags[done, count[done]] = np.linalg.norm(field(end), axis=1)
            count[done] += 1
        live = live[~stop]

    keep = np.where(count >= 2)[0]
    paths = [pts[i, :count[i]] for i in keep]
    strengths = [mags[i, :count[i]] for i in keep]
    return paths, strengths, signs[keep]


def _clip_segments(p0: np.ndarray, p1: np.ndarray, half: float) -> np.ndarray:
    """Shorten each segment so its far end lands on the box of half-width."""
    outside = np.abs(p1).max(axis=1) > half
    if not outside.any():
        return p1
    d = p1 - p0
    t = np.ones(len(p1))
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(3):
            for sgn in (-half, half):
                tk = (sgn - p0[:, k]) / d[:, k]
                use = np.isfinite(tk) & (tk > 0.0) & (tk <= 1.0)
                t = np.where(use, np.minimum(t, tk), t)
    t = np.where(outside, t, 1.0)
    return p0 + d * t[:, None]


def segments(paths: list[np.ndarray], strengths: list[np.ndarray]
             ) -> tuple[np.ndarray, np.ndarray]:
    """Flatten traced lines into ``mode='lines'`` vertex pairs.

    Returns ``(vertices, |E| per vertex)``.  Emitting explicit pairs rather
    than a strip is what lets each segment take its own colour, which is the
    whole point of colouring a field line by strength.
    """
    if not paths:
        return np.zeros((0, 3), np.float32), np.zeros(0)
    v, m = [], []
    for p, s in zip(paths, strengths):
        if len(p) < 2:
            continue
        v.append(np.repeat(p, 2, axis=0)[1:-1])
        m.append(np.repeat(s, 2)[1:-1])
    if not v:
        return np.zeros((0, 3), np.float32), np.zeros(0)
    return np.vstack(v).astype(np.float32), np.concatenate(m)


def direction_markers(paths: list[np.ndarray], strengths: list[np.ndarray],
                      signs_per_path: np.ndarray, per_line: int = 3
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points, field directions and |E| at a few places along each line.

    A bare line is ambiguous about which way the field points -- information
    the glyph mode gives for free -- so the renderer puts a small head at each
    of these.  The path is traced along ``sign * Ehat``, so multiplying the
    path tangent by the sign recovers the field direction itself.
    """
    pos, tan, mag = [], [], []
    for p, m, sgn in zip(paths, strengths, signs_per_path):
        if len(p) < 4:
            continue
        stride = max(2, len(p) // (per_line + 1))
        idx = np.arange(stride, len(p) - 1, stride)
        if not len(idx):
            continue
        t = p[idx + 1] - p[idx - 1]
        n = np.linalg.norm(t, axis=1, keepdims=True)
        t = np.divide(t, n, out=np.zeros_like(t), where=n > 0) * sgn
        pos.append(p[idx])
        tan.append(t)
        mag.append(m[idx])
    if not pos:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    return np.vstack(pos), np.vstack(tan), np.concatenate(mag)
