"""Value -> [0, 1] normalisations, each able to describe its own colorbar ticks.

Keeping tick generation on the norm (rather than in the colorbar widget) is
what lets a log or symlog colorbar be labelled correctly without the widget
knowing anything about the mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LinearNorm:
    vmin: float
    vmax: float

    def __call__(self, x):
        span = self.vmax - self.vmin
        if span <= 0:
            return np.zeros_like(np.asarray(x, dtype=float))
        return np.clip((np.asarray(x, dtype=float) - self.vmin) / span, 0.0, 1.0)

    def ticks(self) -> list[tuple[float, float]]:
        vals = _nice_ticks(self.vmin, self.vmax, 5)
        return [(float(self(v)), v) for v in vals]


@dataclass
class LogNorm:
    vmin: float
    vmax: float

    def __post_init__(self) -> None:
        self.vmin = max(self.vmin, 1e-300)
        self.vmax = max(self.vmax, self.vmin * 10.0)

    def __call__(self, x):
        x = np.abs(np.asarray(x, dtype=float))
        lo, hi = math.log10(self.vmin), math.log10(self.vmax)
        with np.errstate(divide="ignore"):
            lx = np.log10(np.maximum(x, 1e-300))
        return np.clip((lx - lo) / (hi - lo), 0.0, 1.0)

    def ticks(self) -> list[tuple[float, float]]:
        lo, hi = math.log10(self.vmin), math.log10(self.vmax)
        decades = list(range(math.ceil(lo), math.floor(hi) + 1))
        if len(decades) > 8:
            step = math.ceil(len(decades) / 8)
            decades = decades[::step]
        out = [(float(self(10.0**d)), 10.0**d) for d in decades]
        if len(out) < 4:
            out = [(0.0, self.vmin)] + out + [(1.0, self.vmax)]
        return out


@dataclass
class SymLogNorm:
    """Symmetric log about zero: linear inside +-``linthresh``, log outside.

    ``lin_frac`` is the fraction of each half of the bar given to the linear
    core.  Output is in [0, 1] with 0.5 at zero.
    """

    vmax: float
    linthresh: float
    lin_frac: float = 0.12

    def __post_init__(self) -> None:
        self.vmax = max(abs(self.vmax), 1e-300)
        self.linthresh = min(max(abs(self.linthresh), self.vmax * 1e-9),
                             self.vmax * 0.5)

    def _half(self, a):
        """|value| -> [0, 1] position within one half of the bar."""
        lt, vm, lf = self.linthresh, self.vmax, self.lin_frac
        a = np.asarray(a, dtype=float)
        lin = (a / lt) * lf
        with np.errstate(divide="ignore", invalid="ignore"):
            log = lf + (1.0 - lf) * np.log10(np.maximum(a, lt) / lt) / math.log10(vm / lt)
        return np.clip(np.where(a <= lt, lin, log), 0.0, 1.0)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return np.clip(0.5 + 0.5 * np.sign(x) * self._half(np.abs(x)), 0.0, 1.0)

    def ticks(self) -> list[tuple[float, float]]:
        lo = math.ceil(math.log10(self.linthresh))
        hi = math.floor(math.log10(self.vmax))
        decades = list(range(lo, hi + 1))
        if len(decades) > 4:
            step = math.ceil(len(decades) / 4)
            decades = decades[::step]
        vals = [0.0]
        for d in decades:
            vals += [10.0**d, -(10.0**d)]
        vals += [self.vmax, -self.vmax]
        seen, out = set(), []
        for v in sorted(vals):
            k = f"{v:.4g}"
            if k not in seen:
                seen.add(k)
                out.append((float(self(v)), v))
        return out


def _nice_ticks(lo: float, hi: float, count: int) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo, hi]
    raw = (hi - lo) / max(count - 1, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            step = m * mag
            break
    else:
        step = 10.0 * mag
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi * (1 + 1e-9) and len(out) < 20:
        out.append(round(v, 12))
        v += step
    return out or [lo, hi]


def robust_range(values: np.ndarray, lo_pct=2.0, hi_pct=98.0) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    return float(np.percentile(v, lo_pct)), float(np.percentile(v, hi_pct))
