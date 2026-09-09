"""Self-contained colormap lookup tables (no matplotlib dependency)."""

from __future__ import annotations

import numpy as np

_ANCHORS: dict[str, list[tuple[int, int, int]]] = {
    # perceptually uniform, sequential -- for |E|
    "viridis": [
        (68, 1, 84), (72, 26, 108), (71, 47, 125), (65, 68, 135),
        (57, 86, 140), (49, 104, 142), (42, 120, 142), (35, 136, 142),
        (31, 152, 139), (34, 168, 132), (53, 183, 121), (84, 197, 104),
        (122, 209, 81), (165, 219, 54), (210, 226, 27), (253, 231, 37),
    ],
    "inferno": [
        (0, 0, 4), (20, 11, 52), (58, 9, 99), (96, 19, 110),
        (133, 33, 107), (169, 46, 94), (203, 65, 73), (230, 93, 47),
        (247, 131, 17), (252, 173, 18), (245, 216, 70), (252, 255, 164),
    ],
    "plasma": [
        (13, 8, 135), (75, 3, 161), (125, 3, 168), (168, 34, 150),
        (203, 70, 121), (229, 107, 93), (248, 148, 65), (253, 195, 40),
        (240, 249, 33),
    ],
    # diverging -- for potential (sign matters)
    "coolwarm": [
        (59, 76, 192), (98, 130, 234), (141, 176, 254), (184, 208, 249),
        (221, 221, 221), (245, 196, 173), (244, 154, 123), (222, 96, 77),
        (180, 4, 38),
    ],
    "bwr": [
        (0, 0, 255), (128, 128, 255), (255, 255, 255),
        (255, 128, 128), (255, 0, 0),
    ],
    "seismic": [
        (0, 0, 76), (0, 0, 255), (255, 255, 255), (255, 0, 0), (128, 0, 0),
    ],
}

SEQUENTIAL = ("viridis", "inferno", "plasma")
DIVERGING = ("coolwarm", "bwr", "seismic")

_cache: dict[str, np.ndarray] = {}


def lut(name: str, n: int = 256) -> np.ndarray:
    """Return an ``(n, 3)`` uint8 lookup table."""
    key = f"{name}:{n}"
    if key in _cache:
        return _cache[key]
    anchors = np.array(_ANCHORS[name], dtype=float)
    src = np.linspace(0.0, 1.0, len(anchors))
    dst = np.linspace(0.0, 1.0, n)
    table = np.stack([np.interp(dst, src, anchors[:, c]) for c in range(3)],
                     axis=1)
    table = np.clip(np.round(table), 0, 255).astype(np.uint8)
    _cache[key] = table
    return table


def map_rgb(name: str, t: np.ndarray) -> np.ndarray:
    """Map ``t`` in [0, 1] to ``(..., 3)`` uint8 RGB."""
    table = lut(name)
    idx = np.clip((np.nan_to_num(t) * (len(table) - 1)).round(), 0,
                  len(table) - 1).astype(np.int32)
    return table[idx]


def map_rgba_float(name: str, t: np.ndarray, alpha=1.0) -> np.ndarray:
    """Map ``t`` in [0, 1] to ``(..., 4)`` float RGBA in [0, 1]."""
    rgb = map_rgb(name, t).astype(np.float32) / 255.0
    a = np.broadcast_to(np.asarray(alpha, dtype=np.float32),
                        rgb.shape[:-1])[..., None]
    return np.concatenate([rgb, a], axis=-1)
