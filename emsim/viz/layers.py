"""Visualization layers.

Each layer owns a set of GL items and knows how to rebuild them from a
:class:`~emsim.core.scene.Scene`.  A layer may publish a :class:`ColorScale`,
which the UI turns into a colorbar.  New physics (B-field streamlines,
radiated power, ...) plugs in by adding a class here and registering it in
:class:`LayerStack`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph.opengl as gl

from ..core.entities import REFERENCE_CHARGE
from ..units import K_COULOMB, Quantity, UnitSystem
from . import colormaps as cmaps
from .geometry import arrow_soup
from .overlay import Label3D, LabelSet
from .norms import LinearNorm, LogNorm, SymLogNorm, robust_range

POSITIVE_RGBA = (0.95, 0.35, 0.30, 1.0)
NEGATIVE_RGBA = (0.30, 0.55, 0.98, 1.0)
NEUTRAL_RGBA = (0.75, 0.75, 0.78, 1.0)
FORCE_RGBA = (1.0, 0.86, 0.25, 1.0)

# -- Force arrow calibration -----------------------------------------------
# The force scale is *absolute*: it never renormalises to whatever happens to
# be the largest force on screen.  That is the whole point -- an arrow has to
# visibly grow as two charges approach, and shrink as they separate, so the
# 1/r^2 falloff is something you can watch rather than something you have to
# read off the labels.
#
# The calibration below says: an arrow of FORCE_REF_LENGTH x the domain
# half-width represents FORCE_REFERENCE, the Coulomb force between two
# reference charges 0.7 m apart -- which is exactly the dipole the app opens
# with, so the default scene starts at a sensible arrow length.
FORCE_REFERENCE = K_COULOMB * REFERENCE_CHARGE**2 / 0.7**2      # 1.83 uN
FORCE_REF_LENGTH = 0.30
#: Hard ceiling, in units of the domain half-width.  Well outside the visible
#: box, so it never truncates a force you are actually looking at; it exists
#: only to stop near-coincident charges generating absurd geometry.
FORCE_MAX_LENGTH = 8.0


def force_arrow_length(magnitude, domain: float, gain: float = 1.0):
    """Map force magnitude [N] to arrow length [m] on the fixed scale.

    Strictly proportional to ``magnitude`` -- no dependence on the other
    forces in the scene -- up to :data:`FORCE_MAX_LENGTH`.
    """
    mag = np.asarray(magnitude, dtype=float)
    length = (FORCE_REF_LENGTH * domain * gain) * (mag / FORCE_REFERENCE)
    return np.minimum(length, FORCE_MAX_LENGTH * domain)


def force_gain_for(magnitude, gain_limits=(1e-6, 1e6)) -> float:
    """Gain that would draw a force of ``magnitude`` at the reference length.

    Used by the panel's one-shot 'Fit' button, which recalibrates the scale
    without making it adaptive.
    """
    mag = float(np.max(magnitude)) if np.size(magnitude) else 0.0
    if mag <= 0.0:
        return 1.0
    return float(np.clip(FORCE_REFERENCE / mag, *gain_limits))


@dataclass
class ColorScale:
    """Everything a colorbar needs to draw itself."""

    norm: object
    cmap: str
    title: str
    quantity: Quantity


@dataclass
class RenderSettings:
    units: UnitSystem
    domain: float = 1.0                # half-width of the region of interest, m
    selected_uid: int | None = None

    show_force: bool = True
    force_gain: float = 1.0
    force_labels: bool = True

    show_efield: bool = False
    field_grid: int = 9
    field_log: bool = True
    field_cmap: str = "viridis"
    field_len_frac: float = 0.55       # arrow length as a fraction of spacing

    show_potential: bool = False
    pot_mode: str = "volume"           # "volume" | "slice"
    pot_res: int = 40
    pot_cmap: str = "coolwarm"
    pot_symlog: bool = True
    pot_alpha: float = 0.90
    slice_axis: int = 2                # 0=x, 1=y, 2=z
    slice_pos: float = 0.0             # metres

    quality: float = 1.0               # <1 while dragging, for responsiveness

    def eff_field_grid(self) -> int:
        return max(3, int(round(self.field_grid * self.quality)))

    def eff_pot_res(self) -> int:
        return max(8, int(round(self.pot_res * self.quality)))


class Layer:
    """Base class: owns GL items, rebuilt wholesale on update."""

    def __init__(self, view: gl.GLViewWidget) -> None:
        self.view = view
        self.items: list[gl.GLGraphicsItem.GLGraphicsItem] = []
        self.scale: ColorScale | None = None
        self.labels = LabelSet()
        self.visible = True

    # -- item bookkeeping --------------------------------------------------
    def _add(self, item):
        self.view.addItem(item)
        self.items.append(item)
        return item

    def clear(self) -> None:
        for item in self.items:
            self.view.removeItem(item)
        self.items.clear()

    def update(self, scene, st: RenderSettings) -> None:
        self.clear()
        self.scale = None
        self.labels = LabelSet()
        if self.visible:
            self.rebuild(scene, st)

    def rebuild(self, scene, st: RenderSettings) -> None:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------
class ChargeLayer(Layer):
    """Spheres for the charges, plus a highlight ring on the selected one."""

    def __init__(self, view) -> None:
        super().__init__(view)
        self._sphere = gl.MeshData.sphere(rows=16, cols=24, radius=1.0)

    def rebuild(self, scene, st: RenderSettings) -> None:
        for c in scene.charges:
            rgba = POSITIVE_RGBA if c.q > 0 else (
                NEGATIVE_RGBA if c.q < 0 else NEUTRAL_RGBA)
            if c.uid == st.selected_uid:
                rgba = tuple(min(1.0, v * 1.35) for v in rgba[:3]) + (1.0,)
            item = gl.GLMeshItem(meshdata=self._sphere, smooth=True,
                                 color=rgba, shader="shaded",
                                 glOptions="opaque")
            item.scale(c.radius, c.radius, c.radius)
            item.translate(*c.position)
            self._add(item)

            if c.uid == st.selected_uid:
                ring = gl.GLMeshItem(meshdata=self._sphere, smooth=True,
                                     color=(1.0, 1.0, 1.0, 0.22),
                                     shader="balloon", glOptions="translucent")
                r = c.radius * 1.45
                ring.scale(r, r, r)
                ring.translate(*c.position)
                self._add(ring)


# --------------------------------------------------------------------------
class ForceLayer(Layer):
    """One arrow per charge, length proportional to |F| on a fixed scale."""

    def rebuild(self, scene, st: RenderSettings) -> None:
        if not scene.charges:
            return
        F = scene.forces()
        mag = np.linalg.norm(F, axis=1)
        lengths = force_arrow_length(mag, st.domain, st.force_gain)

        # Skip anything too short to form sane geometry; do not floor it to a
        # minimum, which would misrepresent a genuinely tiny force.
        keep = lengths > 1e-4 * st.domain
        if not keep.any():
            return

        origins = np.array([c.position for c in scene.charges])[keep]
        radii = np.array([c.radius for c in scene.charges])[keep]
        dirs = F[keep] / mag[keep][:, None]
        L = lengths[keep]

        # Start the arrow at the sphere surface, not its centre.
        origins = origins + dirs * radii[:, None]
        colors = np.tile(np.array(FORCE_RGBA, dtype=np.float32), (len(L), 1))
        tris, cols = arrow_soup(origins, dirs, L, 0.30 * L, colors,
                                ambient=0.55)
        self._add(gl.GLMeshItem(vertexes=tris, vertexColors=cols, smooth=False,
                                shader=None, glOptions="opaque"))

        if st.force_labels:
            # Anchor at the arrow midpoint, lifted slightly, so labels for a
            # symmetric pair land on opposite sides instead of on top of
            # each other at the tips.
            anchors = (origins + dirs * (0.5 * L)[:, None]
                       + np.array([0.0, 0.0, 0.055 * st.domain]))
            for pos, m in zip(anchors, mag[keep]):
                self.labels.add(pos, st.units.fmt(float(m), Quantity.FORCE),
                                (255, 226, 130))


# --------------------------------------------------------------------------
class EFieldLayer(Layer):
    """Fixed-length arrows on a 3-D grid, coloured by |E|."""

    def rebuild(self, scene, st: RenderSettings) -> None:
        if not scene.charges:
            return
        n = st.eff_field_grid()
        axis = np.linspace(-st.domain, st.domain, n)
        gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        # Drop samples buried inside a charge, where the arrow would be hidden
        # and the magnitude is dominated by the softening model.
        pos = scene.positions()
        rad = np.array([c.radius for c in scene.charges])
        d = np.linalg.norm(pts[:, None, :] - pos[None, :, :], axis=2)
        pts = pts[(d > 1.3 * rad[None, :]).all(axis=1)]
        if len(pts) == 0:
            return

        E = scene.evaluate(pts)["E"]
        mag = np.linalg.norm(E, axis=1)
        live = mag > 0
        pts, E, mag = pts[live], E[live], mag[live]
        if len(pts) == 0:
            return

        lo, hi = robust_range(mag, 8.0, 98.0)
        if st.field_log:
            hi = max(hi, 1e-300)
            # Three decades keeps the weak-field glyphs off the very dark end
            # of the colormap, where they vanish against the background.
            norm = LogNorm(max(lo, hi * 1e-3), hi)
        else:
            norm = LinearNorm(0.0, max(hi, 1e-300))

        t = np.asarray(norm(mag), dtype=float)
        colors = cmaps.map_rgba_float(st.field_cmap, t, 1.0)

        spacing = (2.0 * st.domain) / max(n - 1, 1)
        length = st.field_len_frac * spacing
        dirs = E / mag[:, None]
        # Centre each glyph on its grid point rather than starting there.
        origins = pts - dirs * (0.5 * length)

        tris, cols = arrow_soup(origins, dirs, np.full(len(pts), length),
                                np.full(len(pts), length), colors, n_sides=7,
                                ambient=0.72)
        self._add(gl.GLMeshItem(vertexes=tris, vertexColors=cols, smooth=False,
                                shader=None, glOptions="opaque"))

        self.scale = ColorScale(norm, st.field_cmap,
                                f"|E|  ({st.units.unit_symbol(Quantity.EFIELD)})",
                                Quantity.EFIELD)


# --------------------------------------------------------------------------
class PotentialLayer(Layer):
    """Semi-transparent heatmap of V, as a volume render or a cut plane."""

    def rebuild(self, scene, st: RenderSettings) -> None:
        if not scene.charges:
            return
        if st.pot_mode == "slice":
            self._rebuild_slice(scene, st)
        else:
            self._rebuild_volume(scene, st)

    # -- shared -----------------------------------------------------------
    def _make_norm(self, V: np.ndarray, st: RenderSettings):
        """Return ``(norm, vmax)`` for this potential field, or ``None``."""
        a = np.abs(V[np.isfinite(V)])
        if a.size == 0 or a.max() == 0:
            return None
        vmax = float(np.percentile(a, 99.0))
        if vmax <= 0:
            vmax = float(a.max())
        if st.pot_symlog:
            pos = a[a > 0]
            lin = float(np.percentile(pos, 25.0)) if pos.size else vmax * 1e-3
            return SymLogNorm(vmax, max(lin, vmax * 1e-6)), vmax
        return LinearNorm(-vmax, vmax), vmax

    @staticmethod
    def _opacity_weight(V: np.ndarray, vmax: float) -> np.ndarray:
        """Per-sample opacity, always linear in |V| even when the colour is not.

        Driving alpha off the symlog norm would leave the whole box hazy,
        because a log scale gives far-field samples a large normalised value.
        A square-root ramp on |V| / vmax keeps weak regions see-through while
        still lifting the mid range above nothing.
        """
        return np.clip(np.abs(V) / max(vmax, 1e-300), 0.0, 1.0) ** 0.5

    def _publish(self, norm, st: RenderSettings) -> None:
        sym = st.units.unit_symbol(Quantity.POTENTIAL)
        self.scale = ColorScale(norm, st.pot_cmap, f"V  ({sym})",
                                Quantity.POTENTIAL)

    # -- volume -----------------------------------------------------------
    def _rebuild_volume(self, scene, st: RenderSettings) -> None:
        n = st.eff_pot_res()
        dx = 2.0 * st.domain / n
        c = -st.domain + (np.arange(n) + 0.5) * dx          # cell centres
        gx, gy, gz = np.meshgrid(c, c, c, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        V = scene.evaluate(pts)["V"].reshape(n, n, n)
        made = self._make_norm(V, st)
        if made is None:
            return
        norm, vmax = made

        t = np.asarray(norm(V), dtype=float)                # [0, 1], 0.5 = zero
        rgb = cmaps.map_rgb(st.pot_cmap, t)
        weight = self._opacity_weight(V, vmax)
        # The volume is composited from ~n stacked slices, so a per-voxel
        # alpha of `a` accumulates to 1 - (1 - a)^n across the stack.  Invert
        # that so `pot_alpha` means the opacity of a full traverse.
        a_max = 1.0 - (1.0 - np.clip(st.pot_alpha, 0.0, 0.995)) ** (1.0 / n)
        alpha = np.clip(a_max * 255.0 * weight, 0, 255)

        data = np.empty((n, n, n, 4), dtype=np.ubyte)
        data[..., :3] = rgb
        data[..., 3] = alpha.astype(np.ubyte)

        item = gl.GLVolumeItem(data, smooth=True, glOptions="translucent")
        item.scale(dx, dx, dx)
        item.translate(-st.domain, -st.domain, -st.domain)
        self._add(item)
        self._publish(norm, st)

    # -- slice ------------------------------------------------------------
    def _rebuild_slice(self, scene, st: RenderSettings) -> None:
        n = max(st.eff_pot_res() * 3, 48)
        dx = 2.0 * st.domain / n
        c = -st.domain + (np.arange(n) + 0.5) * dx
        u, v = np.meshgrid(c, c, indexing="ij")
        flat = np.full(u.size, float(st.slice_pos))
        ax = st.slice_axis
        if ax == 2:      # z = const, image spans (x, y)
            pts = np.stack([u.ravel(), v.ravel(), flat], axis=1)
        elif ax == 1:    # y = const, image spans (x, z)
            pts = np.stack([u.ravel(), flat, v.ravel()], axis=1)
        else:            # x = const, image spans (y, z)
            pts = np.stack([flat, u.ravel(), v.ravel()], axis=1)

        V = scene.evaluate(pts)["V"].reshape(n, n)
        made = self._make_norm(V, st)
        if made is None:
            return
        norm, _vmax = made

        t = np.asarray(norm(V), dtype=float)
        data = np.empty((n, n, 4), dtype=np.ubyte)
        data[..., :3] = cmaps.map_rgb(st.pot_cmap, t)
        # A cut plane is a single surface, so its opacity is the slider value.
        data[..., 3] = np.ubyte(np.clip(st.pot_alpha * 255.0, 0, 255))

        item = gl.GLImageItem(data, smooth=True, glOptions="translucent")
        item.scale(dx, dx, 1.0)
        d = st.domain
        if ax == 2:
            item.translate(-d, -d, float(st.slice_pos))
        elif ax == 1:
            item.rotate(90, 1, 0, 0)
            item.translate(-d, float(st.slice_pos), -d)
        else:
            item.rotate(90, 1, 0, 0)
            item.rotate(90, 0, 0, 1)
            item.translate(float(st.slice_pos), -d, -d)
        self._add(item)
        self._publish(norm, st)


# --------------------------------------------------------------------------
class LayerStack:
    """Owns the layers and drives them from a single settings object."""

    def __init__(self, view) -> None:
        self.charges = ChargeLayer(view)
        self.force = ForceLayer(view)
        self.efield = EFieldLayer(view)
        self.potential = PotentialLayer(view)
        # Draw order matters for translucency: opaque geometry first.
        self.all: list[Layer] = [self.charges, self.force, self.efield,
                                 self.potential]

    def update(self, scene, st: RenderSettings) -> None:
        self.force.visible = st.show_force
        self.efield.visible = st.show_efield
        self.potential.visible = st.show_potential
        for layer in self.all:
            layer.update(scene, st)

    def scales(self) -> list[ColorScale]:
        return [l.scale for l in self.all if l.scale is not None]

    def labels(self) -> list[Label3D]:
        out: list[Label3D] = []
        for layer in self.all:
            out.extend(layer.labels.items)
        return out
