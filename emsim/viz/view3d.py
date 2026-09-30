"""Interactive 3-D viewport.

Adds to :class:`GLViewWidget` the two things it lacks for direct manipulation:
picking (screen-space projection of the charges) and dragging (unprojecting
the cursor onto a plane through the grabbed charge).  Camera orbit/pan/zoom
behaviour is inherited untouched.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6 import QtCore, QtGui

from ..units import Quantity, UnitSystem
from .geometry import box_lines

PICK_SLOP_PX = 8.0


class View3D(gl.GLViewWidget):
    #: emitted continuously while a charge or body is dragged
    entityMoved = QtCore.pyqtSignal(int)
    #: uid or None
    selectionChanged = QtCore.pyqtSignal(object)
    dragStarted = QtCore.pyqtSignal()
    dragFinished = QtCore.pyqtSignal()
    #: double-click on empty space, carries a world position (3,)
    addRequested = QtCore.pyqtSignal(object)
    #: emitted after every GL repaint, so 2-D overlays can follow the camera
    viewChanged = QtCore.pyqtSignal()
    #: uid under the cursor, or None
    hoverChanged = QtCore.pyqtSignal(object)
    #: a uid clicked while a non-select tool is active
    toolClicked = QtCore.pyqtSignal(int)

    def __init__(self, scene, parent=None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.domain = 1.0
        self.selected_uid: int | None = None
        self.hover_uid: int | None = None
        #: "select" | "charge" | "measure"; set by the main window's tool bar.
        self.tool = "select"

        self.setBackgroundColor(QtGui.QColor(18, 20, 26))
        self.setCameraPosition(distance=3.5, elevation=24, azimuth=35)
        self.setMouseTracking(True)

        self._decor: list = []
        self.axis_labels: list = []
        self._drag_uid: int | None = None
        self._drag_normal = np.array([0.0, 0.0, 1.0])
        self._drag_offset = np.zeros(3)
        self.rebuild_decorations(1.0, None)

    # ------------------------------------------------------------------
    # Scene decorations
    # ------------------------------------------------------------------
    def rebuild_decorations(self, domain: float, units: UnitSystem | None) -> None:
        self.domain = float(domain)
        for item in self._decor:
            self.removeItem(item)
        self._decor.clear()

        d = self.domain
        grid = gl.GLGridItem(color=(255, 255, 255, 40))
        grid.setSize(2 * d, 2 * d, 1)
        grid.setSpacing(d / 4.0, d / 4.0, 1)
        grid.translate(0, 0, -d)
        self._decor.append(grid)

        box = gl.GLLinePlotItem(pos=box_lines(d), mode="lines", width=1.0,
                                color=(1, 1, 1, 0.16), antialias=True)
        self._decor.append(box)

        axes = gl.GLLinePlotItem(
            pos=np.array([[-d, 0, -d], [d, 0, -d], [0, -d, -d], [0, d, -d]],
                         dtype=np.float32),
            mode="lines", width=1.4, color=(1, 1, 1, 0.30), antialias=True)
        self._decor.append(axes)

        self.axis_labels = []
        if units is not None:
            off = 0.05 * d
            for vec, name in (((d, 0.0, -d), "x"), ((0.0, d, -d), "y"),
                              ((0.0, 0.0, d), "z")):
                self.axis_labels.append(
                    (np.array(vec, dtype=float) + off,
                     f"{name} = {units.fmt(d, Quantity.LENGTH)}"))

        for item in self._decor:
            self.addItem(item)

    # ------------------------------------------------------------------
    # Camera maths
    # ------------------------------------------------------------------
    def _mvp(self) -> np.ndarray:
        vp = self.getViewport()
        m = self.projectionMatrix(vp, vp) * self.viewMatrix()
        return np.array(m.data(), dtype=float).reshape(4, 4).T

    def camera_forward(self) -> np.ndarray:
        cam = self.cameraPosition()
        ctr = self.opts["center"]
        f = np.array([ctr.x() - cam.x(), ctr.y() - cam.y(), ctr.z() - cam.z()])
        n = np.linalg.norm(f)
        return f / n if n > 0 else np.array([0.0, 0.0, -1.0])

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points -> (screen px ``(N, 2)``, eye depth ``(N,)``)."""
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        clip = np.c_[pts, np.ones(len(pts))] @ self._mvp().T
        w = clip[:, 3]
        safe = np.where(np.abs(w) > 1e-12, w, 1e-12)
        ndc = clip[:, :3] / safe[:, None]
        W, H = self.width(), self.height()
        sx = (ndc[:, 0] + 1.0) * 0.5 * W
        sy = (1.0 - ndc[:, 1]) * 0.5 * H
        return np.stack([sx, sy], axis=1), w

    def ray(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """Screen px -> (ray origin on the near plane, unit direction)."""
        inv = np.linalg.inv(self._mvp())
        W, H = max(self.width(), 1), max(self.height(), 1)
        nx, ny = 2.0 * x / W - 1.0, 1.0 - 2.0 * y / H

        def un(z: float) -> np.ndarray:
            p = inv @ np.array([nx, ny, z, 1.0])
            return p[:3] / p[3]

        near, far = un(-1.0), un(1.0)
        d = far - near
        n = np.linalg.norm(d)
        return near, (d / n if n > 0 else np.array([0.0, 0.0, -1.0]))

    @staticmethod
    def _plane_hit(origin, direction, point, normal):
        denom = float(np.dot(direction, normal))
        if abs(denom) < 1e-9:
            return None
        t = float(np.dot(point - origin, normal)) / denom
        if t < 0:
            return None
        return origin + t * direction

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------
    def pick(self, x: float, y: float) -> tuple[int | None, float]:
        """Topmost entity under the cursor, as ``(uid, eye depth)``.

        Point charges are matched in screen space, which keeps small spheres
        comfortably clickable; bodies are matched by true ray casting against
        their geometry.  Both are reduced to an eye-space depth so the two can
        be compared and the nearer one wins.
        """
        best_uid, best_depth = None, float("inf")

        charges = self.scene.charges
        if charges:
            pos = np.array([c.position for c in charges])
            screen, depth = self.project(pos)
            right = np.cross(self.camera_forward(), np.array([0.0, 0.0, 1.0]))
            n = np.linalg.norm(right)
            right = right / n if n > 0 else np.array([1.0, 0.0, 0.0])
            radii = np.array([c.radius for c in charges])
            edge, _ = self.project(pos + right[None, :] * radii[:, None])
            r_px = np.linalg.norm(edge - screen, axis=1) + PICK_SLOP_PX
            dist = np.linalg.norm(screen - np.array([x, y]), axis=1)
            hits = np.where((dist <= r_px) & (depth > 0))[0]
            if hits.size:
                k = int(hits[np.argmin(depth[hits])])
                best_uid, best_depth = charges[k].uid, float(depth[k])

        bodies = getattr(self.scene, "bodies", [])
        if bodies:
            origin, direction = self.ray(x, y)
            cam = self.cameraPosition()
            cam = np.array([cam.x(), cam.y(), cam.z()])
            fwd = self.camera_forward()
            for b in bodies:
                t = b.raycast(origin, direction)
                if t is None:
                    continue
                depth = float((origin + t * direction - cam) @ fwd)
                if 0 < depth < best_depth:
                    best_uid, best_depth = b.uid, depth
        return best_uid, best_depth

    def set_selected(self, uid: int | None) -> None:
        if uid != self.selected_uid:
            self.selected_uid = uid
            self.selectionChanged.emit(uid)

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------
    def mousePressEvent(self, ev) -> None:
        p = ev.position()
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            uid, _ = self.pick(p.x(), p.y())
            if self.tool != "select":
                # A tool consumes the click on an entity; clicking past
                # everything still orbits the camera.
                if uid is not None:
                    self.toolClicked.emit(uid)
                    ev.accept()
                    return
                super().mousePressEvent(ev)
                return
            self.set_selected(uid)
            if uid is not None:
                charge = self.scene.by_uid(uid)
                self._drag_uid = uid
                self._drag_normal = self._pick_plane_normal(ev.modifiers())
                origin, direction = self.ray(p.x(), p.y())
                hit = self._plane_hit(origin, direction, charge.position,
                                      self._drag_normal)
                self._drag_offset = (charge.position - hit) if hit is not None \
                    else np.zeros(3)
                self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                self.dragStarted.emit()
                ev.accept()
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        p = ev.position()
        if self._drag_uid is not None:
            charge = self.scene.by_uid(self._drag_uid)
            if charge is None:
                self._end_drag()
                return
            origin, direction = self.ray(p.x(), p.y())
            hit = self._plane_hit(origin, direction, charge.position,
                                  self._drag_normal)
            if hit is not None:
                target = hit + self._drag_offset
                if ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    # Vertical-only drag: keep x, y, take the new z.
                    target = np.array([charge.position[0], charge.position[1],
                                       target[2]])
                lim = 4.0 * self.domain
                charge.position = np.clip(target, -lim, lim)
                self.entityMoved.emit(self._drag_uid)
            ev.accept()
            return

        uid, _ = self.pick(p.x(), p.y())
        if uid != self.hover_uid:
            self.hover_uid = uid
            self.hoverChanged.emit(uid)
        if self.tool != "select":
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor if uid
                           else QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor if uid
                           else QtCore.Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        if self._drag_uid is not None:
            self._end_drag()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev) -> None:
        p = ev.position()
        if (ev.button() == QtCore.Qt.MouseButton.LeftButton
                and self.tool == "select"
                and self.pick(p.x(), p.y())[0] is None):
            origin, direction = self.ray(p.x(), p.y())
            ctr = self.opts["center"]
            hit = self._plane_hit(origin, direction,
                                  np.array([ctr.x(), ctr.y(), ctr.z()]),
                                  self.camera_forward())
            if hit is not None:
                lim = self.domain
                self.addRequested.emit(np.clip(hit, -lim, lim))
                ev.accept()
                return
        super().mouseDoubleClickEvent(ev)

    def _pick_plane_normal(self, modifiers) -> np.ndarray:
        f = self.camera_forward()
        if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
            # Plane containing world z and facing the camera as much as possible.
            n = np.array([f[0], f[1], 0.0])
            mag = np.linalg.norm(n)
            if mag > 1e-6:
                return n / mag
        return f

    def paintGL(self, *args, **kwargs) -> None:
        super().paintGL(*args, **kwargs)
        self.viewChanged.emit()

    def _end_drag(self) -> None:
        self._drag_uid = None
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.dragFinished.emit()
