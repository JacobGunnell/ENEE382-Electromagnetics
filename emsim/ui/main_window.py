"""Main window: viewport + colorbars + control panel, and the update loop."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ..core import PointCharge, Scene
from ..core.entities import radius_for_charge
from ..units import SI, Quantity
from ..viz.colorbar import ColorBarColumn
from ..viz.layers import LayerStack, RenderSettings
from ..viz.overlay import Label3D, ViewportStack
from ..viz.view3d import View3D
from .controls import ControlPanel

#: While dragging, rebuild the field/volume at reduced resolution so the
#: interaction stays at interactive frame rates on modest hardware.
DRAG_QUALITY = 0.6
REDRAW_MS = 16


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("emsim — Coulomb field, potential and force")
        self.resize(1440, 900)

        self.scene = Scene()
        self.settings = RenderSettings(units=SI, domain=1.0)

        self.view = View3D(self.scene)
        self.viewport = ViewportStack(self.view)
        self.layers = LayerStack(self.view)
        self.bars = ColorBarColumn(count=2)
        self.panel = ControlPanel(self.scene, self.settings)

        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.viewport, 1)
        lay.addWidget(self.bars)
        lay.addWidget(self.panel)
        self.setCentralWidget(central)

        self.status = self.statusBar()
        self.status.showMessage("Drag charges to move them · Shift-drag for z "
                                "· double-click empty space to add")

        # Coalesce redraws: many signals can fire per mouse move.
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REDRAW_MS)
        self._timer.timeout.connect(self._redraw)

        self._connect()
        self.view.rebuild_decorations(self.settings.domain, self.settings.units)
        self.load_demo()
        self._install_shortcuts()

    # ------------------------------------------------------------------
    def _connect(self) -> None:
        p, v = self.panel, self.view
        p.settingsChanged.connect(self.request_redraw)
        p.unitsChanged.connect(self._on_units)
        p.domainChanged.connect(self._on_domain)
        p.sceneEdited.connect(self._on_scene_edited)
        p.selectRequested.connect(self._on_select_from_table)
        p.addRequested.connect(lambda: self.add_charge())
        p.deleteRequested.connect(self.delete_selected)
        p.clearRequested.connect(self.clear_charges)
        p.resetViewRequested.connect(self.reset_view)
        p.forceFitRequested.connect(self.fit_force_scale)

        v.chargeMoved.connect(self._on_charge_moved)
        v.selectionChanged.connect(self._on_select_from_view)
        v.dragStarted.connect(self._on_drag_started)
        v.dragFinished.connect(self._on_drag_finished)
        v.addRequested.connect(self.add_charge)

        self.scene.subscribe(lambda _reason: self.request_redraw())

    def _install_shortcuts(self) -> None:
        for key in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(self.delete_selected)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+N"), self,
                        activated=lambda: self.add_charge())
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self,
                        activated=self.reset_view)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def load_demo(self) -> None:
        """A dipole: the smallest configuration that exercises every layer."""
        for q, pos in ((10e-9, (-0.35, 0.0, 0.0)), (-10e-9, (0.35, 0.0, 0.0))):
            self.scene.charges.append(
                PointCharge(q=q, position=np.array(pos),
                            radius=radius_for_charge(q, self.settings.domain)))
        self.panel.sync_table()
        self.request_redraw()

    def add_charge(self, position=None) -> None:
        d = self.settings.domain
        if position is None:
            position = np.array([0.0, 0.0, 0.0])
            if self.scene.charges:
                rng = np.random.default_rng()
                position = rng.uniform(-0.55 * d, 0.55 * d, 3)
        q = 10e-9
        charge = PointCharge(q=q, position=np.asarray(position, dtype=float),
                             radius=radius_for_charge(q, d))
        self.scene.add_charge(charge)
        self.view.set_selected(charge.uid)
        self.panel.sync_table(charge.uid)
        self.request_redraw()

    def delete_selected(self) -> None:
        uid = self.view.selected_uid
        if uid is None:
            return
        self.scene.remove_charge(uid)
        self.view.set_selected(None)
        self.panel.sync_table()
        self.request_redraw()

    def clear_charges(self) -> None:
        self.scene.clear()
        self.view.set_selected(None)
        self.panel.sync_table()
        self.request_redraw()

    def fit_force_scale(self) -> None:
        """One-shot recalibration of the (otherwise fixed) force arrow scale."""
        F = self.scene.forces()
        if len(F) == 0:
            return
        self.panel.fit_force_gain(np.linalg.norm(F, axis=1))

    def reset_view(self) -> None:
        d = self.settings.domain
        self.view.opts["center"] = QtGui.QVector3D(0, 0, 0)
        self.view.setCameraPosition(distance=3.5 * d, elevation=24, azimuth=35)
        self.view.update()

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------
    def _on_units(self) -> None:
        self.view.rebuild_decorations(self.settings.domain, self.settings.units)
        self.panel.update_readout(self.view.selected_uid)
        self.request_redraw()

    def _on_domain(self) -> None:
        d = self.settings.domain
        for c in self.scene.charges:
            c.radius = radius_for_charge(c.q, d)
        self.view.rebuild_decorations(d, self.settings.units)
        self.reset_view()
        self.request_redraw()

    def _on_scene_edited(self) -> None:
        self.panel.update_readout(self.view.selected_uid)
        self.request_redraw()

    def _on_charge_moved(self, uid: int) -> None:
        self.panel.sync_table(uid)
        self.panel.update_readout(uid)
        self.request_redraw()

    def _on_select_from_view(self, uid) -> None:
        self.settings.selected_uid = uid
        self.panel.select_uid(uid)
        self.panel.update_readout(uid)
        self.request_redraw()

    def _on_select_from_table(self, uid) -> None:
        self.view.set_selected(uid)

    def _on_drag_started(self) -> None:
        self.settings.quality = DRAG_QUALITY
        self._timer.setInterval(0)

    def _on_drag_finished(self) -> None:
        self.settings.quality = 1.0
        self._timer.setInterval(REDRAW_MS)
        self.request_redraw()

    # ------------------------------------------------------------------
    def request_redraw(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def _redraw(self) -> None:
        self.settings.selected_uid = self.view.selected_uid
        self.layers.update(self.scene, self.settings)
        self.bars.set_scales(self.layers.scales(), self.settings.units)

        labels = list(self.layers.labels())
        labels += [Label3D(pos, text, (150, 162, 185), plate=False, font_pt=8.5)
                   for pos, text in self.view.axis_labels]
        self.viewport.set_labels(labels)
        self.view.update()

