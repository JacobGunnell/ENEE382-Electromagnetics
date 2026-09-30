"""Main window: viewport + colorbars + control panel, and the update loop."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ..core import BODY_TYPES, Material, PointCharge, Scene
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
#: Seconds for deposited charge to settle onto a conductor's surface.  Long
#: enough to watch, short enough not to be in the way.
RELAX_SECONDS = 0.75
RELAX_MS = 16
#: Conductor sites are coarsened this much while a body is dragged, because
#: moving a body forces the O(N^3) factorisation to be rebuilt every frame.
DRAG_SITE_SCALE = 0.45


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

        self._relax = QtCore.QTimer(self)
        self._relax.setInterval(RELAX_MS)
        self._relax.timeout.connect(self._step_relaxation)

        self.measure_pair: list[int] = []
        self._connect()
        self.view.rebuild_decorations(self.settings.domain, self.settings.units)
        self.load_demo()
        self.panel.sync_bodies()
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

        p.toolChanged.connect(self._on_tool_changed)
        p.addBodyRequested.connect(self.add_body)
        p.deleteBodyRequested.connect(self.delete_selected)
        p.clearMeasureRequested.connect(self.clear_measurement)
        p.exampleRequested.connect(self.load_example)

        v.entityMoved.connect(self._on_charge_moved)
        v.hoverChanged.connect(self._on_hover)
        v.toolClicked.connect(self._on_tool_clicked)
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

    def add_body(self, name: str) -> None:
        d = self.settings.domain
        cls = BODY_TYPES[name]
        body = cls(position=np.zeros(3), n_sites=220)
        # Scale the default geometry to the region of interest.
        for attr, _label in body.params:
            setattr(body, attr, getattr(body, attr) * d)
        if name in ("Line", "Loop"):
            body.wire_radius = 0.012 * d
        # Offset so a second body of the same kind does not land on the first.
        same = [b for b in self.scene.bodies if b.kind == body.kind]
        if same:
            body.position = np.array([0.0, 0.0, (len(same) % 2 * 2 - 1)
                                      * 0.35 * d])
        self.scene.add_body(body)
        self.view.set_selected(body.uid)
        self.panel.sync_bodies(body.uid)
        self.request_redraw()

    def load_example(self, name: str) -> None:
        """Preset scenes, so each feature has somewhere obvious to start."""
        from ..core import Disk, Sphere
        self.scene.clear()
        self.clear_measurement()
        d = self.settings.domain
        if name == "Parallel plates":
            for z, q in ((-0.18 * d, 4e-9), (0.18 * d, -4e-9)):
                pl = Disk(radius=0.45 * d, position=[0, 0, z], axis=[0, 0, 1],
                          n_sites=320)
                pl.charge = q
                self.scene.bodies.append(pl)
            self.panel.tool_buttons.button(2).setChecked(True)
            self.panel._on_tool(2)
            self.measure_pair = [b.uid for b in self.scene.bodies]
        elif name == "Sphere + point charge":
            b = Sphere(radius=0.3 * d, position=[-0.25 * d, 0, 0], n_sites=320)
            self.scene.bodies.append(b)
            self.scene.charges.append(PointCharge(
                q=12e-9, position=np.array([0.7 * d, 0.0, 0.0]),
                radius=radius_for_charge(12e-9, d)))
        elif name == "Insulating ball":
            b = Sphere(radius=0.35 * d, position=np.zeros(3), n_sites=400,
                       material=Material.INSULATOR)
            b.charge = 8e-9
            self.scene.bodies.append(b)
        else:
            for q, x in ((10e-9, -0.35 * d), (-10e-9, 0.35 * d)):
                self.scene.charges.append(PointCharge(
                    q=q, position=np.array([x, 0.0, 0.0]),
                    radius=radius_for_charge(q, d)))
        self.scene.notify("example")
        self.panel.sync_table()
        self.panel.sync_bodies()
        self.request_redraw()

    def delete_selected(self) -> None:
        uid = self.view.selected_uid
        if uid is None:
            return
        if self.scene.body(uid) is not None:
            self.scene.remove_body(uid)
            self.measure_pair = [u for u in self.measure_pair if u != uid]
        else:
            self.scene.remove_charge(uid)
        self.view.set_selected(None)
        self.panel.sync_table()
        self.panel.sync_bodies()
        self.request_redraw()

    def clear_charges(self) -> None:
        self.scene.clear()
        self.view.set_selected(None)
        self.clear_measurement()
        self.panel.sync_table()
        self.panel.sync_bodies()
        self.request_redraw()

    def fit_force_scale(self) -> None:
        """One-shot recalibration of the (otherwise fixed) force arrow scale.

        Covers bodies as well as point charges -- a scene of two capacitor
        plates has no point charges at all, and Fit did nothing in it.
        """
        mags = list(np.linalg.norm(self.scene.forces(), axis=1))
        mags += [float(np.linalg.norm(v))
                 for v in self.scene.body_forces().values()]
        if not mags:
            return
        self.panel.fit_force_gain(np.array(mags))

    def reset_view(self) -> None:
        d = self.settings.domain
        self.view.opts["center"] = QtGui.QVector3D(0, 0, 0)
        self.view.setCameraPosition(distance=3.5 * d, elevation=24, azimuth=35)
        self.view.update()

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------
    def _on_tool_changed(self, tool: str) -> None:
        self.view.tool = tool
        msg = {"select": "Drag charges and bodies · Shift-drag for z · "
                         "double-click empty space to add a charge",
               "charge": "Click a body to deposit charge on it. On a "
                         "conductor it will migrate to the surface.",
               "measure": "Click two conductors to read ΔV and C between "
                          "them."}
        self.status.showMessage(msg.get(tool, ""))

    def _on_tool_clicked(self, uid: int) -> None:
        tool = self.panel.current_tool()
        if tool == "charge":
            if self.scene.body(uid) is None:
                self.status.showMessage("Charge can only be deposited on a "
                                        "body, not on a point charge.", 4000)
                return
            self.scene.deposit_charge(uid, self.panel.deposit_amount())
            self.panel.sync_bodies(self.view.selected_uid)
            self._relax.start()
        elif tool == "measure":
            body = self.scene.body(uid)
            if body is None or not body.is_conductor:
                self.status.showMessage("Voltage and capacitance are defined "
                                        "between two conductors.", 4000)
                return
            if uid in self.measure_pair:
                self.measure_pair.remove(uid)
            else:
                self.measure_pair.append(uid)
                self.measure_pair = self.measure_pair[-2:]
        self.request_redraw()

    def clear_measurement(self) -> None:
        self.measure_pair = []
        self.panel.show_measurement(None)
        self.request_redraw()

    def _on_hover(self, uid) -> None:
        """Repaint the hover highlight only -- never re-solve for a mouse move."""
        self.settings.hover_uid = uid
        self.layers.bodies.apply_highlight(
            self.scene, uid, self.view.selected_uid, self.settings.measure_pair)
        self.view.update()

    def _step_relaxation(self) -> None:
        step = RELAX_MS / 1000.0 / RELAX_SECONDS
        busy = False
        for b in self.scene.bodies:
            if b.relax_t < 1.0:
                b.relax_t = min(1.0, b.relax_t + step)
                busy = busy or b.relax_t < 1.0
        self.settings.quality = DRAG_QUALITY if busy else 1.0
        self.scene.invalidate()
        self._redraw()
        if not busy:
            self._relax.stop()

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
        self.scene.invalidate()
        if self.scene.body(uid) is not None:
            self.panel.sync_bodies(uid)
        else:
            self.panel.sync_table(uid)
        self.panel.update_readout(uid)
        self.request_redraw()

    def _on_select_from_view(self, uid) -> None:
        self.settings.selected_uid = uid
        self.panel.select_uid(uid)
        self.panel.select_body(uid)
        self.panel.show_body_editor(uid)
        self.panel.update_readout(uid)
        self.request_redraw()

    def _on_select_from_table(self, uid) -> None:
        self.view.set_selected(uid)

    def _on_drag_started(self) -> None:
        self.settings.quality = DRAG_QUALITY
        self._timer.setInterval(0)
        if self.scene.body(self.view.selected_uid) is not None:
            self.scene.site_scale = DRAG_SITE_SCALE

    def _on_drag_finished(self) -> None:
        self.settings.quality = 1.0
        self.scene.site_scale = 1.0
        self._timer.setInterval(REDRAW_MS)
        self.request_redraw()

    # ------------------------------------------------------------------
    def request_redraw(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def _redraw(self) -> None:
        self.scene.invalidate()
        self.settings.selected_uid = self.view.selected_uid
        self.settings.hover_uid = self.view.hover_uid
        self.settings.measure_pair = (tuple(self.measure_pair)
                                      if len(self.measure_pair) == 2 else None)
        self.layers.update(self.scene, self.settings)
        self.panel.show_measurement(
            self.scene.measure(*self.settings.measure_pair)
            if self.settings.measure_pair else None)
        self.bars.set_scales(self.layers.scales(), self.settings.units)

        labels = list(self.layers.labels())
        labels += [Label3D(pos, text, (150, 162, 185), plate=False, font_pt=8.5)
                   for pos, text in self.view.axis_labels]
        self.viewport.set_labels(labels)
        self.view.update()

