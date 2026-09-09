"""Right-hand control panel: visualization toggles, units, and the charge list."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ..core.entities import radius_for_charge
from ..units import UNIT_SYSTEMS, Quantity, UnitSystem
from ..viz import colormaps as cmaps
from ..viz.layers import RenderSettings, force_gain_for

COL_LABEL, COL_Q, COL_X, COL_Y, COL_Z = range(5)

#: Force-gain slider: value v maps to a gain of 10**(v / GAIN_DECADE),
#: spanning 1e-6 to 1e+6.
GAIN_DECADE = 20.0
GAIN_SLIDER_RANGE = (-120, 120)


def _group(title: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QVBoxLayout]:
    box = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(10, 8, 10, 10)
    lay.setSpacing(6)
    return box, lay


def _row(label: str, widget: QtWidgets.QWidget, indent: int = 14
         ) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(indent, 0, 0, 0)
    lay.setSpacing(6)
    lab = QtWidgets.QLabel(label)
    lab.setMinimumWidth(94)
    lay.addWidget(lab)
    lay.addWidget(widget, 1)
    return w


def _slider(lo: int, hi: int, val: int) -> QtWidgets.QSlider:
    s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    return s


class ControlPanel(QtWidgets.QWidget):
    settingsChanged = QtCore.pyqtSignal()
    unitsChanged = QtCore.pyqtSignal()
    domainChanged = QtCore.pyqtSignal()
    sceneEdited = QtCore.pyqtSignal()
    selectRequested = QtCore.pyqtSignal(object)
    addRequested = QtCore.pyqtSignal()
    deleteRequested = QtCore.pyqtSignal()
    clearRequested = QtCore.pyqtSignal()
    resetViewRequested = QtCore.pyqtSignal()
    forceFitRequested = QtCore.pyqtSignal()

    def __init__(self, scene, settings: RenderSettings, parent=None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.st = settings
        self._syncing = False

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        body = QtWidgets.QWidget()
        self.v = QtWidgets.QVBoxLayout(body)
        self.v.setContentsMargins(10, 10, 10, 10)
        self.v.setSpacing(10)
        scroll.setWidget(body)

        self._build_units()
        self._build_force()
        self._build_efield()
        self._build_potential()
        self._build_charges()
        self._build_readout()
        self.v.addStretch(1)

        self.setMinimumWidth(372)
        self._update_enabled()
        self._update_gain_label()
        self.refresh_units()

    # ------------------------------------------------------------------
    @property
    def units(self) -> UnitSystem:
        return self.st.units

    def _emit(self) -> None:
        if not self._syncing:
            self.settingsChanged.emit()

    # ------------------------------------------------------------------
    def _build_units(self) -> None:
        box, lay = _group("Units && region")

        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(list(UNIT_SYSTEMS))
        self.unit_combo.setCurrentText(self.st.units.name)
        self.unit_combo.currentTextChanged.connect(self._on_units)
        lay.addWidget(_row("System", self.unit_combo, 0))

        self.domain_spin = QtWidgets.QDoubleSpinBox()
        self.domain_spin.setRange(0.001, 1e9)
        self.domain_spin.setDecimals(3)
        self.domain_spin.setValue(self.st.domain)
        self.domain_spin.valueChanged.connect(self._on_domain)
        self.domain_row = _row("Half-width", self.domain_spin, 0)
        lay.addWidget(self.domain_row)

        btn = QtWidgets.QPushButton("Reset view")
        btn.clicked.connect(self.resetViewRequested.emit)
        lay.addWidget(btn)
        self.v.addWidget(box)

    def _build_force(self) -> None:
        box, lay = _group("Force on charges")
        self.cb_force = QtWidgets.QCheckBox("Show force arrows")
        self.cb_force.setChecked(self.st.show_force)
        self.cb_force.toggled.connect(self._on_toggle)
        lay.addWidget(self.cb_force)

        # Log slider: the force scale is absolute, so a run of charges at an
        # unusual length or charge scale can need many decades of adjustment.
        self.force_gain = _slider(*GAIN_SLIDER_RANGE, 0)
        self.force_gain.valueChanged.connect(self._on_toggle)
        self.force_gain_label = QtWidgets.QLabel()
        self.force_gain_label.setMinimumWidth(60)
        self.force_fit = QtWidgets.QPushButton("Fit")
        self.force_fit.setMaximumWidth(44)
        self.force_fit.setToolTip(
            "Set the gain once so the largest force now present draws at the "
            "reference length. The scale stays fixed afterwards.")
        self.force_fit.clicked.connect(self.forceFitRequested.emit)
        holder = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        hl.addWidget(self.force_gain, 1)
        hl.addWidget(self.force_gain_label)
        hl.addWidget(self.force_fit)
        self.force_row = _row("Arrow scale", holder)
        lay.addWidget(self.force_row)

        self.cb_force_lbl = QtWidgets.QCheckBox("Numeric labels")
        self.cb_force_lbl.setChecked(self.st.force_labels)
        self.cb_force_lbl.toggled.connect(self._on_toggle)
        self.force_lbl_row = _row("", self.cb_force_lbl)
        lay.addWidget(self.force_lbl_row)

        note = QtWidgets.QLabel(
            "Fixed scale: arrow length is proportional to |F|, so arrows grow "
            "and shrink as you move charges. At gain 1x, 0.30 × half-width is "
            "the force between two 10 nC charges 0.7 m apart.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #8a93a6; font-size: 10px;")
        lay.addWidget(note)
        self.v.addWidget(box)

    def _build_efield(self) -> None:
        box, lay = _group("Electric field")
        self.cb_efield = QtWidgets.QCheckBox("Show vector field")
        self.cb_efield.setChecked(self.st.show_efield)
        self.cb_efield.toggled.connect(self._on_toggle)
        lay.addWidget(self.cb_efield)

        self.field_grid = QtWidgets.QSpinBox()
        self.field_grid.setRange(3, 21)
        self.field_grid.setValue(self.st.field_grid)
        self.field_grid.valueChanged.connect(self._on_toggle)
        self.field_rows = [_row("Grid n³", self.field_grid)]

        self.field_len = _slider(20, 100, int(self.st.field_len_frac * 100))
        self.field_len.valueChanged.connect(self._on_toggle)
        self.field_rows.append(_row("Glyph size", self.field_len))

        self.cb_field_log = QtWidgets.QCheckBox("Log colour scale")
        self.cb_field_log.setChecked(self.st.field_log)
        self.cb_field_log.toggled.connect(self._on_toggle)
        self.field_rows.append(_row("", self.cb_field_log))

        self.field_cmap = QtWidgets.QComboBox()
        self.field_cmap.addItems(list(cmaps.SEQUENTIAL))
        self.field_cmap.setCurrentText(self.st.field_cmap)
        self.field_cmap.currentTextChanged.connect(self._on_toggle)
        self.field_rows.append(_row("Colormap", self.field_cmap))

        for r in self.field_rows:
            lay.addWidget(r)
        self.v.addWidget(box)

    def _build_potential(self) -> None:
        box, lay = _group("Electric potential")
        self.cb_pot = QtWidgets.QCheckBox("Show heatmap")
        self.cb_pot.setChecked(self.st.show_potential)
        self.cb_pot.toggled.connect(self._on_toggle)
        lay.addWidget(self.cb_pot)

        self.pot_mode = QtWidgets.QComboBox()
        self.pot_mode.addItems(["Volume", "Slice plane"])
        self.pot_mode.setCurrentIndex(0 if self.st.pot_mode == "volume" else 1)
        self.pot_mode.currentIndexChanged.connect(self._on_toggle)
        self.pot_rows = [_row("Mode", self.pot_mode)]

        self.pot_res = QtWidgets.QSpinBox()
        self.pot_res.setRange(12, 96)
        self.pot_res.setSingleStep(4)
        self.pot_res.setValue(self.st.pot_res)
        self.pot_res.valueChanged.connect(self._on_toggle)
        self.pot_rows.append(_row("Resolution", self.pot_res))

        self.pot_alpha = _slider(5, 100, int(self.st.pot_alpha * 100))
        self.pot_alpha.valueChanged.connect(self._on_toggle)
        self.pot_rows.append(_row("Opacity", self.pot_alpha))

        self.cb_pot_symlog = QtWidgets.QCheckBox("Symmetric-log scale")
        self.cb_pot_symlog.setChecked(self.st.pot_symlog)
        self.cb_pot_symlog.toggled.connect(self._on_toggle)
        self.pot_rows.append(_row("", self.cb_pot_symlog))

        self.pot_cmap = QtWidgets.QComboBox()
        self.pot_cmap.addItems(list(cmaps.DIVERGING))
        self.pot_cmap.setCurrentText(self.st.pot_cmap)
        self.pot_cmap.currentTextChanged.connect(self._on_toggle)
        self.pot_rows.append(_row("Colormap", self.pot_cmap))

        self.slice_axis = QtWidgets.QComboBox()
        self.slice_axis.addItems(["x = const", "y = const", "z = const"])
        self.slice_axis.setCurrentIndex(self.st.slice_axis)
        self.slice_axis.currentIndexChanged.connect(self._on_toggle)
        self.slice_rows = [_row("Cut plane", self.slice_axis)]

        self.slice_pos = _slider(-100, 100, 0)
        self.slice_pos.valueChanged.connect(self._on_toggle)
        self.slice_pos_label = QtWidgets.QLabel("0")
        self.slice_pos_label.setMinimumWidth(58)
        holder = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.slice_pos, 1)
        hl.addWidget(self.slice_pos_label)
        self.slice_rows.append(_row("Position", holder))

        for r in self.pot_rows + self.slice_rows:
            lay.addWidget(r)
        self.v.addWidget(box)

    def _build_charges(self) -> None:
        box, lay = _group("Charges")
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setMinimumHeight(150)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(COL_LABEL,
                                QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(COL_LABEL, 34)
        hh.setMinimumSectionSize(34)
        self.table.itemChanged.connect(self._on_table_edit)
        self.table.itemSelectionChanged.connect(self._on_table_select)
        lay.addWidget(self.table)

        btns = QtWidgets.QHBoxLayout()
        for text, sig in (("Add", self.addRequested),
                          ("Delete", self.deleteRequested),
                          ("Clear", self.clearRequested)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(sig.emit)
            btns.addWidget(b)
        lay.addLayout(btns)

        hint = QtWidgets.QLabel(
            "Drag a charge to move it in the view plane · Shift-drag for z · "
            "double-click empty space to add · Del to remove")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8a93a6; font-size: 10px;")
        lay.addWidget(hint)
        self.v.addWidget(box)

    def _build_readout(self) -> None:
        box, lay = _group("Selected charge")
        self.readout = QtWidgets.QLabel("—")
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.readout.setWordWrap(True)
        self.readout.setStyleSheet("font-family: monospace; font-size: 11px;")
        lay.addWidget(self.readout)
        self.v.addWidget(box)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    def _on_units(self, name: str) -> None:
        self.st.units = UNIT_SYSTEMS[name]
        self.refresh_units()
        self.unitsChanged.emit()

    def _on_domain(self, _v: float) -> None:
        if self._syncing:
            return
        self.st.domain = self.units.from_entry(self.domain_spin.value(),
                                               Quantity.LENGTH)
        # The cut plane is positioned as a fraction of the domain.
        self.st.slice_pos = self.st.domain * self.slice_pos.value() / 100.0
        self._update_slice_label()
        self.domainChanged.emit()

    def _on_toggle(self, *_a) -> None:
        st = self.st
        st.show_force = self.cb_force.isChecked()
        st.force_gain = 10.0 ** (self.force_gain.value() / GAIN_DECADE)
        st.force_labels = self.cb_force_lbl.isChecked()

        st.show_efield = self.cb_efield.isChecked()
        st.field_grid = self.field_grid.value()
        st.field_len_frac = self.field_len.value() / 100.0
        st.field_log = self.cb_field_log.isChecked()
        st.field_cmap = self.field_cmap.currentText()

        st.show_potential = self.cb_pot.isChecked()
        st.pot_mode = "volume" if self.pot_mode.currentIndex() == 0 else "slice"
        st.pot_res = self.pot_res.value()
        st.pot_alpha = self.pot_alpha.value() / 100.0
        st.pot_symlog = self.cb_pot_symlog.isChecked()
        st.pot_cmap = self.pot_cmap.currentText()
        st.slice_axis = self.slice_axis.currentIndex()
        st.slice_pos = st.domain * self.slice_pos.value() / 100.0

        self._update_enabled()
        self._update_slice_label()
        self._update_gain_label()
        self._emit()

    def _update_enabled(self) -> None:
        on = self.cb_force.isChecked()
        for w in (self.force_row, self.force_lbl_row):
            w.setEnabled(on)
        on = self.cb_efield.isChecked()
        for w in self.field_rows:
            w.setEnabled(on)
        on = self.cb_pot.isChecked()
        for w in self.pot_rows:
            w.setEnabled(on)
        is_slice = self.pot_mode.currentIndex() == 1
        for w in self.slice_rows:
            w.setEnabled(on and is_slice)
            w.setVisible(is_slice)

    def _update_gain_label(self) -> None:
        g = self.st.force_gain
        text = f"{g:.2f}×" if 0.01 <= g < 1000 else f"{g:.0e}×"
        self.force_gain_label.setText(text)

    def set_force_gain(self, gain: float) -> None:
        """Move the gain slider to ``gain`` (used by the Fit button)."""
        v = int(round(GAIN_DECADE * np.log10(max(gain, 1e-30))))
        self.force_gain.setValue(int(np.clip(v, *GAIN_SLIDER_RANGE)))

    def fit_force_gain(self, magnitudes) -> None:
        self.set_force_gain(force_gain_for(magnitudes))

    def _update_slice_label(self) -> None:
        self.slice_pos_label.setText(
            self.units.fmt(self.st.slice_pos, Quantity.LENGTH, sig=2))

    # -- units -------------------------------------------------------------
    def refresh_units(self) -> None:
        """Re-label every unit-bearing widget and re-express its value."""
        self._syncing = True
        u = self.units
        qs, ls = u.entry_symbol(Quantity.CHARGE), u.entry_symbol(Quantity.LENGTH)
        self.table.setHorizontalHeaderLabels(
            ["", f"q ({qs})", f"x ({ls})", f"y ({ls})", f"z ({ls})"])

        self.domain_row.layout().itemAt(0).widget().setText(f"Half-width ({ls})")
        self.domain_spin.setValue(u.to_entry(self.st.domain, Quantity.LENGTH))
        self.domain_spin.setSingleStep(
            max(u.to_entry(self.st.domain, Quantity.LENGTH) / 10.0, 1e-6))
        self._syncing = False
        self._update_slice_label()
        self.sync_table()

    # -- charge table ------------------------------------------------------
    def sync_table(self, keep_uid: int | None = None) -> None:
        self._syncing = True
        u = self.units
        self.table.setRowCount(len(self.scene.charges))
        for r, c in enumerate(self.scene.charges):
            vals = [c.label,
                    f"{u.to_entry(c.q, Quantity.CHARGE):.4g}",
                    *[f"{u.to_entry(x, Quantity.LENGTH):.4g}" for x in c.position]]
            for col, text in enumerate(vals):
                item = self.table.item(r, col)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    if col == COL_LABEL:
                        item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled
                                      | QtCore.Qt.ItemFlag.ItemIsSelectable)
                    self.table.setItem(r, col, item)
                item.setText(text)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, c.uid)
                if col == COL_Q:
                    item.setForeground(QtGui.QColor(
                        "#f0594f" if c.q > 0 else "#4f8ef0" if c.q < 0 else "#bbb"))
        if keep_uid is not None:
            self.select_uid(keep_uid)
        self._syncing = False

    def select_uid(self, uid: int | None) -> None:
        """Select a row without emitting :attr:`selectRequested` back out."""
        prev = self._syncing
        self._syncing = True
        self.table.clearSelection()
        if uid is not None:
            for r in range(self.table.rowCount()):
                it = self.table.item(r, COL_LABEL)
                if it is not None and it.data(QtCore.Qt.ItemDataRole.UserRole) == uid:
                    self.table.selectRow(r)
                    break
        self._syncing = prev

    def _on_table_select(self) -> None:
        if self._syncing:
            return
        rows = self.table.selectionModel().selectedRows()
        uid = None
        if rows:
            it = self.table.item(rows[0].row(), COL_LABEL)
            uid = it.data(QtCore.Qt.ItemDataRole.UserRole) if it else None
        self.selectRequested.emit(uid)

    def _on_table_edit(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._syncing:
            return
        uid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        charge = self.scene.by_uid(uid)
        if charge is None:
            return
        try:
            value = float(item.text())
        except ValueError:
            self.sync_table(uid)
            return
        col = item.column()
        if col == COL_Q:
            charge.q = self.units.from_entry(value, Quantity.CHARGE)
            charge.radius = radius_for_charge(charge.q, self.st.domain)
        elif col in (COL_X, COL_Y, COL_Z):
            charge.position[col - COL_X] = self.units.from_entry(
                value, Quantity.LENGTH)
        self.sceneEdited.emit()

    # -- readout -----------------------------------------------------------
    def update_readout(self, uid: int | None) -> None:
        charge = self.scene.by_uid(uid) if uid is not None else None
        if charge is None:
            self.readout.setText("<span style='color:#8a93a6'>"
                                 "No charge selected</span>")
            return
        u = self.units
        sample = self.scene.evaluate(charge.position[None, :],
                                     exclude_uid=charge.uid)
        E, V = sample["E"][0], float(sample["V"][0])
        F = charge.q * E
        rows = [
            ("q", u.fmt(charge.q, Quantity.CHARGE)),
            ("r", "(" + ", ".join(u.fmt(x, Quantity.LENGTH)
                                  for x in charge.position) + ")"),
            ("|F|", u.fmt(float(np.linalg.norm(F)), Quantity.FORCE)),
            ("|E|", u.fmt(float(np.linalg.norm(E)), Quantity.EFIELD)),
            ("V", u.fmt(V, Quantity.POTENTIAL)),
        ]
        cells = "".join(
            f"<tr><td style='color:#9aa4b8;padding-right:8px'>{k}</td>"
            f"<td style='color:#e8edf6'>{v}</td></tr>" for k, v in rows)
        self.readout.setText(f"<table cellspacing='0'>{cells}</table>")


