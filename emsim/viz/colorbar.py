"""A compact vertical colorbar widget driven by a :class:`ColorScale`."""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from ..units import UnitSystem
from . import colormaps as cmaps

BAR_W = 18
PAD_L = 8
PAD_TOP = 30
PAD_BOT = 10
TEXT_W = 74


class ColorBar(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scale = None
        self.units: UnitSystem | None = None
        self.setFixedWidth(PAD_L + BAR_W + 6 + TEXT_W)
        self.setMinimumHeight(150)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                           QtWidgets.QSizePolicy.Policy.Expanding)

    def set_scale(self, scale, units: UnitSystem) -> None:
        self.scale = scale
        self.units = units
        self.setVisible(scale is not None)
        self.update()

    def paintEvent(self, _ev) -> None:
        if self.scale is None or self.units is None:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)

        top, bottom = PAD_TOP, self.height() - PAD_BOT
        h = max(bottom - top, 1)
        table = cmaps.lut(self.scale.cmap)

        # gradient bar (t = 0 at the bottom)
        grad = QtGui.QLinearGradient(0, bottom, 0, top)
        for i in range(0, len(table), 8):
            r, g, b = table[i]
            grad.setColorAt(i / (len(table) - 1), QtGui.QColor(int(r), int(g), int(b)))
        r, g, b = table[-1]
        grad.setColorAt(1.0, QtGui.QColor(int(r), int(g), int(b)))

        rect = QtCore.QRectF(PAD_L, top, BAR_W, h)
        p.fillRect(rect, QtGui.QBrush(grad))
        p.setPen(QtGui.QPen(QtGui.QColor(150, 155, 170), 1))
        p.drawRect(rect)

        title_font = p.font()
        title_font.setPointSizeF(max(title_font.pointSizeF() - 0.5, 7.0))
        title_font.setBold(True)
        p.setFont(title_font)
        p.setPen(QtGui.QColor(225, 230, 240))
        p.drawText(QtCore.QRectF(0, 2, self.width(), PAD_TOP - 4),
                   QtCore.Qt.AlignmentFlag.AlignLeft
                   | QtCore.Qt.AlignmentFlag.AlignVCenter, self.scale.title)

        tick_font = p.font()
        tick_font.setBold(False)
        tick_font.setPointSizeF(max(tick_font.pointSizeF() - 0.5, 7.0))
        p.setFont(tick_font)

        x0 = PAD_L + BAR_W
        last_y = None
        for frac, value in self.scale.norm.ticks():
            y = bottom - frac * h
            if last_y is not None and abs(y - last_y) < 11:
                continue
            last_y = y
            p.setPen(QtGui.QPen(QtGui.QColor(150, 155, 170), 1))
            p.drawLine(QtCore.QPointF(x0, y), QtCore.QPointF(x0 + 4, y))
            p.setPen(QtGui.QColor(205, 212, 225))
            label = self.units.fmt_bare(value, self.scale.quantity, sig=2)
            ty = min(max(y - 8, 2.0), self.height() - 18.0)
            p.drawText(QtCore.QRectF(x0 + 6, ty, TEXT_W, 16),
                       QtCore.Qt.AlignmentFlag.AlignLeft
                       | QtCore.Qt.AlignmentFlag.AlignVCenter, label)
        p.end()


class ColorBarColumn(QtWidgets.QWidget):
    """Stacks however many colorbars the active layers ask for."""

    def __init__(self, parent=None, count: int = 2) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 6, 2, 6)
        lay.setSpacing(14)
        self.bars = [ColorBar(self) for _ in range(count)]
        for b in self.bars:
            lay.addWidget(b)
            b.setVisible(False)
        lay.addStretch(0)
        self.setFixedWidth(PAD_L + BAR_W + 6 + TEXT_W + 6)

    def set_scales(self, scales, units: UnitSystem) -> None:
        for i, bar in enumerate(self.bars):
            bar.set_scale(scales[i] if i < len(scales) else None, units)
        self.setVisible(bool(scales))
