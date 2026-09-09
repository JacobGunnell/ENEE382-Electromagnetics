"""2-D text overlay drawn on top of the 3-D viewport.

pyqtgraph's :class:`GLTextItem` paints with a ``QPainter`` in the middle of
``paintGL``, which silently draws nothing on some GL stacks.  Projecting the
anchor points ourselves and painting them in a sibling widget is both more
reliable and lets us add readable backing plates and collision avoidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets


@dataclass
class Label3D:
    pos: np.ndarray
    text: str
    color: tuple[int, int, int] = (235, 240, 250)
    plate: bool = True
    font_pt: float = 9.5


@dataclass
class LabelSet:
    items: list[Label3D] = field(default_factory=list)

    def add(self, pos, text, color=(235, 240, 250), **kw) -> None:
        self.items.append(Label3D(np.asarray(pos, dtype=float), text, color, **kw))


class LabelOverlay(QtWidgets.QWidget):
    """Transparent sibling of the GL view; paints labels at projected points."""

    def __init__(self, view, parent=None) -> None:
        super().__init__(parent)
        self.view = view
        self.labels: list[Label3D] = []
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

    def set_labels(self, labels: list[Label3D]) -> None:
        self.labels = labels
        self.update()

    def paintEvent(self, _ev) -> None:
        if not self.labels or self.view.width() < 2:
            return
        pts = np.array([l.pos for l in self.labels], dtype=float)
        screen, depth = self.view.project(pts)

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)

        placed: list[QtCore.QRectF] = []
        order = np.argsort(depth)                      # nearest first
        for i in order:
            if depth[i] <= 0:
                continue
            label = self.labels[i]
            font = p.font()
            font.setPointSizeF(label.font_pt)
            font.setBold(True)
            p.setFont(font)

            fm = QtGui.QFontMetricsF(font)
            w = fm.horizontalAdvance(label.text) + 8
            h = fm.height() + 2
            x = float(screen[i, 0]) - w / 2.0
            y = float(screen[i, 1]) - h / 2.0
            rect = QtCore.QRectF(x, y, w, h)
            rect = self._avoid(rect, placed)
            if not self.rect().adjusted(-4, -4, 4, 4).contains(rect.toRect()):
                if not self.rect().intersects(rect.toRect()):
                    continue
            placed.append(rect)

            if label.plate:
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(QtGui.QColor(10, 12, 18, 165))
                p.drawRoundedRect(rect, 3, 3)
            p.setPen(QtGui.QColor(*label.color))
            p.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, label.text)
        p.end()

    @staticmethod
    def _avoid(rect: QtCore.QRectF, placed: list[QtCore.QRectF]) -> QtCore.QRectF:
        """Nudge a label downward until it stops overlapping earlier ones."""
        for _ in range(12):
            hit = next((r for r in placed if r.intersects(rect)), None)
            if hit is None:
                return rect
            rect = rect.translated(0.0, hit.bottom() - rect.top() + 2.0)
        return rect


class ViewportStack(QtWidgets.QWidget):
    """Holds the GL view with the label overlay stacked on top of it."""

    def __init__(self, view, parent=None) -> None:
        super().__init__(parent)
        self.view = view
        self.overlay = LabelOverlay(view, self)
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(view, 0, 0)
        lay.addWidget(self.overlay, 0, 0)
        self.overlay.raise_()
        view.viewChanged.connect(self.overlay.update)

    def set_labels(self, labels) -> None:
        self.overlay.set_labels(labels)
