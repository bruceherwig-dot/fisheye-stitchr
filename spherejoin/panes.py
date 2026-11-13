"""A source image pane: viewer, circle tool, numeric fields."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from .circle import auto_detect_circle
from . import ui_state
from .collapsible import CollapsibleSection
from .imageio_utils import SUPPORTED_EXTENSIONS, load_image, open_failure_text
from .viewer import ZoomPanView


class ImagePane(QWidget):
    circle_changed = Signal()   # circle edited on screen, in the fields, or by auto-detect
    image_loaded = Signal()
    files_dropped = Signal(list)   # picture files dropped on this pane, sorted by name

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.path: str | None = None
        self._array = None
        self._cursor = (-1, -1)

        self.setAcceptDrops(True)
        self.header = QLabel(f"<b>{title}</b>: no image loaded")
        self.hint = QLabel("drop another photo here to replace it")
        self.hint.setStyleSheet("color: #8a8a8a; font-size: 11px;")
        self.hint.hide()
        self.fit_btn = QPushButton("Fit")
        self.one_btn = QPushButton("100%")
        self.view = ZoomPanView()
        self.view.setAcceptDrops(False)             # let drops reach this pane instead of the picture area
        self.view.viewport().setAcceptDrops(False)
        self._mode = "two"          # "two" separate photos, or "one" photo holding both lenses
        self._empty_two = "Drop your front photo here" if title[:1].upper() == "A" else "Drop your back photo here"
        self.view.set_empty_text(self._empty_two)
        self.tool_btn = QPushButton("Place circle")
        self.tool_btn.setCheckable(True)
        self.tool_btn.setToolTip(
            "On: click to place the center, drag to set the radius. "
            "Drag the center or the rim to adjust. Middle-drag still pans.")
        self.detect_btn = QPushButton("Auto-detect")
        self.spin_x = self._spin("X")
        self.spin_y = self._spin("Y")
        self.spin_r = self._spin("R")
        self.status = QLabel(" ")
        self.status.setStyleSheet("font-family: Menlo, monospace;")

        top = QHBoxLayout()
        top.addWidget(self.header, 1)
        top.addWidget(self.fit_btn)
        top.addWidget(self.one_btn)
        circ = QHBoxLayout()
        circ.addWidget(self.tool_btn)
        circ.addWidget(self.detect_btn)
        circ.addSpacing(8)
        for name, sp in (("X", self.spin_x), ("Y", self.spin_y), ("R", self.spin_r)):
            circ.addWidget(QLabel(name))
            circ.addWidget(sp)
        circ.addStretch(1)
        circ.setContentsMargins(0, 0, 0, 0)
        circle_bar = QWidget()
        circle_bar.setLayout(circ)
        self.circle_section = CollapsibleSection("Circle", circle_bar, False, f"panes/{title[:1].lower()}/circle")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addLayout(top)
        lay.addWidget(self.hint)
        lay.addWidget(self.circle_section)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.status)

        self.fit_btn.clicked.connect(self.view.fit_to_window)
        self.one_btn.clicked.connect(self.view.actual_pixels)
        self.view.cursor_pixel.connect(self._update_status)
        self.view.view_changed.connect(self._update_status_zoom)
        self.tool_btn.toggled.connect(self.view.set_circle_tool)
        self.detect_btn.clicked.connect(self.auto_detect)
        self.view.circle_edited.connect(self._circle_from_view)
        for sp in (self.spin_x, self.spin_y, self.spin_r):
            sp.valueChanged.connect(self._circle_from_fields)

    @staticmethod
    def _spin(tip: str) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setDecimals(1)
        sp.setRange(0.0, 100000.0)
        sp.setSingleStep(1.0)
        sp.setKeyboardTracking(False)
        sp.setToolTip(f"{tip} in image pixels (pixel centers at whole numbers)")
        sp.setMinimumWidth(84)
        return sp

    def array(self):
        return self._array

    def circle(self):
        return self.view.circle()

    def set_circle(self, cx: float, cy: float, r: float):
        """Set the circle from code; updates view and fields together."""
        self.view.set_circle(cx, cy, r)
        self._sync_fields(cx, cy, r)
        self.circle_changed.emit()

    def _sync_fields(self, cx, cy, r):
        for sp, v in ((self.spin_x, cx), (self.spin_y, cy), (self.spin_r, r)):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)

    def _circle_from_view(self, cx, cy, r):
        self._sync_fields(cx, cy, r)
        self.circle_changed.emit()

    def _circle_from_fields(self, _=None):
        self.view.set_circle(self.spin_x.value(), self.spin_y.value(), self.spin_r.value())
        self.circle_changed.emit()

    def auto_detect(self):
        if self._array is None:
            return
        c = auto_detect_circle(self._array)
        if c is None:
            QMessageBox.information(self, "Auto-detect", "No clear bright disc found. Place the circle by hand.")
            return
        self.set_circle(round(c.cx, 1), round(c.cy, 1), round(c.r, 1))

    # ---- drag and drop ----
    @staticmethod
    def image_paths(mime) -> list:
        """The local picture files in a drag, sorted by file name. Anything else is ignored."""
        out = []
        for url in mime.urls():
            p = url.toLocalFile()
            if p and os.path.splitext(p)[1].lower() in SUPPORTED_EXTENSIONS and os.path.isfile(p):
                out.append(p)
        return sorted(out, key=lambda p: os.path.basename(p).lower())

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.isLocalFile() for u in e.mimeData().urls()):
            e.acceptProposedAction()
            n = len(self.image_paths(e.mimeData()))
            if self._mode == "one":
                label = "Replace both lenses" if self._array is not None else "Fill A and B"
            elif n >= 2:
                label = "Fill A and B"
            elif n == 1 and self._array is not None:
                label = f"Replace {self.title[:1].upper()}"
            else:
                label = ""
            self.view.set_drop_highlight(True, label)
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dragLeaveEvent(self, e):
        self.view.set_drop_highlight(False)

    def dropEvent(self, e):
        self.view.set_drop_highlight(False)
        paths = self.image_paths(e.mimeData())
        if not paths:
            self.status.setText("drop a JPG, PNG or TIFF picture")
            e.ignore()
            return
        e.acceptProposedAction()
        self.files_dropped.emit(paths)

    def load(self, path: str):
        try:
            arr = load_image(path)
        except Exception as e:  # show the problem, keep the app alive
            QMessageBox.warning(self, "Could not open image", open_failure_text(path, e))
            return
        ui_state.set_str(f"last_files/{self.title[:1].lower()}", os.path.abspath(path))   # reopened next time
        self.load_array(arr, path, os.path.basename(path))

    def set_input_mode(self, mode: str):
        """"two": one photo per pane. "one": both panes are filled from a single two-lens photo."""
        self._mode = mode
        if mode == "one":
            self.view.set_empty_text("Drop your photo with both lenses here")
            self.hint.setText("drop another photo here to replace both lenses")
        else:
            self.view.set_empty_text(self._empty_two)
            self.hint.setText("drop another photo here to replace it")

    def clear(self):
        """Back to the empty state (no picture, no circle)."""
        self.path = None
        self._array = None
        self.view.set_image(None)
        self.view.clear_circle()
        self._sync_fields(0.0, 0.0, 0.0)
        self.header.setText(f"<b>{self.title}</b>: no image loaded")
        self.header.setToolTip("")
        self.hint.hide()
        self.status.setText(" ")

    def load_array(self, arr, path: str, shown_name: str, circle=None):
        """Show a picture that is already in memory. circle (cx, cy, r) replaces any earlier circle."""
        self.path = path
        self._array = arr
        self.view.set_image(arr)
        h, w = arr.shape[:2]
        bits = 16 if arr.dtype.itemsize == 2 else 8
        self.header.setText(f"<b>{self.title}</b>: {shown_name} &nbsp; {w} x {h}, {bits} bit")
        self.header.setToolTip(path)
        self.hint.show()
        self.spin_x.setMaximum(w)
        self.spin_y.setMaximum(h)
        self.spin_r.setMaximum(max(w, h))
        if circle is not None:
            self.set_circle(round(circle[0], 1), round(circle[1], 1), round(circle[2], 1))
        elif self.view.circle() is None:
            c = auto_detect_circle(arr)  # starting guess only; always editable
            if c is not None:
                self.set_circle(round(c.cx, 1), round(c.cy, 1), round(c.r, 1))
        self._update_status_zoom()
        self.image_loaded.emit()

    def _update_status(self, x: int, y: int):
        self._cursor = (x, y)
        self._update_status_zoom()

    def _update_status_zoom(self):
        z = self.view.zoom_level() * 100
        x, y = self._cursor
        if x >= 0:
            v = self.view.pixel_value(x, y)
            self.status.setText(f"x {x}  y {y}  RGB {v}   zoom {z:.0f}%")
        else:
            self.status.setText(f"zoom {z:.0f}%")


class DualDropArea(QWidget):
    """One big drop target shown in "one photo with both lenses" mode until a photo is loaded."""
    files_dropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WA_StyledBackground, True)          # so the black background and dashed border paint
        self.label = QLabel("Drop your photo with both lenses here")
        self.label.setAlignment(Qt.AlignCenter)
        lay = QVBoxLayout(self)
        lay.addWidget(self.label)
        self._style(False)

    def _style(self, hot: bool):
        border = "2px dashed #4a90e2" if hot else "2px dashed #555"
        self.setStyleSheet(f"DualDropArea {{ background: black; border: {border}; }}"
                           "QLabel { color: #aaa; font-size: 20px; }")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.isLocalFile() for u in e.mimeData().urls()):
            e.acceptProposedAction()
            self._style(True)
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dragLeaveEvent(self, e):
        self._style(False)

    def dropEvent(self, e):
        self._style(False)
        paths = ImagePane.image_paths(e.mimeData())
        if not paths:
            self.label.setText("Drop a JPG, PNG, TIFF or Insta360 .insp picture")
            e.ignore()
            return
        e.acceptProposedAction()
        self.files_dropped.emit(paths)
