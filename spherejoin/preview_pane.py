"""Preview pane: the stitched equirectangular result with display options."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from .collapsible import CollapsibleSection
from .preview import MODES
from .viewer import ZoomPanView

MODE_LABELS = ("Blended", "Layer A only", "Layer B only", "Flicker A / B (overlap)", "Difference (overlap)")


class PreviewPane(QWidget):
    options_changed = Signal()
    flicker_tick = Signal()
    open_360 = Signal()
    undo_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.flicker_phase = 0
        self.mode_box = QComboBox()
        self.mode_box.addItems(MODE_LABELS)
        self.chk_seam = QCheckBox("Seam line")
        self.chk_overlap = QCheckBox("Overlap zone")
        self.pick_btn = QPushButton("Pick center")
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip("Click a spot on the preview to make it the center of the panorama")
        self.btn_360 = QPushButton("360 viewer")
        self.fit_btn = QPushButton("Fit")
        self.view = ZoomPanView()
        # ---- seam mask painting ----
        self.paint_btn = QPushButton("Paint seam")
        self.paint_btn.setCheckable(True)
        self.paint_btn.setToolTip("Paint where each image shows. Green = always image A, blue = always image B. "
                                  "Middle-drag still pans, and the wheel still zooms.")
        self.use_a = QPushButton("Use A")
        self.use_b = QPushButton("Use B")
        self.erase_btn = QPushButton("Erase")
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        for b in (self.use_a, self.use_b, self.erase_btn):
            b.setCheckable(True)
            self.tool_group.addButton(b)
        self.use_a.setChecked(True)
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(5, 200)
        self.brush_slider.setValue(40)
        self.brush_slider.setFixedWidth(120)
        self.brush_slider.setToolTip("Brush size (preview pixels)")
        self.chk_mask = QCheckBox("Show mask")
        self.chk_mask.setChecked(True)
        self.undo_btn = QPushButton("Undo")
        self.clear_btn = QPushButton("Clear mask")
        self.status = QLabel("Load both images to see the stitched preview")
        self.status.setStyleSheet("font-family: Menlo, monospace;")

        top = QHBoxLayout()
        top.addWidget(QLabel("<b>Preview</b>"))
        top.addWidget(self.mode_box)
        top.addWidget(self.chk_seam)
        top.addWidget(self.chk_overlap)
        top.addStretch(1)
        top.addWidget(self.pick_btn)
        top.addWidget(self.btn_360)
        top.addWidget(self.fit_btn)
        paint_row = QHBoxLayout()
        for b in (self.use_a, self.use_b, self.erase_btn):
            paint_row.addWidget(b)
        paint_row.addWidget(QLabel("Brush"))
        paint_row.addWidget(self.brush_slider)
        paint_row.addWidget(self.chk_mask)
        paint_row.addStretch(1)
        paint_row.addWidget(self.undo_btn)
        paint_row.addWidget(self.clear_btn)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addLayout(top)
        paint_row.setContentsMargins(0, 0, 0, 0)
        self.paint_bar = QWidget()
        self.paint_bar.setLayout(paint_row)
        self.paint_section = CollapsibleSection("Paint seam", self.paint_bar, False, None)
        self.paint_btn.setParent(self)        # kept as the on/off switch behind the arrow, never shown
        self.paint_btn.hide()
        lay.addWidget(self.paint_section)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.status)

        self.timer = QTimer(self)
        self.timer.setInterval(700)
        self.timer.timeout.connect(self._tick)
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        self.chk_seam.toggled.connect(lambda _: self.options_changed.emit())
        self.chk_overlap.toggled.connect(lambda _: self.options_changed.emit())
        self.pick_btn.toggled.connect(self.view.set_pick_mode)
        self.btn_360.clicked.connect(self.open_360.emit)
        self.fit_btn.clicked.connect(self.view.fit_to_window)
        self.paint_btn.toggled.connect(self._paint_toggled)
        self.paint_section.toggled.connect(self.paint_btn.setChecked)
        self.paint_btn.toggled.connect(self.paint_section.setExpanded)
        self.pick_btn.toggled.connect(lambda on: self.paint_btn.setChecked(False) if on else None)
        self.brush_slider.valueChanged.connect(lambda v: self.view.set_brush_radius(v))
        self.view.set_brush_radius(self.brush_slider.value())
        self.chk_mask.toggled.connect(lambda _: self.options_changed.emit())
        self.undo_btn.clicked.connect(self.undo_requested.emit)
        self.clear_btn.clicked.connect(self.clear_requested.emit)
        sc = QShortcut(QKeySequence.Undo, self)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(self.undo_requested.emit)

    def _paint_toggled(self, on: bool):
        if on and self.pick_btn.isChecked():
            self.pick_btn.setChecked(False)
        self.view.set_paint_mode(on)
        self.options_changed.emit()

    def paint_tool(self) -> str:
        """"a", "b" or "erase"."""
        return "b" if self.use_b.isChecked() else ("erase" if self.erase_btn.isChecked() else "a")

    def brush_radius(self) -> float:
        return float(self.brush_slider.value())

    def mode(self) -> str:
        return MODES[self.mode_box.currentIndex()]

    def _mode_changed(self, _):
        if self.mode() == "flicker":
            self.timer.start()
        else:
            self.timer.stop()
        self.options_changed.emit()

    def _tick(self):
        self.flicker_phase ^= 1
        self.flicker_tick.emit()
