"""A section with a small arrow in its header. Click the arrow or the title to open or close it."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSizePolicy, QToolButton, QVBoxLayout, QWidget

from . import ui_state


class CollapsibleSection(QWidget):
    toggled = Signal(bool)      # True when it opens

    def __init__(self, title: str, content: QWidget, expanded: bool = False, key: str | None = None, parent=None):
        super().__init__(parent)
        self.key = key
        self.content = content
        self.header = QToolButton()
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.setStyleSheet("QToolButton { border: none; font-weight: bold; text-align: left; padding: 5px 2px; }")
        self.header.setToolTip("Click to open or close")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.header)
        content.setParent(self)
        lay.addWidget(content)
        start = ui_state.get_bool(f"sections/{key}", expanded) if key else expanded
        self._apply(start)
        self.header.toggled.connect(self._on_header)

    def _apply(self, on: bool):
        self.header.blockSignals(True)
        self.header.setChecked(on)
        self.header.blockSignals(False)
        self.header.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.content.setVisible(on)

    def _on_header(self, on: bool):
        self._apply(on)
        if self.key:
            ui_state.set_bool(f"sections/{self.key}", on)
        self.toggled.emit(on)

    def isExpanded(self) -> bool:
        return self.header.isChecked()

    def setExpanded(self, on: bool):
        if on != self.isExpanded():
            self._apply(on)
            if self.key:
                ui_state.set_bool(f"sections/{self.key}", on)
            self.toggled.emit(on)
