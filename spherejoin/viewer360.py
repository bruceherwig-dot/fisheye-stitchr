"""Interactive 360 viewer: look around the equirectangular result as a sphere."""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from .imageio_utils import to_display_uint8
from .projection import direction_to_equirect, rotation
from .viewer import is_trackpad_scroll


def render_perspective(equi: np.ndarray, yaw_deg: float, pitch_deg: float, fov_deg: float,
                       out_w: int, out_h: int) -> np.ndarray:
    """Flat (rectilinear) view into the panorama. fov_deg is the horizontal field of view."""
    f = (out_w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    u, v = np.meshgrid(np.arange(out_w) + 0.5 - out_w / 2.0, np.arange(out_h) + 0.5 - out_h / 2.0)
    d = np.stack([u / f, -v / f, np.ones_like(u)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    d_world = d @ rotation(yaw_deg, pitch_deg, 0.0).T
    ex, ey = direction_to_equirect(d_world, equi.shape[1], equi.shape[0])
    return cv2.remap(equi, ex.astype(np.float32), ey.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


class Viewer360(QWidget):
    """Drag to look around, wheel to zoom, R to reset."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Fisheye StitchR 360 viewer")
        self.resize(960, 600)
        self.label = QLabel("No panorama yet")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.label.setMinimumSize(200, 150)
        self.label.setStyleSheet("background: black; color: #888;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.label)
        self.setFocusPolicy(Qt.StrongFocus)
        self._pano: np.ndarray | None = None
        self.yaw = 0.0
        self.pitch = 0.0
        self.fov = 90.0
        self._drag = None

    def set_panorama(self, arr: np.ndarray):
        self._pano = to_display_uint8(arr)
        self._refresh()

    def reset(self):
        self.yaw, self.pitch, self.fov = 0.0, 0.0, 90.0
        self._refresh()

    def _refresh(self):
        if self._pano is None:
            return
        w, h = max(self.label.width(), 16), max(self.label.height(), 16)
        img = np.ascontiguousarray(render_perspective(self._pano, self.yaw, self.pitch, self.fov, w, h))
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        d = e.position() - self._drag
        self._drag = e.position()
        deg_per_px = self.fov / max(self.label.width(), 1)
        self.yaw = (self.yaw - d.x() * deg_per_px + 180.0) % 360.0 - 180.0   # drag the scene with the cursor
        self.pitch = float(np.clip(self.pitch + d.y() * deg_per_px, -89.0, 89.0))
        self._refresh()

    def mouseReleaseEvent(self, e):
        self._drag = None
        self.unsetCursor()

    def event(self, e):
        """Pinch zooms the view (spread fingers = closer)."""
        if e.type() == QEvent.NativeGesture and e.gestureType() == Qt.ZoomNativeGesture:
            self.fov = float(np.clip(self.fov / (1.0 + e.value()), 20.0, 120.0))
            self._refresh()
            e.accept()
            return True
        return super().event(e)

    def wheelEvent(self, e):
        if is_trackpad_scroll(e) and not (e.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)):
            d = e.pixelDelta()
            dx, dy = (d.x(), d.y()) if not d.isNull() else (e.angleDelta().x() / 2.0, e.angleDelta().y() / 2.0)
            deg_per_px = self.fov / max(self.label.width(), 1)
            self.yaw = (self.yaw - dx * deg_per_px + 180.0) % 360.0 - 180.0     # same direction as dragging
            self.pitch = float(np.clip(self.pitch + dy * deg_per_px, -89.0, 89.0))
            self._refresh()
            e.accept()
            return
        step = e.angleDelta().y() / 120.0
        self.fov = float(np.clip(self.fov * (0.92 ** step), 20.0, 120.0))
        self._refresh()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_R:
            self.reset()
        else:
            super().keyPressEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()
