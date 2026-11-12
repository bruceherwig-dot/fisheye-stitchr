"""Zoom and pan image viewer. Pixel (0,0) of the image sits at scene (0,0)."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
)

from .imageio_utils import to_display_uint8

MIN_ZOOM = 0.02
MAX_ZOOM = 64.0
HIT_CENTER_PX = 14   # screen pixels: grab radius for the center point
HIT_EDGE_PX = 8      # screen pixels: grab width for the circle edge


def is_trackpad_scroll(wheel_event) -> bool:
    """True for a smooth two-finger scroll (trackpad or Magic Mouse), False for a notched mouse wheel."""
    return wheel_event.phase() != Qt.NoScrollPhase or not wheel_event.pixelDelta().isNull()


class ZoomPanView(QGraphicsView):
    # Emits the image pixel under the cursor, or (-1, -1) when outside the image.
    cursor_pixel = Signal(int, int)
    view_changed = Signal()
    # Emits (cx, cy, r) in image pixels whenever the user edits the circle on screen.
    circle_edited = Signal(float, float, float)
    # Emits the clicked position in scene units (pixel EDGES at whole numbers) when pick mode is on.
    picked = Signal(float, float)
    # Seam-mask painting: scene x, scene y, phase (0 = press, 1 = move, 2 = release).
    paint_event = Signal(float, float, int)
    # View turning (the preview only): scene x, scene y, phase (0 = press, 1 = move, 2 = release).
    view_drag = Signal(float, float, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item: QGraphicsPixmapItem | None = None
        self._array: np.ndarray | None = None
        self._pan_start: QPointF | None = None
        self._circle: tuple[float, float, float] | None = None
        self._tool_on = False
        self._pick_mode = False
        self._paint_mode = False
        self._painting = False
        self._turn_mode = False              # left-drag turns the output view instead of panning
        self._turning = False
        self._brush_r = 40.0                 # brush radius in view units
        self._brush_item: QGraphicsEllipseItem | None = None
        self._empty_text = ""               # shown in the middle while no image is loaded
        self._drop_highlight = False       # a border while a file is being dragged over
        self._drop_label = ""             # and what dropping will do, e.g. "Replace A"
        self._disp_size: tuple[int, int] | None = None
        self._drag: str | None = None          # None, "new", "center", "radius"
        self._drag_press: QPointF | None = None  # view position of the press
        self._drag_offset = (0.0, 0.0)
        self._ellipse: QGraphicsEllipseItem | None = None
        self._hline: QGraphicsLineItem | None = None
        self._vline: QGraphicsLineItem | None = None
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(Qt.black)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFrameShape(QGraphicsView.NoFrame)

    # ---- image ----
    def set_image(self, array: np.ndarray | None, display_size: tuple[int, int] | None = None,
                  keep_view: bool = False):
        """Show an image. display_size lets a smaller render stand in for a larger one (the view
        coordinates stay the same). keep_view keeps zoom and pan across updates."""
        state = None
        if keep_view and self._item is not None:
            state = (self.zoom_level(), self.mapToScene(self.viewport().rect().center()))
        self._scene.clear()
        self._item = None
        self._ellipse = self._hline = self._vline = None
        self._brush_item = None
        self._array = array
        if array is None:
            self._disp_size = None
            return
        disp = to_display_uint8(array)
        h, w, _ = disp.shape
        qimg = QImage(disp.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)  # copies, so `disp` can go away
        self._item = self._scene.addPixmap(pix)
        dw, dh = display_size or (w, h)
        if (dw, dh) != (w, h):
            self._item.setScale(dw / w)
        self._disp_size = (dw, dh)
        self._scene.setSceneRect(0, 0, dw, dh)
        self._update_overlay()
        if state is not None:
            z, c = state
            self.resetTransform()
            self.scale(z, z)
            self.centerOn(c)
            self.setRenderHint(QPainter.SmoothPixmapTransform, z < 2.0)
            self.view_changed.emit()
        else:
            self.fit_to_window()

    def image_size(self) -> tuple[int, int] | None:
        """Size in view units (the display size when a stand-in render is shown)."""
        return self._disp_size

    def pixel_value(self, x: int, y: int):
        if self._array is None or self._disp_size is None:
            return None
        h, w = self._array.shape[:2]
        dw, dh = self._disp_size
        ax, ay = int(x * w / dw), int(y * h / dh)
        if 0 <= ax < w and 0 <= ay < h:
            return tuple(int(v) for v in self._array[ay, ax])
        return None

    # ---- circle overlay ----
    def circle(self) -> tuple[float, float, float] | None:
        return self._circle

    def set_circle(self, cx: float, cy: float, r: float):
        """Set the circle (image pixels, pixel-center convention) without emitting a signal."""
        self._circle = (float(cx), float(cy), max(0.0, float(r)))
        self._update_overlay()

    def clear_circle(self):
        self._circle = None
        self._update_overlay()

    def set_empty_text(self, text: str):
        self._empty_text = text
        self.viewport().update()

    def set_drop_highlight(self, on: bool, label: str = ""):
        self._drop_highlight = on
        self._drop_label = label if on else ""
        self.viewport().update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._item is None and self._empty_text or self._drop_highlight:
            p = QPainter(self.viewport())
            r = self.viewport().rect()
            if self._item is None and self._empty_text:
                p.setPen(QColor(150, 150, 150))
                f = p.font()
                f.setPointSize(max(f.pointSize() + 3, 13))
                p.setFont(f)
                p.drawText(r, Qt.AlignCenter | Qt.TextWordWrap, self._empty_text)
            if self._drop_highlight:
                pen = QPen(QColor(59, 142, 234))
                pen.setWidth(4)
                p.setPen(pen)
                p.drawRect(r.adjusted(2, 2, -2, -2))
                if self._drop_label:
                    f = p.font()
                    f.setPointSize(max(f.pointSize() + 4, 15))
                    f.setBold(True)
                    p.setFont(f)
                    box = p.fontMetrics().boundingRect(self._drop_label).adjusted(-14, -8, 14, 8)
                    box.moveCenter(r.center())
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(59, 142, 234, 215))
                    p.drawRoundedRect(box, 8, 8)
                    p.setPen(QColor(255, 255, 255))
                    p.drawText(box, Qt.AlignCenter, self._drop_label)
            p.end()

    def set_paint_mode(self, on: bool):
        self._paint_mode = on
        self._painting = False
        if not on and self._brush_item is not None:
            self._brush_item.setVisible(False)
        self.viewport().setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def set_brush_radius(self, r: float):
        self._brush_r = max(1.0, float(r))

    def _show_brush(self, scene_pos):
        """Ring that shows where a dab would land and how big it is."""
        if self._item is None:
            return
        if self._brush_item is None:
            pen = QPen(QColor(255, 255, 255))
            pen.setCosmetic(True)
            pen.setWidth(1)
            self._brush_item = self._scene.addEllipse(0, 0, 1, 1, pen)
            self._brush_item.setZValue(20)
            self._brush_item.setAcceptedMouseButtons(Qt.NoButton)
        r = self._brush_r
        self._brush_item.setRect(scene_pos.x() - r, scene_pos.y() - r, 2 * r, 2 * r)
        self._brush_item.setVisible(True)

    def set_turn_mode(self, on: bool):
        """On: a left-drag emits view_drag (the preview turns the panorama); double-click zooms in and out."""
        self._turn_mode = on

    def set_pick_mode(self, on: bool):
        self._pick_mode = on
        self.viewport().setCursor(Qt.CrossCursor if on else (Qt.CrossCursor if self._tool_on else Qt.ArrowCursor))

    def set_circle_tool(self, on: bool):
        self._tool_on = on
        self.viewport().setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def _make_pen(self, color: QColor) -> QPen:
        pen = QPen(color)
        pen.setWidth(1)
        pen.setCosmetic(True)  # stays 1 screen pixel at any zoom
        return pen

    def _update_overlay(self):
        if self._item is None:
            return
        if self._circle is None:
            for it in (self._ellipse, self._hline, self._vline):
                if it is not None:
                    it.setVisible(False)
            return
        if self._ellipse is None:
            self._ellipse = self._scene.addEllipse(0, 0, 1, 1, self._make_pen(QColor(255, 220, 0)))
            self._hline = self._scene.addLine(0, 0, 1, 1, self._make_pen(QColor(0, 230, 255)))
            self._vline = self._scene.addLine(0, 0, 1, 1, self._make_pen(QColor(0, 230, 255)))
            for it in (self._ellipse, self._hline, self._vline):
                it.setZValue(10)
                it.setAcceptedMouseButtons(Qt.NoButton)
        cx, cy, r = self._circle
        sx, sy = cx + 0.5, cy + 0.5  # pixel center -> scene coordinates
        self._ellipse.setRect(sx - r, sy - r, 2 * r, 2 * r)
        ext = max(r * 1.08, 12)
        self._hline.setLine(sx - ext, sy, sx + ext, sy)
        self._vline.setLine(sx, sy - ext, sx, sy + ext)
        for it in (self._ellipse, self._hline, self._vline):
            it.setVisible(True)

    def _scene_pt(self, view_pos: QPointF) -> tuple[float, float]:
        p = self.mapToScene(view_pos.toPoint())
        return p.x() - 0.5, p.y() - 0.5  # back to pixel-center coordinates

    def _hit_test(self, view_pos: QPointF) -> str | None:
        if self._circle is None:
            return None
        cx, cy, r = self._circle
        x, y = self._scene_pt(view_pos)
        z = self.zoom_level()
        d = math.hypot(x - cx, y - cy) * z
        if d <= HIT_CENTER_PX:
            return "center"
        if abs(d - r * z) <= HIT_EDGE_PX:
            return "radius"
        return None

    def _emit_circle(self):
        cx, cy, r = self._circle
        self.circle_edited.emit(cx, cy, r)

    def _nudge(self, dx: float, dy: float, dr: float):
        if self._circle is None:
            return
        cx, cy, r = self._circle
        self._circle = (round(cx + dx, 1), round(cy + dy, 1), max(1.0, round(r + dr, 1)))
        self._update_overlay()
        self._emit_circle()

    # ---- zoom ----
    def zoom_level(self) -> float:
        return self.transform().m11()

    def _apply_zoom(self, new_zoom: float, anchor_view_pos: QPointF):
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, new_zoom))
        old_scene = self.mapToScene(anchor_view_pos.toPoint())
        self.resetTransform()
        self.scale(new_zoom, new_zoom)
        # Keep the point under the cursor fixed on screen.
        new_view = self.mapFromScene(old_scene)
        delta = new_view - anchor_view_pos.toPoint()
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())
        # Nearest-neighbour when zoomed in past 2x, so single pixels are clear.
        self.setRenderHint(QPainter.SmoothPixmapTransform, new_zoom < 2.0)
        self.view_changed.emit()

    def fit_to_window(self):
        if self._item is None:
            return
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom_level() < 2.0)
        self.view_changed.emit()

    def actual_pixels(self):
        if self._item is not None:
            self._apply_zoom(1.0, QPointF(self.viewport().rect().center()))

    def zoom_to_pixel(self, x: float, y: float, zoom: float = 8.0):
        """Center the view on an image pixel at the given zoom."""
        if self._item is None:
            return
        self.resetTransform()
        self.scale(zoom, zoom)
        self.centerOn(x + 0.5, y + 0.5)
        self.setRenderHint(QPainter.SmoothPixmapTransform, zoom < 2.0)
        self.view_changed.emit()

    # ---- events ----
    def wheelEvent(self, event):
        """Two-finger scroll on a trackpad pans, like Photoshop. A mouse wheel, or any scroll with Cmd or
        Ctrl held, zooms around the cursor. Pinch is handled in viewportEvent."""
        if self._item is None:
            return
        if is_trackpad_scroll(event) and not (event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)):
            d = event.pixelDelta()
            dx, dy = (d.x(), d.y()) if not d.isNull() else (event.angleDelta().x() / 2.0, event.angleDelta().y() / 2.0)
            hb, vb = self.horizontalScrollBar(), self.verticalScrollBar()
            hb.setValue(hb.value() - int(round(dx)))
            vb.setValue(vb.value() - int(round(dy)))
            self.view_changed.emit()
            event.accept()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta == 0:
            return
        factor = math.pow(1.0015, delta)
        self._apply_zoom(self.zoom_level() * factor, event.position())
        event.accept()

    def viewportEvent(self, event):
        """Trackpad pinch (a native zoom gesture) zooms around the point between the fingers."""
        if (event.type() == QEvent.NativeGesture and event.gestureType() == Qt.ZoomNativeGesture
                and self._item is not None):
            self._apply_zoom(self.zoom_level() * (1.0 + event.value()), event.position())
            event.accept()
            return True
        return super().viewportEvent(event)

    def mousePressEvent(self, event):
        if self._item is None:
            super().mousePressEvent(event)
            return
        self.setFocus()
        if event.button() == Qt.LeftButton and self._paint_mode and not self._pick_mode:
            p = self.mapToScene(event.position().toPoint())
            self._painting = True
            self._show_brush(p)
            self.paint_event.emit(p.x(), p.y(), 0)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._pick_mode:
            p = self.mapToScene(event.position().toPoint())
            self.picked.emit(p.x(), p.y())
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._tool_on:
            hit = self._hit_test(event.position())
            x, y = self._scene_pt(event.position())
            w, h = self.image_size()
            if hit is None:
                # New center at the click. Keep the old radius until the user drags.
                r = self._circle[2] if self._circle else 0.45 * min(w, h)
                self._circle = (round(x, 1), round(y, 1), r)
                self._drag = "new"
            elif hit == "center":
                self._drag = "center"
                self._drag_offset = (x - self._circle[0], y - self._circle[1])
            else:
                self._drag = "radius"
            self._drag_press = event.position()
            self._update_overlay()
            self._emit_circle()
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._turn_mode and not self._paint_mode:
            p = self.mapToScene(event.position().toPoint())
            self._turning = True
            self.viewport().setCursor(Qt.ClosedHandCursor)
            self.view_drag.emit(p.x(), p.y(), 0)
            event.accept()
            return
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._pan_start = event.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._paint_mode:
            p = self.mapToScene(event.position().toPoint())
            self._show_brush(p)
            if self._painting:
                self.paint_event.emit(p.x(), p.y(), 1)
                self._emit_cursor(event.position())
                event.accept()
                return
        if self._turning:
            p = self.mapToScene(event.position().toPoint())
            self.view_drag.emit(p.x(), p.y(), 1)
            self._emit_cursor(event.position())
            event.accept()
            return
        if self._drag is not None and self._circle is not None:
            x, y = self._scene_pt(event.position())
            cx, cy, r = self._circle
            if self._drag == "center":
                cx, cy = x - self._drag_offset[0], y - self._drag_offset[1]
            else:
                moved = (event.position() - self._drag_press).manhattanLength()
                if self._drag == "radius" or moved > 4:
                    r = max(1.0, math.hypot(x - cx, y - cy))
            self._circle = (round(cx, 1), round(cy, 1), round(r, 1))
            self._update_overlay()
            self._emit_circle()
            self._emit_cursor(event.position())
            event.accept()
            return
        if self._pan_start is not None:
            d = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(round(d.x())))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(round(d.y())))
            self.view_changed.emit()
            event.accept()
        self._emit_cursor(event.position())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._turning and event.button() == Qt.LeftButton:
            self._turning = False
            self.viewport().setCursor(Qt.ArrowCursor)
            p = self.mapToScene(event.position().toPoint())
            self.view_drag.emit(p.x(), p.y(), 2)
            event.accept()
            return
        if self._painting:
            self._painting = False
            p = self.mapToScene(event.position().toPoint())
            self.paint_event.emit(p.x(), p.y(), 2)
            event.accept()
            return
        if self._drag is not None:
            self._drag = None
            event.accept()
            return
        if self._pan_start is not None:
            self._pan_start = None
            self.viewport().setCursor(Qt.CrossCursor if self._tool_on else Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.cursor_pixel.emit(-1, -1)
        if self._brush_item is not None and not self._painting:
            self._brush_item.setVisible(False)
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._paint_mode:          # quick dabs must not count as a "fit to window" double-click
            self.mousePressEvent(event)
            return
        if self._turn_mode and self._item is not None:
            self._toggle_zoom(event.position())
            event.accept()
            return
        self.fit_to_window()
        event.accept()

    def fit_zoom(self) -> float:
        """The zoom level at which the whole picture fits the window."""
        r = self._scene.sceneRect()
        vp = self.viewport().rect()
        if r.width() <= 0 or r.height() <= 0:
            return 1.0
        return min(vp.width() / r.width(), vp.height() / r.height())

    def _toggle_zoom(self, anchor):
        """Fit to window -> zoomed in on the clicked spot (twice the fit size, at least 100%) -> back to fit."""
        fit = self.fit_zoom()
        if self.zoom_level() > fit * 1.05:
            self.fit_to_window()
        else:
            self._apply_zoom(max(1.0, fit * 2.0), anchor)

    def keyPressEvent(self, event):
        arrows = {Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0), Qt.Key_Up: (0, -1), Qt.Key_Down: (0, 1)}
        if event.key() in arrows and self._circle is not None:
            step = 10.0 if event.modifiers() & Qt.ShiftModifier else 1.0
            ux, uy = arrows[event.key()]
            if event.modifiers() & (Qt.ControlModifier | Qt.AltModifier):
                # Cmd/Ctrl/Alt + Up/Down grows or shrinks the radius.
                self._nudge(0, 0, -uy * step)
            else:
                self._nudge(ux * step, uy * step, 0)
            event.accept()
            return
        if event.key() == Qt.Key_F:
            self.fit_to_window()
        elif event.key() == Qt.Key_1:
            self.actual_pixels()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.view_changed.emit()

    def _emit_cursor(self, view_pos: QPointF):
        if self._item is None:
            return
        p = self.mapToScene(view_pos.toPoint())
        x, y = int(math.floor(p.x())), int(math.floor(p.y()))
        w, h = self.image_size()
        if 0 <= x < w and 0 <= y < h:
            self.cursor_pixel.emit(x, y)
        else:
            self.cursor_pixel.emit(-1, -1)
