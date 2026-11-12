"""Pinch to zoom and two-finger scroll to pan, using synthetic trackpad events."""
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QNativeGestureEvent, QPointingDevice, QWheelEvent
from PySide6.QtWidgets import QApplication

from spherejoin.viewer import ZoomPanView
from spherejoin.viewer360 import Viewer360


def pinch(widget, value, pos=QPointF(100, 80)):
    ev = QNativeGestureEvent(Qt.ZoomNativeGesture, QPointingDevice.primaryPointingDevice(), 2, pos, pos, pos,
                             value, QPointF(0, 0))
    QApplication.sendEvent(widget, ev)
    return ev


def scroll(widget, dx, dy, trackpad=True, mods=Qt.NoModifier, pos=QPointF(100, 80)):
    """A trackpad scroll carries a pixel delta and a scroll phase; a notched mouse wheel carries neither."""
    if trackpad:
        ev = QWheelEvent(pos, pos, QPoint(dx, dy), QPoint(dx, dy), Qt.NoButton, mods, Qt.ScrollUpdate, False)
    else:
        ev = QWheelEvent(pos, pos, QPoint(0, 0), QPoint(dx, dy), Qt.NoButton, mods, Qt.NoScrollPhase, False)
    QApplication.sendEvent(widget, ev)
    return ev


@pytest.fixture
def view(app):
    v = ZoomPanView()
    v.resize(400, 300)
    v.show()
    v.set_image(np.random.default_rng(0).integers(0, 255, (600, 900, 3), dtype=np.uint8))
    v.zoom_to_pixel(450, 300, 2.0)
    yield v
    v.close()


def test_pinch_out_zooms_in_and_pinch_in_zooms_out_around_the_fingers(view):
    z0 = view.zoom_level()
    anchor = QPointF(100, 80)
    before = view.mapToScene(anchor.toPoint())
    pinch(view.viewport(), 0.25, anchor)
    assert view.zoom_level() == pytest.approx(z0 * 1.25)
    after = view.mapToScene(anchor.toPoint())
    assert abs(after.x() - before.x()) < 1.0 and abs(after.y() - before.y()) < 1.0       # the spot under the fingers stays put
    pinch(view.viewport(), -0.2, anchor)
    assert view.zoom_level() == pytest.approx(z0 * 1.25 * 0.8)


def test_pinch_is_limited_like_every_other_zoom(view):
    for _ in range(80):
        pinch(view.viewport(), 0.5)
    assert view.zoom_level() <= 64.0 + 1e-9
    for _ in range(200):
        pinch(view.viewport(), -0.5)
    assert view.zoom_level() >= 0.02 - 1e-9


def test_two_finger_scroll_pans_and_does_not_zoom(view):
    z0 = view.zoom_level()
    hb, vb = view.horizontalScrollBar(), view.verticalScrollBar()
    h0, v0 = hb.value(), vb.value()
    scroll(view.viewport(), 30, 20)
    assert view.zoom_level() == z0
    assert hb.value() == h0 - 30 and vb.value() == v0 - 20          # content follows the fingers, like any scroll area
    scroll(view.viewport(), -50, -10)
    assert hb.value() == h0 - 30 + 50 and vb.value() == v0 - 20 + 10


def test_a_notched_mouse_wheel_still_zooms_at_the_cursor(view):
    z0 = view.zoom_level()
    h0, v0 = view.horizontalScrollBar().value(), view.verticalScrollBar().value()
    scroll(view.viewport(), 0, 120, trackpad=False)
    assert view.zoom_level() > z0
    scroll(view.viewport(), 0, -240, trackpad=False)
    assert view.zoom_level() < z0 * 1.2


def test_holding_cmd_or_ctrl_while_scrolling_zooms(view):
    for mod in (Qt.ControlModifier, Qt.MetaModifier):
        z0 = view.zoom_level()
        scroll(view.viewport(), 0, 40, trackpad=True, mods=mod)
        assert view.zoom_level() > z0


def test_nothing_happens_without_an_image(app):
    v = ZoomPanView()
    v.show()
    pinch(v.viewport(), 0.3)
    scroll(v.viewport(), 10, 10)
    v.close()


def test_the_360_viewer_pinches_to_zoom_and_scroll_looks_around(app):
    w = Viewer360()
    w.set_panorama(np.full((256, 512, 3), 90, np.uint8))
    w.show()
    fov0, yaw0, pitch0 = w.fov, w.yaw, w.pitch
    pinch(w, 0.25)                                     # spread fingers: closer, so a narrower view
    assert w.fov == pytest.approx(fov0 / 1.25)
    pinch(w, -0.2)
    assert w.fov == pytest.approx(fov0 / 1.25 / 0.8)
    w.fov = 90.0
    scroll(w, 40, 0)
    assert w.yaw < yaw0 and w.fov == 90.0               # same direction as dragging right
    scroll(w, 0, 30)
    assert w.pitch > pitch0
    z = w.fov
    scroll(w, 0, 120, trackpad=False)                   # a mouse wheel still zooms
    assert w.fov < z
    for _ in range(40):
        pinch(w, 0.5)
    assert w.fov >= 20.0 - 1e-9
    w.close()
