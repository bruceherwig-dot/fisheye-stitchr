"""Panel-level GUI tests (ImagePane only, never the full main window)."""
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from spherejoin.panes import ImagePane


@pytest.fixture
def pane(app):
    p = ImagePane("T")
    p.resize(900, 700)
    p.show()
    img = np.zeros((600, 800, 3), np.uint8)
    p._array = img
    p.view.set_image(img)
    p.view.zoom_to_pixel(400, 300, 1.0)
    app.processEvents()
    yield p
    p.close()


def _vp(view, x, y):
    """Viewport position of image pixel (x, y)."""
    pt = view.mapFromScene(x + 0.5, y + 0.5)
    return QPoint(pt.x(), pt.y())


def test_click_places_center_and_drag_sets_radius(pane):
    pane.tool_btn.setChecked(True)
    vp = pane.view.viewport()
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, _vp(pane.view, 400, 300))
    QTest.mouseMove(vp, _vp(pane.view, 500, 300))
    QTest.mouseMove(vp, _vp(pane.view, 600, 300))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, _vp(pane.view, 600, 300))
    cx, cy, r = pane.circle()
    assert abs(cx - 400) <= 1 and abs(cy - 300) <= 1
    assert abs(r - 200) <= 2
    assert abs(pane.spin_x.value() - cx) < 1e-6 and abs(pane.spin_r.value() - r) < 1e-6


def test_drag_center_and_rim(pane):
    pane.set_circle(400, 300, 100)
    pane.tool_btn.setChecked(True)
    vp = pane.view.viewport()
    # Grab the center and move it by (+30, +20).
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, _vp(pane.view, 400, 300))
    QTest.mouseMove(vp, _vp(pane.view, 430, 320))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, _vp(pane.view, 430, 320))
    cx, cy, r = pane.circle()
    assert abs(cx - 430) <= 1 and abs(cy - 320) <= 1 and abs(r - 100) < 1e-6
    # Grab the rim and pull it out to 150.
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, _vp(pane.view, 530, 320))
    QTest.mouseMove(vp, _vp(pane.view, 580, 320))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, _vp(pane.view, 580, 320))
    cx2, cy2, r2 = pane.circle()
    assert abs(r2 - 150) <= 2 and abs(cx2 - cx) < 1e-6


def test_arrow_nudging(pane):
    pane.set_circle(400, 300, 100)
    pane.view.setFocus()
    QTest.keyClick(pane.view, Qt.Key_Right)
    assert pane.circle()[0] == 401.0
    QTest.keyClick(pane.view, Qt.Key_Down, Qt.ShiftModifier)
    assert pane.circle()[1] == 310.0
    QTest.keyClick(pane.view, Qt.Key_Up, Qt.AltModifier)  # radius up
    assert pane.circle()[2] == 101.0
    assert pane.spin_x.value() == 401.0 and pane.spin_y.value() == 310.0


def test_numeric_fields_move_circle(pane):
    pane.set_circle(400, 300, 100)
    pane.spin_x.setValue(250.5)
    assert pane.circle()[0] == 250.5


def test_pan_still_works_with_tool_off(pane):
    pane.view.zoom_to_pixel(400, 300, 4.0)
    h = pane.view.horizontalScrollBar().value()
    vp = pane.view.viewport()
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, QPoint(300, 300))
    QTest.mouseMove(vp, QPoint(250, 300))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, QPoint(250, 300))
    assert pane.view.horizontalScrollBar().value() != h
    assert pane.circle() is None
