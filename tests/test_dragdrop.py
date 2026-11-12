"""Drag and drop of photos onto the picture panes, using simulated drag events."""
import warnings

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, QUrl, Qt
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent
from PySide6.QtWidgets import QApplication

from spherejoin.panes import ImagePane
from spherejoin.workspace import Workspace

warnings.filterwarnings("ignore", category=DeprecationWarning)


def disc(path, r=60, size=(240, 320), shade=90):
    img = np.zeros((size[0], size[1], 3), np.uint8)
    cv2.circle(img, (size[1] // 2, size[0] // 2), r, (shade, shade, shade), -1)
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture
def pics(tmp_path):
    out = {n: disc(tmp_path / n, shade=60 + 20 * i) for i, n in enumerate(("a_front.png", "b_back.png", "c_extra.png", "UPPER.PNG"))}
    (tmp_path / "note.txt").write_text("hello")
    (tmp_path / "pic.gif").write_bytes(b"GIF89a")
    out["note.txt"], out["pic.gif"] = str(tmp_path / "note.txt"), str(tmp_path / "pic.gif")
    return out


def mime(*paths):
    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(p) for p in paths])
    return m


def enter(widget, m):
    ev = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, m, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, ev)
    return ev


def drop(widget, m):
    ev = QDropEvent(QPointF(5, 5), Qt.CopyAction, m, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, ev)
    return ev


def leave(widget):
    QApplication.sendEvent(widget, QDragLeaveEvent())


@pytest.fixture
def ws(app):
    w = Workspace()
    w.resize(1500, 900)
    w.show()
    yield w
    for t in (w._quick, w._full, w._compose):
        t.stop()
    w.close()


def test_the_panes_take_the_drop_and_the_picture_area_lets_it_through(app):
    p = ImagePane("A (front)")
    assert p.acceptDrops() and not p.view.acceptDrops() and not p.view.viewport().acceptDrops()
    assert not hasattr(p, "open_btn")
    assert "front" in p.view._empty_text and "back" in ImagePane("B (back)").view._empty_text
    p.close()


def test_one_photo_lands_on_the_pane_it_was_dropped_on(ws, pics):
    m = mime(pics["a_front.png"])
    assert enter(ws.pane_a, m).isAccepted()
    assert drop(ws.pane_a, m).isAccepted()
    assert ws.pane_a.path == pics["a_front.png"] and ws.pane_b.array() is None
    m2 = mime(pics["b_back.png"])
    enter(ws.pane_b, m2)
    drop(ws.pane_b, m2)
    assert ws.pane_b.path == pics["b_back.png"] and ws.pane_a.path == pics["a_front.png"]
    assert ws.ready()


def test_a_second_single_drop_replaces_the_photo_on_that_pane(ws, pics):
    for name in ("a_front.png", "c_extra.png"):
        m = mime(pics[name]); enter(ws.pane_a, m); drop(ws.pane_a, m)
    assert ws.pane_a.path == pics["c_extra.png"]


@pytest.mark.parametrize("target", ["a", "b"])
def test_two_photos_fill_a_then_b_by_file_name_whichever_pane_got_them(ws, pics, target):
    pane = ws.pane_a if target == "a" else ws.pane_b
    m = mime(pics["b_back.png"], pics["a_front.png"])         # dropped in the "wrong" order on purpose
    enter(pane, m)
    drop(pane, m)
    assert ws.pane_a.path == pics["a_front.png"] and ws.pane_b.path == pics["b_back.png"]
    assert "as A" in ws.preview.status.text() and "as B" in ws.preview.status.text()


def test_more_than_two_photos_uses_the_first_two_and_says_so(ws, pics):
    m = mime(pics["c_extra.png"], pics["b_back.png"], pics["a_front.png"])
    enter(ws.pane_a, m); drop(ws.pane_a, m)
    assert ws.pane_a.path == pics["a_front.png"] and ws.pane_b.path == pics["b_back.png"]
    assert "first two" in ws.preview.status.text()


def test_other_files_are_ignored_with_a_note_and_nothing_changes(ws, pics):
    ws.pane_a.load(pics["a_front.png"])
    before = ws.pane_a.path
    for bad in ("note.txt", "pic.gif"):
        m = mime(pics[bad])
        enter(ws.pane_a, m)
        ev = drop(ws.pane_a, m)
        assert not ev.isAccepted()
        assert ws.pane_a.path == before
        assert "JPG, PNG or TIFF" in ws.pane_a.status.text()
    # A mix keeps just the pictures.
    m = mime(pics["note.txt"], pics["b_back.png"])
    enter(ws.pane_b, m); drop(ws.pane_b, m)
    assert ws.pane_b.path == pics["b_back.png"]


def test_a_web_link_is_not_a_file_and_is_turned_away(app):
    p = ImagePane("A (front)")
    m = QMimeData()
    m.setUrls([QUrl("https://example.com/photo.jpg")])
    assert not enter(p, m).isAccepted()
    t = QMimeData(); t.setText("just words")
    assert not enter(p, t).isAccepted()
    p.close()


def test_the_pane_shows_a_border_while_dragging_and_clears_it_afterwards(ws, pics):
    m = mime(pics["a_front.png"])
    assert not ws.pane_a.view._drop_highlight
    enter(ws.pane_a, m)
    assert ws.pane_a.view._drop_highlight
    leave(ws.pane_a)
    assert not ws.pane_a.view._drop_highlight
    enter(ws.pane_a, m); drop(ws.pane_a, m)
    assert not ws.pane_a.view._drop_highlight


def test_upper_case_extensions_count(ws, pics):
    m = mime(pics["UPPER.PNG"])
    enter(ws.pane_a, m); drop(ws.pane_a, m)
    assert ws.pane_a.path == pics["UPPER.PNG"]


def test_dropped_photos_are_what_reopens_next_time(app, pics):
    w = Workspace()
    m = mime(pics["b_back.png"], pics["a_front.png"])
    enter(w.pane_a, m); drop(w.pane_a, m)
    w.close()
    w2 = Workspace()
    assert w2.open_last_files() == [pics["a_front.png"], pics["b_back.png"]]
    w2.close()


def test_the_empty_hint_paints_and_goes_away_once_a_photo_is_loaded(ws, pics):
    ws.show()
    assert ws.pane_a.view._item is None and ws.pane_a.view._empty_text
    img = ws.pane_a.view.viewport().grab().toImage()
    assert img.width() > 0                                  # painting the hint did not raise
    ws.pane_a.load(pics["a_front.png"])
    assert ws.pane_a.view._item is not None


def test_a_gray_hint_appears_once_a_photo_is_loaded(ws, pics):
    assert not ws.pane_a.hint.isVisible()                  # the empty pane already says "Drop your front photo here"
    ws.pane_a.load(pics["a_front.png"])
    assert ws.pane_a.hint.isVisible() and "replace" in ws.pane_a.hint.text()
    assert not ws.pane_b.hint.isVisible()


def test_the_drag_border_says_what_dropping_will_do(ws, pics):
    ws.pane_a.load(pics["a_front.png"])
    enter(ws.pane_a, mime(pics["b_back.png"]))
    assert ws.pane_a.view._drop_label == "Replace A"
    leave(ws.pane_a)
    assert ws.pane_a.view._drop_label == "" and not ws.pane_a.view._drop_highlight
    enter(ws.pane_b, mime(pics["b_back.png"]))              # B is empty: its own hint text is enough
    assert ws.pane_b.view._drop_label == ""
    leave(ws.pane_b)
    enter(ws.pane_b, mime(pics["a_front.png"], pics["b_back.png"]))
    assert ws.pane_b.view._drop_label == "Fill A and B"
    drop(ws.pane_b, mime(pics["a_front.png"], pics["b_back.png"]))
    assert ws.pane_b.view._drop_label == ""
    ws.pane_b.view.grab()                                    # painting the label does not raise


def test_reopening_the_last_photos_leaves_a_plain_note(app, pics):
    w = Workspace()
    notes = []
    w.note.connect(notes.append)
    assert w.open_last_files() == [] and notes == []         # nothing remembered: no note
    w.pane_a.load(pics["a_front.png"]); w.pane_b.load(pics["b_back.png"])
    w.close()
    w2 = Workspace()
    got = []
    w2.note.connect(got.append)
    w2.open_last_files()
    assert len(got) == 1 and "Reopened your last photos" in got[0] and "replace" in got[0]
    for t in (w2._quick, w2._full, w2._compose):
        t.stop()
    w2.close()
