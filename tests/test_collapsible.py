import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from spherejoin.collapsible import CollapsibleSection
from spherejoin.export import ExportSettings
from spherejoin.export_dialog import ExportDialog
from spherejoin.panes import ImagePane
from spherejoin.workspace import Workspace


def titles(controls):
    return {s.header.text(): s.isExpanded() for s in controls._sections}


def test_section_opens_closes_and_reports(app):
    inner = QWidget()
    sec = CollapsibleSection("Things", inner, expanded=False)
    sec.show()
    seen = []
    sec.toggled.connect(seen.append)
    assert not sec.isExpanded() and not inner.isVisible()
    sec.header.click()
    assert sec.isExpanded() and inner.isVisible() and seen == [True]
    sec.header.click()
    assert not sec.isExpanded() and not inner.isVisible() and seen == [True, False]
    sec.setExpanded(True)
    sec.setExpanded(True)                       # same value again says nothing
    assert seen == [True, False, True]


def test_the_controls_start_with_only_layout_and_blend_open(app):
    w = Workspace()
    w.show()
    t = titles(w.controls)
    assert t == {"Lens settings": False, "Lens A trim (degrees)": False, "Lens B trim (degrees)": False,
                 "Output layout": True, "Output view (what lands in the center)": False,
                 "Blend": True}
    # What is closed is really out of sight, and what is open is shown.
    assert not w.controls.fov.isVisible() and not w.controls.k1.isVisible()
    assert w.controls.seam.isVisible() and w.controls.proj_kind.isVisible()
    w.close()


def test_open_and_close_choices_are_remembered_between_sessions(app):
    w = Workspace()
    w.show()
    sections = {s.header.text(): s for s in w.controls._sections}
    sections["Lens A trim (degrees)"].setExpanded(True)
    sections["Blend"].setExpanded(False)
    w.close()
    w2 = Workspace()                            # a new session: same saved state file
    t = titles(w2.controls)
    assert t["Lens A trim (degrees)"] is True and t["Blend"] is False and t["Lens settings"] is False
    w2.close()


def test_expand_and_collapse_all_and_resets_still_reach_hidden_rows(app):
    w = Workspace()
    w.show()
    w.controls.expand_all()
    assert all(titles(w.controls).values()) and w.controls.fov.isVisible()
    w.controls.collapse_all()
    assert not any(titles(w.controls).values())
    w.controls.fov.setValue(200.0)              # still readable and settable while hidden
    assert w.controls.lens_settings()["fov_deg"] == 200.0
    w.close()


def test_the_circle_row_is_folded_away_by_default_and_keeps_working(app):
    pane = ImagePane("A (front)")
    pane.show()
    assert not pane.circle_section.isExpanded() and not pane.spin_x.isVisible() and not pane.detect_btn.isVisible()
    pane.set_circle(100.0, 120.0, 80.0)         # the circle itself is drawn and editable without the row
    assert pane.circle() == (100.0, 120.0, 80.0)
    pane.circle_section.setExpanded(True)
    assert pane.spin_x.isVisible() and pane.spin_x.value() == 100.0
    pane.close()


def test_the_two_panes_remember_their_circle_rows_separately(app):
    a, b = ImagePane("A (front)"), ImagePane("B (back)")
    a.circle_section.setExpanded(True)
    a2, b2 = ImagePane("A (front)"), ImagePane("B (back)")
    assert a2.circle_section.isExpanded() and not b2.circle_section.isExpanded()
    for p in (a, b, a2, b2):
        p.close()


def test_the_paint_arrow_turns_painting_on_and_off(app):
    w = Workspace()
    w.show()
    pv = w.preview
    assert not pv.paint_section.isExpanded() and not pv.use_a.isVisible()
    assert not pv.paint_btn.isChecked() and not pv.paint_btn.isVisible()
    pv.paint_section.setExpanded(True)
    assert pv.paint_btn.isChecked() and pv.use_a.isVisible() and pv.view._paint_mode
    pv.paint_section.setExpanded(False)
    assert not pv.paint_btn.isChecked() and not pv.view._paint_mode
    # Pick center and painting still take turns, and the arrow follows.
    pv.paint_section.setExpanded(True)
    pv.pick_btn.setChecked(True)
    assert not pv.paint_btn.isChecked() and not pv.paint_section.isExpanded()
    pv.pick_btn.setChecked(False)
    w.close()


def test_export_window_shows_only_the_basics_and_has_no_sampling_box(app):
    d = ExportDialog(None, ExportSettings(width=2000), "/tmp/x.tif", 6528, False, has_exiftool=True)
    d.show()
    assert not d.more_section.isExpanded()
    assert d.width_spin.isVisible() and d.fmt.isVisible() and d.path_edit.isVisible()
    assert not d.layers.isVisible() and not d.depth16.isVisible() and not d.quality.isVisible()
    assert not hasattr(d, "interp") and d.settings().interp == "lanczos"
    d.more_section.setExpanded(True)
    assert d.layers.isVisible() and d.meta.isVisible()
    d.close()


def test_export_window_opens_the_extras_when_one_is_already_on(app):
    for st, has_mask in ((ExportSettings(save_layers=True), False), (ExportSettings(sixteen_bit=True), False),
                         (ExportSettings(use_mask=False), True)):
        d = ExportDialog(None, st, "/tmp/x.tif", 6528, False, has_exiftool=False, has_mask=has_mask)
        assert d.more_section.isExpanded(), st
        d.close()
    d = ExportDialog(None, ExportSettings(), "/tmp/x.tif", 6528, False, has_exiftool=False)
    assert not d.more_section.isExpanded()
    d.close()


def test_the_controls_are_in_the_expected_order(app):
    w = Workspace()
    order = [s.header.text() for s in w.controls._sections]
    assert order == ["Lens settings", "Lens A trim (degrees)", "Lens B trim (degrees)",
                     "Output view (what lands in the center)", "Output layout", "Blend"]
    w.close()
