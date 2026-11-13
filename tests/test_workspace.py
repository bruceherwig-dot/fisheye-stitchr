"""Panel-level tests of the Workspace (never the full main window)."""
import json

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from spherejoin.projection import LensParams
from spherejoin.synth import make_test_equirect, render_fisheye_from_equirect
from spherejoin.workspace import PREVIEW_H, PREVIEW_W, Workspace

EQUI_W, EQUI_H = 2048, 1024


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("sj")
    equi = make_test_equirect(EQUI_W, EQUI_H)
    kw = dict(cx=800.0, cy=800.0, radius=760.0, fov_deg=220.0)
    paths = []
    for name, base in (("a.png", 0.0), ("b.png", 180.0)):
        fish = render_fisheye_from_equirect(equi, LensParams(base_yaw=base, **kw), (1600, 1600))
        p = d / name
        cv2.imwrite(str(p), fish[:, :, ::-1])
        paths.append(str(p))
    return paths, equi


@pytest.fixture
def ws(app, files):
    w = Workspace()
    w.resize(1500, 900)
    w.show()
    w.controls.proj_kind.setCurrentIndex(w.controls.proj_kind.findData("equirectangular"))
    w.pane_a.load(files[0][0])
    w.pane_b.load(files[0][1])
    yield w
    w.close_children()
    w.close()


def test_not_ready_until_both_loaded(app, files):
    w = Workspace()
    assert not w.ready() and w.render(1024) is None
    w.pane_a.load(files[0][0])
    assert not w.ready()
    w.pane_b.load(files[0][1])
    assert w.ready()
    w.close()


def test_auto_circles_and_full_pipeline_rebuild_panorama(ws, files):
    ca, cb = ws.pane_a.circle(), ws.pane_b.circle()
    assert abs(ca[0] - 800) < 4 and abs(ca[2] - 760) < 6
    ws.flush()
    out = ws.last_preview
    assert out.shape == (PREVIEW_W // 2, PREVIEW_W, 3)
    err = np.abs(out.astype(int) - files[1].astype(int))[10:-10].mean()
    print("workspace pipeline mean abs err", round(float(err), 2))
    assert err < 6.0


def test_blend_change_recomposes_without_rewarping(ws):
    ws.flush()
    key = ws._layers_key
    layers = ws._layers
    ws.controls.feather.setValue(5.0)
    ws._compose.stop()
    ws.render(PREVIEW_W)
    assert ws._layers is layers and ws._layers_key == key


def test_geometry_change_rewarps_and_quick_render_works(ws):
    ws.flush()
    layers = ws._layers
    ws.controls.trim_b[1].setValue(1.0)
    out = ws.render(1024)
    assert out.shape == (512, 1024, 3)
    assert ws._layers is not layers
    ws._quick.stop(); ws._full.stop()


def test_modes_and_overlays_render(ws):
    ws.flush()
    for i in range(ws.preview.mode_box.count()):
        ws.preview.mode_box.setCurrentIndex(i)
        ws.preview.timer.stop()
        assert ws.render(PREVIEW_W).shape == (PREVIEW_H, PREVIEW_W, 3)
    ws.preview.chk_seam.setChecked(True)
    ws.preview.chk_overlap.setChecked(True)
    ws.render(PREVIEW_W)


def test_pick_center_sets_view_and_turns_tool_off(ws):
    ws.flush()
    ws.preview.pick_btn.setChecked(True)
    ws._on_pick(PREVIEW_W * 0.75, PREVIEW_H / 2)     # a point 90 degrees to the right
    v = ws.controls.view()
    assert abs(v.yaw - 90.0) < 0.2 and abs(v.pitch) < 0.2
    assert not ws.preview.pick_btn.isChecked()
    ws._quick.stop(); ws._full.stop()


def test_reset_view_and_controls_values(ws):
    ws.controls.set_view(40, 10, 5)
    v = ws.controls.view()
    assert (v.yaw, v.pitch, v.roll) == (40, 10, 5)
    assert ws.controls.v_yaw.slider.value() == 400
    ws.controls.reset_view()
    assert ws.controls.view().yaw == 0
    s = ws.controls.lens_settings()
    assert s["fov_deg"] == 220.0 and s["model"] == "equidistant"
    bp = ws.controls.blend()
    assert bp.wb_a == (1.0, 1.0, 1.0) and bp.seam_deg == 90.0


def test_polynomial_fields_follow_model(ws):
    assert not ws.controls.k1.isEnabled()
    ws.controls.model.setCurrentText("polynomial")
    assert ws.controls.k1.isEnabled()
    ws.controls.model.setCurrentText("equidistant")
    ws._quick.stop(); ws._full.stop()


def test_360_window_opens_with_panorama(ws):
    ws.open_360()
    assert ws.viewer360.isVisible()
    assert ws.viewer360.label.pixmap() is not None and not ws.viewer360.label.pixmap().isNull()


def test_project_roundtrip_restores_everything(app, files, tmp_path):
    paths, _ = files
    w = Workspace()
    w.pane_a.load(paths[0]); w.pane_b.load(paths[1])
    w.pane_a.set_circle(801.5, 799.0, 758.0)
    w.controls.trim_b[1].setValue(1.25)
    w.controls.seam.setValue(88.5)
    w.controls.hard.setChecked(True)
    w.controls.set_view(30.0, -5.0, 2.0)
    w.controls.model.setCurrentText("polynomial")
    w.controls.k2.setValue(0.05)
    w.preview.chk_seam.setChecked(True)
    w.export_settings.fmt = "png"
    w.export_settings.save_layers = True
    proj = str(tmp_path / "t.spherejoin.json")
    w.save_project_to(proj)
    w.close()

    w2 = Workspace()
    warnings = w2.load_project_from(proj)
    assert warnings == []
    assert w2.pane_a.circle() == (801.5, 799.0, 758.0)
    assert w2.pane_a.spin_x.value() == 801.5
    assert w2.controls.trim_b[1].value() == 1.25
    bp = w2.controls.blend()
    assert bp.seam_deg == 88.5 and bp.hard_cut
    v = w2.controls.view()
    assert (v.yaw, v.pitch, v.roll) == (30.0, -5.0, 2.0)
    assert w2.controls.v_yaw.slider.value() == 300
    assert w2.controls.model.currentText() == "polynomial" and w2.controls.k2.value() == 0.05
    assert w2.controls.k1.isEnabled()
    assert w2.preview.chk_seam.isChecked()
    assert w2.export_settings.fmt == "png" and w2.export_settings.save_layers
    for t in (w2._quick, w2._full, w2._compose):
        t.stop()
    w2.flush()
    assert w2.last_preview is not None
    w2.close()


def test_project_with_missing_image_warns_but_loads_settings(app, files, tmp_path):
    paths, _ = files
    w = Workspace()
    w.pane_a.load(paths[0]); w.pane_b.load(paths[1])
    w.controls.seam.setValue(85.0)
    proj = str(tmp_path / "m.spherejoin.json")
    w.save_project_to(proj)
    w.close()
    d = json.load(open(proj))
    d["images"]["b"] = {"path": "/no/such/b.png"}
    json.dump(d, open(proj, "w"))
    w2 = Workspace()
    warnings = w2.load_project_from(proj)
    assert len(warnings) == 1 and "B" in warnings[0]
    assert w2.controls.seam.value() == 85.0 and w2.pane_a.array() is not None and w2.pane_b.array() is None
    w2.close()


def test_export_dialog_defaults_and_recommended_button(app):
    from spherejoin.export import ExportSettings
    from spherejoin.export_dialog import ExportDialog
    d = ExportDialog(None, ExportSettings(width=4000, fmt="tiff"), "/tmp/out_spherejoin_equirect.tif", 6528, False,
                     has_exiftool=False)
    assert d.width_spin.value() == 4000
    d.rec_btn.click()
    assert d.width_spin.value() == 6528
    assert not d.meta.isEnabled() and not d.settings().spherical_metadata
    d.fmt.setCurrentIndex(2)      # JPEG
    assert d.path().endswith(".jpg") and not d.depth16.isEnabled() and d.quality.isEnabled()
    s = d.settings()
    assert s.fmt == "jpg" and s.width % 2 == 0
    d.layers.setChecked(True)
    assert "memory" in d.size_label.text()


def test_projection_controls_state_and_defaults(ws):
    c = ws.controls
    assert c.projection().is_equirect and not c.lat_top.isEnabled()      # this fixture asks for the full sphere
    idx = c.proj_kind.findData("mercator")
    c.proj_kind.setCurrentIndex(idx)
    assert c.lat_top.isEnabled() and (c.lat_top.value(), c.lat_bot.value()) == (75.0, -60.0)
    c.lat_top.setValue(70.0)
    assert c.projection().lat_top_deg == 70.0
    st = c.state_dict()
    assert st["projection"] == {"kind": "mercator", "lat_top_deg": 70.0, "lat_bot_deg": -60.0}
    c.proj_kind.setCurrentIndex(0)
    c.load_state(st)
    assert c.projection().kind == "mercator" and c.lat_top.value() == 70.0 and c.lat_top.isEnabled()
    ws._quick.stop(); ws._full.stop(); ws._compose.stop()


def test_workspace_renders_mercator_with_the_right_shape_and_pick_center_works(ws):
    from spherejoin.reproject import output_size
    ws.controls.proj_kind.setCurrentIndex(ws.controls.proj_kind.findData("mercator"))
    ws.flush()
    proj = ws.controls.projection()
    assert ws.last_preview.shape[:2] == (output_size(proj, PREVIEW_W)[1], PREVIEW_W)
    pw, ph = output_size(proj, PREVIEW_W)
    ws._on_pick(pw * 0.75, ph * 0.5)           # 90 degrees to the right, a little above the horizon row
    v = ws.controls.view()
    assert abs(v.yaw - 90.0) < 1.0
    ws._quick.stop(); ws._full.stop(); ws._compose.stop()
    ws.open_360()                               # the viewer always gets a full-sphere picture
    pm = ws.viewer360.label.pixmap()
    assert pm is not None and not pm.isNull()


def test_project_roundtrip_keeps_the_layout(app, files, tmp_path):
    paths, _ = files
    w = Workspace()
    w.pane_a.load(paths[0]); w.pane_b.load(paths[1])
    w.controls.proj_kind.setCurrentIndex(w.controls.proj_kind.findData("gall"))
    w.controls.lat_bot.setValue(-65.0)
    proj = str(tmp_path / "g.spherejoin.json")
    w.save_project_to(proj)
    w.close()
    w2 = Workspace()
    assert w2.load_project_from(proj) == []
    p = w2.controls.projection()
    assert (p.kind, p.lat_top_deg, p.lat_bot_deg) == ("gall", 85.0, -65.0)
    for t in (w2._quick, w2._full, w2._compose):
        t.stop()
    w2.close()


def test_export_dialog_follows_layout(app):
    from spherejoin.export import ExportSettings
    from spherejoin.export_dialog import ExportDialog
    from spherejoin.reproject import Projection
    d = ExportDialog(None, ExportSettings(width=2000, fmt="tiff", spherical_metadata=True),
                     "/tmp/x_spherejoin_mercator.tif", 6528, False, has_exiftool=True,
                     proj=Projection.default("mercator"))
    assert "2000 x 1065" in d.size_label.text()
    assert not d.meta.isEnabled() and not d.settings().spherical_metadata
    d2 = ExportDialog(None, ExportSettings(width=2000, fmt="tiff"), "/tmp/y.tif", 6528, False, has_exiftool=True)
    assert "2000 x 1000" in d2.size_label.text() and d2.meta.isEnabled()


def test_double_click_a_label_resets_that_control(ws):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLabel
    c = ws.controls
    c.expand_all()
    def label_for(field):
        for lab in c.findChildren(QLabel):
            if lab.buddy() is field:
                return lab
        raise AssertionError("no label for that control")
    def dbl(field):
        QTest.mouseDClick(label_for(field), Qt.LeftButton)

    c.fov.setValue(200.0); dbl(c.fov); assert c.fov.value() == 220.0
    c.k1.setValue(2.0); c.k2.setValue(0.3); c.k3.setValue(-0.2)
    for k, want in ((c.k1, 1.0), (c.k2, 0.0), (c.k3, 0.0)):
        dbl(k); assert k.value() == want
    for sp in list(c.trim_a) + list(c.trim_b):
        sp.setValue(1.7); dbl(sp); assert sp.value() == 0.0
    c.v_pitch.setValue(33.0); dbl(c.v_pitch)
    assert c.v_pitch.value() == 0.0 and c.v_pitch.slider.value() == 0
    c.seam.setValue(80.0); dbl(c.seam); assert c.seam.value() == 90.0
    c.feather.setValue(5.0); dbl(c.feather); assert c.feather.value() == 20.0
    c.model.setCurrentText("stereographic"); dbl(c.model); assert c.model.currentText() == "equidistant"
    c.proj_kind.setCurrentIndex(c.proj_kind.findData("mercator")); c.lat_top.setValue(60.0)
    dbl(c.lat_top); assert c.lat_top.value() == 75.0
    c.lat_bot.setValue(-30.0); dbl(c.lat_bot); assert c.lat_bot.value() == -60.0
    c.proj_kind.setCurrentIndex(c.proj_kind.findData("gall")); dbl(c.proj_kind); assert c.projection().kind == "mercator"
    # an untouched control stays put when a different label is double-clicked
    c.feather.setValue(7.0); dbl(c.seam); assert c.feather.value() == 7.0
    ws._quick.stop(); ws._full.stop(); ws._compose.stop()


def test_double_click_a_view_slider_resets_it(ws):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    c = ws.controls
    c.expand_all()
    c.v_yaw.setValue(75.0)
    QTest.mouseDClick(c.v_yaw.slider, Qt.LeftButton)
    assert c.v_yaw.value() == 0.0 and c.v_yaw.slider.value() == 0
    ws._quick.stop(); ws._full.stop(); ws._compose.stop()


def test_painting_on_the_preview_changes_the_mask_with_undo_and_clear(ws):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    ws.flush()
    assert ws.seam_mask.is_empty
    ws.preview.paint_btn.setChecked(True)
    ws.preview.use_a.setChecked(True)
    ws.preview.brush_slider.setValue(40)
    # A stroke through the real viewer: press, drag, release.
    vp = ws.preview.view.viewport()
    c = vp.rect().center()
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, QPoint(c.x() - 60, c.y()))
    QTest.mouseMove(vp, QPoint(c.x(), c.y()))
    QTest.mouseMove(vp, QPoint(c.x() + 60, c.y()))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, QPoint(c.x() + 60, c.y()))
    assert not ws.seam_mask.is_empty and ws.seam_mask.paint.max() > 0.9 and ws.seam_mask.paint.min() == 0.0
    painted = ws.seam_mask.paint.copy()
    # Painting "use B" over part of it, then undo, restores the earlier state.
    ws.preview.use_b.setChecked(True)
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, QPoint(c.x(), c.y()))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, QPoint(c.x(), c.y()))
    assert ws.seam_mask.paint.min() < -0.5
    ws.undo_paint()
    assert np.allclose(ws.seam_mask.paint, painted, atol=1e-3)
    ws.clear_mask()
    assert ws.seam_mask.is_empty
    ws.undo_paint()
    assert not ws.seam_mask.is_empty
    for t in (ws._quick, ws._full, ws._compose):
        t.stop()


def test_quick_double_dabs_do_not_trigger_fit_to_window(ws):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    ws.flush()
    ws.preview.view.zoom_to_pixel(500, 400, 3.0)
    z = ws.preview.view.zoom_level()
    ws.preview.paint_btn.setChecked(True)
    vp = ws.preview.view.viewport()
    QTest.mouseDClick(vp, Qt.LeftButton, Qt.NoModifier, vp.rect().center())
    assert abs(ws.preview.view.zoom_level() - z) < 1e-9
    assert not ws.seam_mask.is_empty
    for t in (ws._quick, ws._full, ws._compose):
        t.stop()


def test_pick_and_paint_modes_exclude_each_other_and_pan_still_works(ws):
    ws.preview.pick_btn.setChecked(True)
    ws.preview.paint_btn.setChecked(True)
    assert ws.preview.paint_btn.isChecked() and not ws.preview.pick_btn.isChecked()
    ws.preview.pick_btn.setChecked(True)
    assert ws.preview.pick_btn.isChecked() and not ws.preview.paint_btn.isChecked()
    ws.preview.pick_btn.setChecked(False)


def test_mask_is_saved_with_the_project_and_comes_back(app, files, tmp_path):
    paths, _ = files
    w = Workspace()
    w.pane_a.load(paths[0]); w.pane_b.load(paths[1])
    w.seam_mask.stamp(w.controls.view(), w.controls.projection(), PREVIEW_W, PREVIEW_H, 900.0, 520.0, 70.0, "a")
    w.seam_mask.stamp(w.controls.view(), w.controls.projection(), PREVIEW_W, PREVIEW_H, 1500.0, 520.0, 70.0, "b")
    before = w.seam_mask.paint.copy()
    proj = str(tmp_path / "pm.spherejoin.json")
    w.save_project_to(proj)
    assert (tmp_path / "pm.seammask.png").exists()
    assert json.load(open(proj))["seam_mask"] == "pm.seammask.png"
    w.close()
    w2 = Workspace()
    assert w2.load_project_from(proj) == []
    assert np.abs(w2.seam_mask.paint - before).max() <= 1 / 127 + 1e-6
    # A project saved without paint has no mask file reference, and loading it clears an old mask.
    w2.seam_mask.clear()
    p2 = str(tmp_path / "plain.spherejoin.json")
    w2.save_project_to(p2)
    assert "seam_mask" not in json.load(open(p2)) and not (tmp_path / "plain.seammask.png").exists()
    w3 = Workspace()
    w3.seam_mask.stamp(w3.controls.view(), w3.controls.projection(), PREVIEW_W, PREVIEW_H, 900.0, 520.0, 70.0, "a")
    w3.load_project_from(p2)
    assert w3.seam_mask.is_empty
    for x in (w2, w3):
        for t in (x._quick, x._full, x._compose):
            t.stop()
        x.close()


def test_missing_mask_file_warns_but_loads_the_rest(app, files, tmp_path):
    paths, _ = files
    w = Workspace()
    w.pane_a.load(paths[0]); w.pane_b.load(paths[1])
    w.seam_mask.stamp(w.controls.view(), w.controls.projection(), PREVIEW_W, PREVIEW_H, 900.0, 520.0, 70.0, "a")
    w.controls.seam.setValue(84.0)
    proj = str(tmp_path / "mm.spherejoin.json")
    w.save_project_to(proj)
    w.close()
    (tmp_path / "mm.seammask.png").unlink()
    w2 = Workspace()
    warnings = w2.load_project_from(proj)
    assert any("seam mask" in x for x in warnings) and w2.seam_mask.is_empty and w2.controls.seam.value() == 84.0
    for t in (w2._quick, w2._full, w2._compose):
        t.stop()
    w2.close()


def test_export_dialog_mask_checkbox(app):
    from spherejoin.export import ExportSettings
    from spherejoin.export_dialog import ExportDialog
    d = ExportDialog(None, ExportSettings(width=2000), "/tmp/x.tif", 6528, False, has_exiftool=False, has_mask=False)
    assert not d.use_mask.isEnabled() and d.settings().use_mask       # nothing painted: harmless default
    d2 = ExportDialog(None, ExportSettings(width=2000, use_mask=False), "/tmp/x.tif", 6528, False,
                      has_exiftool=False, has_mask=True)
    assert d2.use_mask.isEnabled() and not d2.use_mask.isChecked() and not d2.settings().use_mask
    d2.use_mask.setChecked(True)
    assert d2.settings().use_mask


def test_a_fresh_session_starts_in_mercator_and_the_layout_label_resets_to_it(app):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLabel
    w = Workspace()
    c = w.controls
    c.expand_all()
    p = c.projection()
    assert (p.kind, p.lat_top_deg, p.lat_bot_deg) == ("mercator", 75.0, -60.0)
    assert c.lat_top.isEnabled() and c.lat_top.value() == 75.0 and c.lat_bot.value() == -60.0
    c.proj_kind.setCurrentIndex(c.proj_kind.findData("gall"))
    label = next(l for l in c.findChildren(QLabel) if l.buddy() is c.proj_kind)
    QTest.mouseDClick(label, Qt.LeftButton)
    assert c.projection().kind == "mercator" and c.lat_top.value() == 75.0
    # An old project saved before layouts existed has no layout entry and keeps the full-sphere picture it was made with.
    c.load_state({"lens": {"fov_deg": 220.0}})
    assert c.projection().is_equirect
    w.close()


def test_brightness_and_white_balance_are_not_in_the_interface_and_stay_neutral(app):
    from PySide6.QtWidgets import QLabel
    w = Workspace()
    c = w.controls
    assert not any("Gain" in lab.text() or "red / blue" in lab.text() for lab in c.findChildren(QLabel))
    assert not any("Brightness" in s.header.text() for s in c._sections)
    bp = c.blend()
    assert (bp.gain_a, bp.gain_b, bp.wb_a, bp.wb_b) == (1.0, 1.0, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    # An older project that stored other values does not apply them invisibly.
    c.load_state({"blend": {"seam_deg": 85.0, "gain_b": 0.5, "wb_a": [1.3, 1.0, 0.7]}})
    bp = c.blend()
    assert bp.seam_deg == 85.0 and bp.gain_b == 1.0 and bp.wb_a == (1.0, 1.0, 1.0)
    st = c.state_dict()["blend"]
    assert "gain_a" not in st and "wb_a" not in st
    w.close()


def test_the_last_two_photos_reopen_at_the_next_start(app, files, tmp_path):
    paths, _ = files
    w = Workspace()
    assert w.open_last_files() == []                     # nothing remembered yet: nothing happens
    w.pane_a.load(paths[0]); w.pane_b.load(paths[1])
    w.close()
    w2 = Workspace()                                     # a new session, same remembered state
    assert not w2.ready()
    assert w2.open_last_files() == [paths[0], paths[1]]
    assert w2.ready() and w2.pane_a.path == paths[0] and w2.pane_b.path == paths[1]
    assert w2.pane_a.circle() is not None                # the circle is found again as usual
    for t in (w2._quick, w2._full, w2._compose):
        t.stop()
    w2.close()


def test_a_missing_last_file_is_skipped_quietly_and_the_other_still_opens(app, files, tmp_path):
    import shutil
    paths, _ = files
    a_copy = str(tmp_path / "gone_a.png"); b_copy = str(tmp_path / "kept_b.png")
    shutil.copy(paths[0], a_copy); shutil.copy(paths[1], b_copy)
    w = Workspace()
    w.pane_a.load(a_copy); w.pane_b.load(b_copy)
    w.close()
    os_remove = __import__("os").remove
    os_remove(a_copy)                                    # the first photo has moved away (or its drive is unplugged)
    w2 = Workspace()
    assert w2.open_last_files() == [b_copy]
    assert w2.pane_a.array() is None and w2.pane_b.array() is not None
    for t in (w2._quick, w2._full, w2._compose):
        t.stop()
    w2.close()


def test_loading_a_different_photo_updates_what_is_remembered(app, files, tmp_path):
    import shutil
    paths, _ = files
    other = str(tmp_path / "other.png"); shutil.copy(paths[1], other)
    w = Workspace()
    w.pane_a.load(paths[0]); w.pane_b.load(paths[1])
    w.pane_b.load(other)                                 # swapped the back photo
    w.close()
    w2 = Workspace()
    assert w2.open_last_files() == [paths[0], other]
    for t in (w2._quick, w2._full, w2._compose):
        t.stop()
    w2.close()


def test_a_last_photo_that_cannot_be_read_gives_no_error_box_at_start(app, files, monkeypatch):
    """Reopening last time's photos must never pop an error (the drive may be off, or macOS may block it)."""
    from PySide6.QtWidgets import QMessageBox
    paths, _ = files
    w = Workspace()
    w.pane_a.load(paths[0]); w.pane_b.load(paths[1])
    w.close()
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a))
    import spherejoin.panes as panes_mod
    def boom(path):
        raise PermissionError(1, "Operation not permitted", path)
    monkeypatch.setattr(panes_mod, "load_image", boom)
    w2 = Workspace()
    assert w2.open_last_files() == []
    assert shown == []
    w2.close()
