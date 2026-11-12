"""One photo with both lenses: the Input switch, splitting, the X4 preset, and putting settings back."""
import cv2
import numpy as np
import pytest
from PIL import Image

from spherejoin import dual
from spherejoin.workspace import Workspace


def make_insp(path, model="Insta360 X4", w=400, h=200):
    """A 2:1 picture with two bright discs, saved as a JPEG with a camera model in the EXIF."""
    img = np.zeros((h, w, 3), np.uint8)
    for cx in (w // 4, 3 * w // 4):
        cv2.circle(img, (cx, h // 2), int(h * 0.485), (90, 90, 90), -1)
    im = Image.fromarray(img)
    exif = Image.Exif()
    exif[0x0110] = model
    im.save(str(path), "JPEG", exif=exif)
    return str(path)


def test_split_halves():
    a = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    left, right = dual.split_halves(a)
    assert left.shape == (6, 4, 3) and right.shape == (6, 4, 3)
    assert np.array_equal(left, a[:, :4]) and np.array_equal(right, a[:, 4:])


def test_preset_found_from_exif(tmp_path):
    assert dual.preset_for(make_insp(tmp_path / "a.insp")).name == "Insta360 X4"
    assert dual.preset_for(make_insp(tmp_path / "b.jpg", model="Some Other Camera")) is None


def test_mode_switch_clears_and_drop_fills_both(tmp_path, app):
    ws = Workspace()
    assert ws.input_mode == "two"
    ws.input_box.setCurrentIndex(1)
    assert ws.input_mode == "one"
    ws._on_files_dropped("a", [make_insp(tmp_path / "x.insp")])
    assert ws.pane_a.array().shape[:2] == (200, 200) and ws.pane_b.array().shape[:2] == (200, 200)
    ca, cb = ws.pane_a.circle(), ws.pane_b.circle()
    assert abs(ca[0] - 100.0) < 0.1 and abs(ca[1] - 100.0) < 0.1 and abs(ca[2] - 0.4855 * 200) < 0.2
    assert abs(cb[0] - 100.0) < 0.5 and abs(cb[1] - 100.0) < 0.5
    assert ws.ready()
    ws.input_box.setCurrentIndex(0)
    assert ws.pane_a.array() is None and ws.pane_b.array() is None and not ws.ready()


def test_x4_settings_applied_then_handed_back(tmp_path, app):
    ws = Workspace()
    ws.controls.fov.setValue(220.0)
    ws.input_box.setCurrentIndex(1)
    ws.load_dual(make_insp(tmp_path / "x.insp"))
    assert abs(ws.controls.fov.value() - 202.1) < 1e-9
    assert abs(ws.controls.trim_b[1].value() - 1.376) < 0.006 and ws.controls.trim_a[1].value() == 0.0
    ws.input_box.setCurrentIndex(0)                        # back to two photos: the earlier settings return
    assert abs(ws.controls.fov.value() - 220.0) < 1e-9
    assert ws.controls.trim_b[0].value() == 0.0


def test_other_camera_keeps_lens_settings_and_uses_detection(tmp_path, app):
    ws = Workspace()
    ws.input_box.setCurrentIndex(1)
    ws.controls.fov.setValue(215.0)
    ws.load_dual(make_insp(tmp_path / "o.jpg", model="Some Other Camera"))
    assert abs(ws.controls.fov.value() - 215.0) < 1e-9
    assert ws.pane_a.circle() is not None            # auto-detect gave a starting circle


def test_choice_is_remembered(app):
    ws = Workspace()
    ws.input_box.setCurrentIndex(1)
    assert Workspace().input_mode == "one"
    ws2 = Workspace()
    ws2.input_box.setCurrentIndex(0)
    assert Workspace().input_mode == "two"


def test_per_lens_circles_from_preset(tmp_path, app):
    ws = Workspace()
    ws.input_box.setCurrentIndex(1)
    ws.load_dual(make_insp(tmp_path / "x.insp", w=11904, h=5952))
    (ax, ay, ar), (bx, by, br) = ws.pane_a.circle(), ws.pane_b.circle()
    assert abs(ax - 2975.4) < 0.2 and abs(ay - 2976.0) < 0.2
    assert abs(bx - 2969.1) < 0.2 and abs(by - 2970.1) < 0.2 and abs(ar - 2890.0) < 0.2 and abs(br - 2890.0) < 0.2


def test_exposure_gains_make_the_strip_match():
    from spherejoin.exposure import gains_from_layers
    from spherejoin.projection import LensParams
    h, w = 100, 100
    la = np.full((h, w, 3), 100.0, np.float32)
    lb = np.full((h, w, 3), 80.0, np.float32) * np.array([1.0, 1.0, 0.9], np.float32)
    th = np.full((h, w), 90.0, np.float32)
    lens = LensParams(0, 0, 1, fov_deg=202.0)
    ga, gb = gains_from_layers(la, lb, th, th, lens, lens)
    assert np.allclose(la[0, 0] * ga, lb[0, 0] * gb, rtol=1e-4)
    assert np.all(ga < 1.0) and np.all(gb > 1.0)


def test_sky_gains_are_set_and_ground_is_left_alone(tmp_path, app):
    import numpy as np
    from spherejoin.blend import sky_factor
    ws = Workspace()
    ws.input_box.setCurrentIndex(1)
    ws.load_dual(make_insp(tmp_path / "x.insp"))
    bp = ws.blend_params()
    assert len(bp.sky_gain_a) == 3 and len(bp.sky_gain_b) == 3
    f = sky_factor((1.2, 1.2, 1.2), np.array([-20.0, 0.0, 30.0]))      # ground, horizon, sky
    assert np.allclose(f[0], 1.0) and np.allclose(f[2], 1.2) and np.all(f[1] <= 1.2)


def test_one_photo_mode_shows_a_single_drop_area_until_a_photo_is_loaded(tmp_path, app):
    ws = Workspace()
    assert ws.source_stack.currentWidget() is not ws.dual_drop          # two-photo mode: the two pictures
    ws.input_box.setCurrentIndex(1)
    assert ws.source_stack.currentWidget() is ws.dual_drop              # one photo: one drop area
    ws.dual_drop.files_dropped.emit([make_insp(tmp_path / "x.insp")])
    assert ws.source_stack.currentWidget() is not ws.dual_drop          # loaded: the two halves
    assert ws.pane_a.array() is not None and ws.pane_b.array() is not None
    ws.input_box.setCurrentIndex(0)
    assert ws.source_stack.currentWidget() is not ws.dual_drop
    ws.input_box.setCurrentIndex(1)
    assert ws.source_stack.currentWidget() is ws.dual_drop              # cleared again on the way in


def test_x4_recognized_by_size_when_the_camera_tag_is_gone(tmp_path):
    from PIL import Image
    p = str(tmp_path / "stacked.jpg")
    Image.new("RGB", (8, 4)).save(p)                      # no camera tag at all
    assert dual.preset_for(p) is None
    assert dual.preset_for(p, (11904, 5952)).name == "Insta360 X4"
    assert dual.preset_for(p, (4000, 2000)) is None
    other = make_insp(tmp_path / "o.jpg", model="Some Other Camera")
    assert dual.preset_for(other, (11904, 5952)) is None   # a named different camera is never overridden


def test_dragging_the_preview_turns_yaw_and_leaves_pitch(qapp_or_none=None):
    from spherejoin.reproject import output_size
    from spherejoin.workspace import PREVIEW_W
    ws = Workspace()
    ws.controls.set_view(10.0, 7.0)
    pw, ph = output_size(ws.controls.projection(), PREVIEW_W)
    start = ws.controls.view()
    ws._on_view_drag(pw / 2, ph / 2, 0)
    ws._on_view_drag(pw / 2 + 100, ph / 2, 1)
    moved = ws.controls.view()
    assert moved.pitch == pytest.approx(start.pitch, abs=0.05)
    assert moved.roll == pytest.approx(start.roll, abs=0.05)
    assert abs(moved.yaw - start.yaw) == pytest.approx(100 * 360.0 / pw, abs=0.2)
    ws._on_view_drag(pw / 2, ph / 2, 2)                       # back where it started
    assert ws.controls.view().yaw == pytest.approx(start.yaw, abs=0.2)


def test_double_click_on_the_preview_zooms_in_then_back_to_fit():
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    ws = Workspace()
    ws.resize(900, 700)
    ws.show()
    view = ws.preview.view
    view.set_image(np.zeros((540, 1024, 3), np.uint8))
    view.fit_to_window()
    fit = view.zoom_level()
    QTest.mouseDClick(view.viewport(), Qt.LeftButton)
    assert view.zoom_level() > fit * 1.5
    QTest.mouseDClick(view.viewport(), Qt.LeftButton)
    assert view.zoom_level() == pytest.approx(fit, rel=0.02)
