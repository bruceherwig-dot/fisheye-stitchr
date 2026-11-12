import numpy as np
import pytest

from spherejoin.blend import (
    SKY_LAT_HI, SKY_LAT_LO, BlendParams, blend_weight_a, compose, feather_array, world_lat_deg,
)
from spherejoin.preview import compose_preview, warp_preview_layers
from spherejoin.projection import LensParams, ViewParams
from spherejoin.reproject import EQUIRECT, Projection
from spherejoin.synth import make_test_equirect, render_fisheye_from_equirect

A_ = LensParams(cx=450.0, cy=450.0, radius=420.0, base_yaw=0.0)
B_ = LensParams(cx=450.0, cy=450.0, radius=420.0, base_yaw=180.0)


def test_feather_is_ground_width_low_sky_width_high_and_smooth_between():
    bp = BlendParams(feather_deg=20.0, sky_feather_deg=3.0)
    lat = np.array([-30.0, 0.0, SKY_LAT_LO, (SKY_LAT_LO + SKY_LAT_HI) / 2, SKY_LAT_HI, 40.0, 85.0])
    f = feather_array(bp, lat)
    assert f[0] == f[1] == f[2] == pytest.approx(20.0)
    assert f[4] == f[5] == f[6] == pytest.approx(3.0)
    assert 3.0 < f[3] < 20.0 and f[3] == pytest.approx(11.5)           # halfway band: halfway width
    assert np.all(np.diff(f) <= 1e-6)                                    # never widens as you go up
    assert feather_array(bp, None) == 20.0                               # no latitude known: the ground width
    assert feather_array(BlendParams(hard_cut=True), lat) == 0.0


def test_sky_equal_to_ground_gives_exactly_the_old_uniform_blend():
    th = np.linspace(60, 120, 61)[None, :].astype(np.float32)
    thb = 180.0 - th
    lat = np.linspace(-40, 80, 61)[None, :].astype(np.float32)
    uniform = BlendParams(feather_deg=20.0, sky_feather_deg=20.0)
    a, _ = blend_weight_a(th, thb, A_, B_, uniform, None, lat)
    b, _ = blend_weight_a(th, thb, A_, B_, uniform, None, None)
    assert np.array_equal(a, b)


def test_a_given_angle_off_the_seam_blends_on_the_ground_but_is_decided_in_the_sky():
    bp = BlendParams(feather_deg=20.0, sky_feather_deg=3.0)
    th = np.full((1, 2), 94.0, np.float32)        # 4 degrees past the seam
    thb = 180.0 - th
    lat = np.array([[-20.0, 40.0]], np.float32)   # one pixel on the ground, one in the sky
    w, _ = blend_weight_a(th, thb, A_, B_, bp, None, lat)
    assert 0.05 < w[0, 0] < 0.95                  # still inside the wide ground fade
    assert w[0, 1] == 0.0                         # already fully image B in the narrow sky fade (half width 1.5)


def test_world_latitude_does_not_move_when_the_view_turns():
    # The center of the picture looks at the world latitude equal to the view pitch.
    for pitch in (0.0, 25.0, -30.0):
        lat = world_lat_deg(ViewParams(pitch=pitch), EQUIRECT, 360, 180)
        assert lat[90, 180] == pytest.approx(pitch, abs=0.6)
    # And a direction keeps its latitude under yaw and roll about the vertical-ish axes.
    a = world_lat_deg(ViewParams(), EQUIRECT, 360, 180)
    b = world_lat_deg(ViewParams(yaw=77.0), EQUIRECT, 360, 180)
    assert np.allclose(np.sort(a, axis=None), np.sort(b, axis=None), atol=1e-4)
    m = world_lat_deg(ViewParams(), Projection.default("mercator"), 400, 240)
    assert m[0].min() > m[-1].max() and m.max() <= 80.5 and m.min() >= -60.5


@pytest.fixture(scope="module")
def world():
    equi = make_test_equirect(1024, 512)
    fa = render_fisheye_from_equirect(equi, A_, (900, 900))
    fb = render_fisheye_from_equirect(equi, B_, (900, 900))
    return equi, fa, fb


def blend_weight_columns(fa, fb, bp, rows, W=1024, H=512, lo=0.08, hi=0.92):
    """How many columns around the lon +90 seam are mid-fade, read straight from the finished picture.

    Rendering twice with image B at gain 1 and gain 2 gives, per pixel, ratio = w + 2 (1 - w) = 2 - w when the
    two photos agree, so w = 2 - ratio whatever the picture contains."""
    from dataclasses import replace
    fa, fb = fa // 2, fb // 2                 # headroom, so doubling image B never clips
    o1 = compose(fa, fb, A_, B_, ViewParams(), replace(bp, gain_b=1.0), W, H).astype(float).mean(axis=2)
    o2 = compose(fa, fb, A_, B_, ViewParams(), replace(bp, gain_b=2.0), W, H).astype(float).mean(axis=2)
    x0, x1 = int(W * (0.5 + 50 / 360)), int(W * (0.5 + 130 / 360))
    out = []
    for r in rows:
        a = o1[r - 3:r + 4, x0:x1].mean(axis=0)
        b = o2[r - 3:r + 4, x0:x1].mean(axis=0)
        w = 2.0 - b / np.maximum(a, 1.0)
        out.append(int(((w > lo) & (w < hi)).sum()))
    return out


def test_the_sky_transition_is_narrow_and_the_ground_transition_wide_in_the_picture(world):
    equi, fa, fb = world
    H = 512
    ground_row = int(H * (0.5 + 15 / 180))     # latitude -15
    sky_row = int(H * (0.5 - 30 / 180))        # latitude +30
    g, s = blend_weight_columns(fa, fb, BlendParams(feather_deg=24.0, sky_feather_deg=3.0), (ground_row, sky_row))
    print("columns mid-fade: ground", g, "sky", s)
    assert g > 40 and s < 14 and g > 3 * s
    # Setting the sky width equal to the ground width makes the two rows blend alike again.
    g2, s2 = blend_weight_columns(fa, fb, BlendParams(feather_deg=24.0, sky_feather_deg=24.0), (ground_row, sky_row))
    assert s2 > 3 * s and abs(g2 - g) <= 6


def test_the_preview_uses_the_same_blend_as_the_export(world):
    equi, fa, fb = world
    W, H = 1024, 512
    bp = BlendParams(feather_deg=24.0, sky_feather_deg=3.0)
    L = warp_preview_layers(fa, fb, A_, B_, ViewParams(), W, H)
    assert L.lat is not None and L.lat.shape == (H, W)
    prev = compose_preview(L, A_, B_, bp)
    exp = compose(fa, fb, A_, B_, ViewParams(), bp, W, H, interp="linear")
    assert np.abs(prev.astype(int) - exp.astype(int)).mean() < 1.0
    seam = compose_preview(L, A_, B_, bp, show_seam=True)               # the lines follow the narrowing blend
    assert (seam != prev).any()


def test_turning_the_view_does_not_change_which_parts_are_sky(world):
    equi, fa, fb = world
    W, H = 1024, 512
    bp = BlendParams(feather_deg=24.0, sky_feather_deg=3.0, gain_b=2.0)
    base = compose(fa, fb, A_, B_, ViewParams(), bp, W, H)
    turned = compose(fa, fb, A_, B_, ViewParams(yaw=90.0), bp, W, H)
    expect = np.roll(base, -W // 4, axis=1)
    assert np.abs(turned.astype(int) - expect.astype(int))[8:-8].mean() < 2.0


def test_old_project_without_a_sky_value_gets_the_default(app):
    from spherejoin.workspace import Workspace
    w = Workspace()
    c = w.controls
    assert c.blend().sky_feather_deg == 7.0
    c.sky_feather.setValue(8.0)
    assert c.state_dict()["blend"]["sky_feather_deg"] == 8.0
    c.load_state({"blend": {"seam_deg": 88.0, "feather_deg": 15.0}})          # saved before the sky value existed
    assert c.sky_feather.value() == 7.0 and c.feather.value() == 15.0
    c.load_state({"blend": {"sky_feather_deg": 6.5}})
    assert c.sky_feather.value() == 6.5
    w.close()


def test_double_clicking_the_sky_label_resets_it(app):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLabel
    from spherejoin.workspace import Workspace
    w = Workspace()
    w.show()
    c = w.controls
    c.expand_all()
    c.sky_feather.setValue(17.0)
    label = next(l for l in c.findChildren(QLabel) if l.buddy() is c.sky_feather)
    assert "Sky" in label.text()
    QTest.mouseDClick(label, Qt.LeftButton)
    assert c.sky_feather.value() == 7.0
    w.close()
