import numpy as np
import pytest

from spherejoin.blend import BlendParams, blend_weight_a, compose, overlap_difference, overlap_info
from spherejoin.projection import LensParams, ViewParams
from spherejoin.synth import make_test_equirect, render_fisheye_from_equirect

W, H = 1024, 512


def lens_pair(**trim_b):
    kw = dict(cx=450.0, cy=450.0, radius=420.0, fov_deg=220.0)
    return LensParams(base_yaw=0.0, **kw), LensParams(base_yaw=180.0, **trim_b, **kw)


@pytest.fixture(scope="module")
def world():
    equi = make_test_equirect(W, H)
    A, B = lens_pair()
    fa = render_fisheye_from_equirect(equi, A, (900, 900))
    fb = render_fisheye_from_equirect(equi, B, (900, 900))
    return equi, A, B, fa, fb


def _err(a, b, rows=slice(8, H - 8)):
    return np.abs(a.astype(np.int16) - b.astype(np.int16))[rows].mean()


def test_weights_hard_cut_and_feather_shape():
    A, B = lens_pair()
    th = np.linspace(0, 180, 361)[None, :].astype(np.float32)
    thb = 180.0 - th
    w_hard, _ = blend_weight_a(th, thb, A, B, BlendParams(seam_deg=90, hard_cut=True))
    assert set(np.unique(w_hard)) <= {0.0, 1.0}
    assert w_hard[0, th[0] <= 90].min() == 1.0 and w_hard[0, th[0] > 90].max() == 0.0
    w, _ = blend_weight_a(th, thb, A, B, BlendParams(seam_deg=90, feather_deg=20))
    inside = (th[0] > 80) & (th[0] < 100)
    assert np.all(np.diff(w[0, inside]) <= 1e-7)            # steadily falling
    assert abs(w[0, np.argmin(abs(th[0] - 90))] - 0.5) < 1e-6  # half and half at the seam
    assert w[0, th[0] <= 80].min() == 1.0 and w[0, th[0] >= 100].max() == 0.0


def test_only_one_lens_valid_forces_that_lens():
    A, B = lens_pair()
    th = np.array([[10.0, 150.0]], np.float32)       # A sees the first, B sees the second
    thb = np.array([[170.0, 30.0]], np.float32)
    w, anyv = blend_weight_a(th, thb, A, B, BlendParams(seam_deg=90, feather_deg=40))
    assert w[0, 0] == 1.0 and w[0, 1] == 0.0 and anyv.all()


def test_feathered_stitch_rebuilds_panorama(world):
    equi, A, B, fa, fb = world
    for bp in (BlendParams(feather_deg=20), BlendParams(hard_cut=True)):
        out = compose(fa, fb, A, B, ViewParams(), bp, W, H)
        e = _err(out, equi)
        print("feather" if not bp.hard_cut else "hard", "mean abs err", round(float(e), 3))
        assert e < 2.0 and out.dtype == np.uint8


def test_gain_darkens_only_b_side_and_fades_in_overlap(world):
    equi, A, B, fa, fb = world
    base = compose(fa, fb, A, B, ViewParams(), BlendParams(feather_deg=20), W, H).astype(np.float32)
    dim = compose(fa, fb, A, B, ViewParams(), BlendParams(feather_deg=20, gain_b=0.5), W, H).astype(np.float32)
    mid_row = H // 2
    ratio = dim[mid_row].mean(axis=1) / np.maximum(base[mid_row].mean(axis=1), 1.0)
    center = ratio[W // 2]              # A side (lon 0)
    back = ratio[10]                    # B side (lon about -180)
    assert abs(center - 1.0) < 0.01 and abs(back - 0.5) < 0.02
    seam_x = int(W * (0.5 + 90 / 360))  # lon 90 degrees, middle of the fade
    assert 0.65 < ratio[seam_x] < 0.85


def test_white_balance_changes_only_that_channel(world):
    equi, A, B, fa, fb = world
    base = compose(fa, fb, A, B, ViewParams(), BlendParams(), W, H).astype(np.float32)
    wb = compose(fa, fb, A, B, ViewParams(), BlendParams(wb_a=(1.0, 1.0, 0.5)), W, H).astype(np.float32)
    c = slice(W // 2 - 20, W // 2 + 20)   # firmly A territory
    assert np.allclose(wb[:, c, 0], base[:, c, 0], atol=1) and np.allclose(wb[:, c, 1], base[:, c, 1], atol=1)
    assert wb[:, c, 2].mean() < 0.55 * base[:, c, 2].mean() + 2


def test_view_yaw_90_slides_panorama_quarter_turn(world):
    equi, A, B, fa, fb = world
    out = compose(fa, fb, A, B, ViewParams(yaw=90), BlendParams(), W, H)
    expect = np.roll(equi, -W // 4, axis=1)   # lon +90 (the green blob) now at the center
    assert _err(out, expect) < 2.0
    assert out[H // 2, W // 2, 1] > 200 and out[H // 2, W // 2, 0] < 60


def test_view_pitch_puts_up_blob_in_center(world):
    equi, A, B, fa, fb = world
    out = compose(fa, fb, A, B, ViewParams(pitch=60), BlendParams(), W, H)
    r, g, b = (int(v) for v in out[H // 2, W // 2])   # blob at lat +60, lon 0 is yellow
    assert r > 200 and g > 200 and b < 80


def test_view_roll_keeps_center(world):
    equi, A, B, fa, fb = world
    a = compose(fa, fb, A, B, ViewParams(), BlendParams(), W, H)
    b = compose(fa, fb, A, B, ViewParams(roll=30), BlendParams(), W, H)
    assert np.abs(a[H // 2, W // 2].astype(int) - b[H // 2, W // 2].astype(int)).max() < 12
    assert _err(a, b) > 5     # but the rest really did rotate


def test_matching_trim_reproduces_tilted_lens():
    equi = make_test_equirect(W, H)
    A, B0 = lens_pair()
    _, B_tilt = lens_pair(trim_pitch=1.5)
    fa = render_fisheye_from_equirect(equi, A, (900, 900))
    fb = render_fisheye_from_equirect(equi, B_tilt, (900, 900))     # B was really tilted 1.5 degrees
    hard = BlendParams(hard_cut=True)
    wrong = compose(fa, fb, A, B0, ViewParams(), hard, W, H)
    right = compose(fa, fb, A, B_tilt, ViewParams(), hard, W, H)
    back_cols = np.r_[0:W // 8, 7 * W // 8:W]    # well inside B territory
    e_wrong = np.abs(wrong.astype(int) - equi.astype(int))[8:-8][:, back_cols].mean()
    e_right = np.abs(right.astype(int) - equi.astype(int))[8:-8][:, back_cols].mean()
    print("trim error: untrimmed", round(float(e_wrong), 2), "trimmed", round(float(e_right), 2))
    assert e_right < 2.0 and e_wrong > 3 * e_right


def test_tiling_does_not_change_result(world):
    equi, A, B, fa, fb = world
    bp = BlendParams(feather_deg=24, gain_b=0.9)
    a = compose(fa, fb, A, B, ViewParams(yaw=20, pitch=5), bp, 512, 256, tile_rows=37)
    b = compose(fa, fb, A, B, ViewParams(yaw=20, pitch=5), bp, 512, 256, tile_rows=256)
    assert np.array_equal(a, b)


def test_16bit_stays_16bit_and_overlap_tools(world):
    equi, A, B, fa, fb = world
    fa16, fb16 = fa.astype(np.uint16) * 257, fb.astype(np.uint16) * 257
    out = compose(fa16, fb16, A, B, ViewParams(), BlendParams(), W, H)
    assert out.dtype == np.uint16 and out.max() > 255 * 200
    tha, thb, ov = overlap_info(A, B, ViewParams(), 360, 180)
    assert ov.any() and not ov.all()
    d = overlap_difference(fa, fb, A, B, ViewParams(), BlendParams(), 360, 180)
    assert d.shape == (180, 360) and d[~ov].max() == 0.0
    d_gain = overlap_difference(fa, fb, A, B, ViewParams(), BlendParams(gain_b=0.5), 360, 180)
    assert d_gain[ov].mean() > d[ov].mean()


def test_mismatched_inputs_rejected(world):
    equi, A, B, fa, fb = world
    with pytest.raises(ValueError):
        compose(fa, fb.astype(np.uint16), A, B, ViewParams(), BlendParams(), 64, 32)


def test_seam_sits_where_both_lenses_are_the_same_angle():
    """With lens B turned, the seam stays in the middle of the overlap (same angle off both axes)."""
    from spherejoin.blend import BlendParams, blend_weight_a, seam_coordinate
    from spherejoin.projection import LensParams
    lens = LensParams(0, 0, 1, fov_deg=220.0)
    bp = BlendParams(hard_cut=True)
    # exactly opposite lenses: theta_a + theta_b = 180, so the coordinate is theta_a
    assert np.allclose(seam_coordinate(np.array([70.0, 90.0, 100.0]), np.array([110.0, 90.0, 80.0])), [70.0, 90.0, 100.0])
    # B turned so a point is 100 off A and 100 off B: still exactly the seam, not B-only
    ta, tb = np.array([[100.0]]), np.array([[100.0]])
    w, _ = blend_weight_a(ta, tb, lens, lens, BlendParams(feather_deg=20.0, sky_feather_deg=20.0))
    assert abs(float(w[0, 0]) - 0.5) < 1e-6
    # the old behaviour would have given 0 here (theta_a=100 is past the 90 degree seam)
    w2, _ = blend_weight_a(np.array([[85.0]]), np.array([[100.0]]), lens, lens, bp)
    assert float(w2[0, 0]) == 1.0                     # closer to A's axis than B's: A wins
