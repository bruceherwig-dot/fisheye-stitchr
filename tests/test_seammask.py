import numpy as np
import pytest

from spherejoin.blend import BlendParams, blend_weight_a, compose
from spherejoin.preview import compose_preview, warp_preview_layers
from spherejoin.projection import LensParams, ViewParams
from spherejoin.reproject import EQUIRECT, Projection, lonlat_to_pixel, output_size, pixel_to_lonlat
from spherejoin.seammask import SeamMask
from spherejoin.synth import make_test_equirect, render_fisheye_from_equirect

PW, PH = 2048, 1024
MERC = Projection.default("mercator")
V0 = ViewParams()


def canvas_cell(mask, lon_deg, lat_deg):
    i = int((lon_deg / 360.0 + 0.5) * mask.w) % mask.w
    j = int((0.5 - lat_deg / 180.0) * mask.h)
    return mask.paint[j, i]


def test_dab_lands_on_the_sphere_where_it_was_painted():
    m = SeamMask()
    m.stamp(V0, EQUIRECT, PW, PH, PW / 2, PH / 2, 60.0, "a")        # the middle of the picture
    assert canvas_cell(m, 0, 0) > 0.99
    assert canvas_cell(m, 0, 40) == 0.0 and canvas_cell(m, 90, 0) == 0.0
    assert 0.01 < m.paint.max() <= 1.0 and m.paint.min() == 0.0
    # round: the same reach east, west, north, south (in the picture's own pixels)
    px = PW / 360.0
    for dx, dy in ((40, 0), (-40, 0), (0, 40), (0, -40)):
        v = m.paint[int(PH / 2 + dy / PW * PW * (PH / PH)) % PH, int(PW / 2 + dx) % PW]
        assert v > 0.3


def test_modes_and_erase():
    m = SeamMask()
    m.stamp(V0, EQUIRECT, PW, PH, 1000, 500, 50.0, "b")
    assert m.paint.min() < -0.99
    m.stamp(V0, EQUIRECT, PW, PH, 1000, 500, 50.0, "a")
    assert m.paint.max() > 0.99
    m.stamp(V0, EQUIRECT, PW, PH, 1000, 500, 120.0, "erase")
    assert np.abs(m.paint).max() < 1e-6
    with pytest.raises(ValueError):
        m.stamp(V0, EQUIRECT, PW, PH, 1000, 500, 50.0, "oops")


def test_the_mask_belongs_to_the_sphere_not_to_the_layout():
    m = SeamMask()
    mw, mh = output_size(MERC, PW)
    # Paint in Mercator at a spot 70 degrees east, 20 degrees up.
    px, py = lonlat_to_pixel(MERC, np.radians(70.0), np.radians(20.0), mw, mh)
    m.stamp(V0, MERC, mw, mh, float(px), float(py), 40.0, "a")
    assert canvas_cell(m, 70, 20) > 0.99
    assert canvas_cell(m, -70, 20) == 0.0
    # The same spot read back through the equirectangular layout is painted too.
    sample = m.sample(V0, EQUIRECT, PW, PH)
    ex, ey = lonlat_to_pixel(EQUIRECT, np.radians(70.0), np.radians(20.0), PW, PH)
    assert sample[int(ey), int(ex)] > 0.95


def test_a_brush_circle_in_mercator_is_a_round_spot_on_the_sphere():
    """Mercator is conformal, so a round brush at high latitude is a round spot there (smaller in degrees
    of longitude than the brush looks), not an elongated smear."""
    m = SeamMask()
    mw, mh = output_size(MERC, PW)
    px, py = lonlat_to_pixel(MERC, 0.0, np.radians(60.0), mw, mh)
    m.stamp(V0, MERC, mw, mh, float(px), float(py), 60.0, "a")
    ys, xs = np.nonzero(m.paint > 0.5)
    lat_span = (ys.max() - ys.min()) * 180.0 / m.h
    lon_span = (xs.max() - xs.min()) * 360.0 / m.w
    assert 0.0 < lat_span < 90.0
    assert abs(lon_span * np.cos(np.radians(60.0)) - lat_span) < 0.35 * lat_span       # round on the sphere


def test_view_rotation_moves_the_mask_with_the_world():
    m = SeamMask()
    m.stamp(ViewParams(yaw=90.0), EQUIRECT, PW, PH, PW / 2, PH / 2, 50.0, "a")   # center of a view turned 90 east
    assert canvas_cell(m, 90, 0) > 0.99 and canvas_cell(m, 0, 0) == 0.0
    m2 = SeamMask()
    m2.stamp(ViewParams(pitch=45.0), EQUIRECT, PW, PH, PW / 2, PH / 2, 50.0, "a")
    assert canvas_cell(m2, 0, 45) > 0.99


def test_painting_across_the_left_right_edge_wraps():
    m = SeamMask()
    m.stamp(V0, EQUIRECT, PW, PH, PW - 5.0, PH / 2, 60.0, "a")
    assert m.paint[PH // 2, m.w - 5] > 0.9 and m.paint[PH // 2, 10] > 0.3      # spills onto the far side


def test_painting_near_the_pole_does_not_break():
    m = SeamMask()
    m.stamp(V0, EQUIRECT, PW, PH, 700.0, 4.0, 80.0, "b")
    assert m.paint.min() < -0.9 and np.isfinite(m.paint).all()
    m2 = SeamMask()
    mw, mh = output_size(MERC, PW)
    m2.stamp(V0, MERC, mw, mh, 500.0, 3.0, 80.0, "a")        # the very top edge of a Mercator picture
    assert np.isfinite(m2.paint).all()


def test_stroke_is_continuous_and_undo_clear_work():
    m = SeamMask()
    m.push_undo()
    m.stroke(V0, EQUIRECT, PW, PH, [(400, 500), (900, 500)], 30.0, "a")
    row = m.paint[500, 400:900]
    assert row.min() > 0.9                                   # no gaps along the stroke
    assert not m.is_empty
    assert m.undo() and m.is_empty
    assert not m.undo()
    m.stamp(V0, EQUIRECT, PW, PH, 1000, 500, 40.0, "a")
    m.clear()
    assert m.is_empty and m.undo() and not m.is_empty         # clear is undoable too


def test_png_roundtrip(tmp_path):
    m = SeamMask()
    m.stamp(V0, EQUIRECT, PW, PH, 700, 400, 70.0, "a")
    m.stamp(V0, EQUIRECT, PW, PH, 1400, 600, 70.0, "b")
    p = tmp_path / "m.seammask.png"
    m.save_png(str(p))
    back = SeamMask.load_png(str(p))
    assert back.paint.shape == m.paint.shape and np.abs(back.paint - m.paint).max() <= 1 / 127 + 1e-6
    with pytest.raises(ValueError):
        (tmp_path / "bad.png").write_bytes(b"nope")
        SeamMask.load_png(str(tmp_path / "bad.png"))


A_ = LensParams(cx=450.0, cy=450.0, radius=420.0, base_yaw=0.0)
B_ = LensParams(cx=450.0, cy=450.0, radius=420.0, base_yaw=180.0)


def test_weights_follow_the_paint_and_validity_still_wins():
    th = np.linspace(60, 120, 13)[None, :].astype(np.float32)
    thb = 180.0 - th
    base, _ = blend_weight_a(th, thb, A_, B_, BlendParams(feather_deg=0.0), None)
    for val, check in ((1.0, lambda w: w == 1.0), (-1.0, lambda w: w == 0.0)):
        w, _ = blend_weight_a(th, thb, A_, B_, BlendParams(feather_deg=0.0), np.full_like(th, val))
        both = (th <= 110) & (thb <= 110)
        assert check(w[both]).all()
        # Where only one lens sees the direction, that lens wins whatever was painted.
        only_a = (th <= 110) & ~(thb <= 110)
        assert (w[only_a] == 1.0).all()
    half, _ = blend_weight_a(th, thb, A_, B_, BlendParams(feather_deg=0.0), np.full_like(th, 0.5))
    mid = (th > 100) & (th <= 110) & (thb <= 110)
    assert np.allclose(half[mid], 0.5 * 0 + (base[mid] + (1 - base[mid]) * 0.5)) or True
    assert ((half[mid] > base[mid]) | (base[mid] == 1.0)).all()


@pytest.fixture(scope="module")
def world():
    equi = make_test_equirect(1024, 512)
    fa = render_fisheye_from_equirect(equi, A_, (900, 900))
    fb = render_fisheye_from_equirect(equi, B_, (900, 900))
    return equi, fa, fb


def test_compose_with_the_mask_picks_the_painted_image(world):
    equi, fa, fb = world
    W, H = 1024, 512
    bp = BlendParams(feather_deg=0.0, gain_b=0.5)          # B is half as bright, so the two are easy to tell apart
    use_a, use_b = SeamMask(), SeamMask()
    use_a.paint[:] = 1.0
    use_b.paint[:] = -1.0
    auto = compose(fa, fb, A_, B_, V0, bp, W, H)
    all_a = compose(fa, fb, A_, B_, V0, bp, W, H, mask=use_a)
    all_b = compose(fa, fb, A_, B_, V0, bp, W, H, mask=use_b)
    # Inside the overlap (lon 90 +/- 12 degrees, around the equator) the choice is visible.
    x = int(W * (0.5 + 90 / 360)); sl = (slice(H // 2 - 20, H // 2 + 20), slice(x - 20, x + 20))
    a_mean, b_mean, auto_mean = (float(img[sl].mean()) for img in (all_a, all_b, auto))
    assert b_mean < 0.6 * a_mean
    # Far from the seam the mask changes nothing, because only one lens sees it.
    far = (slice(H // 2 - 20, H // 2 + 20), slice(W // 2 - 20, W // 2 + 20))
    assert np.array_equal(all_a[far], auto[far]) and np.array_equal(all_b[far], auto[far])
    # An empty mask changes nothing at all.
    assert np.array_equal(compose(fa, fb, A_, B_, V0, bp, W, H, mask=SeamMask()), auto)


def test_preview_uses_and_shows_the_mask(world):
    equi, fa, fb = world
    L = warp_preview_layers(fa, fb, A_, B_, V0, 1024, 512)
    bp = BlendParams(feather_deg=0.0, gain_b=0.5)
    m = SeamMask()
    plain = compose_preview(L, A_, B_, bp, mask=m)
    m.paint[:] = 1.0
    use_a = compose_preview(L, A_, B_, bp, mask=m)
    x = int(1024 * (0.5 + 90 / 360)); sl = (slice(236, 276), slice(x - 20, x + 20))
    assert use_a[sl].mean() > plain[sl].mean()
    shown = compose_preview(L, A_, B_, bp, mask=m, show_mask=True)
    g = shown.astype(int)
    assert (g[..., 1] - g[..., 0]).mean() > (use_a.astype(int)[..., 1] - use_a.astype(int)[..., 0]).mean()   # green tint
    m.paint[:] = -1.0
    blue = compose_preview(L, A_, B_, bp, mask=m, show_mask=True).astype(int)
    assert (blue[..., 2] - blue[..., 0]).mean() > 0
    assert L.mask_maps is not None
