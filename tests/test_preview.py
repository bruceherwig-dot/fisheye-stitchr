import numpy as np
import pytest

from spherejoin.blend import BlendParams
from spherejoin.preview import (
    compose_preview, make_proxy, scale_lens, view_for_center, warp_preview_layers,
)
from spherejoin.projection import LensParams, ViewParams, rotation
from spherejoin.synth import make_test_equirect, render_fisheye_from_equirect
from spherejoin.viewer360 import render_perspective

W, H = 1024, 512


@pytest.fixture(scope="module")
def world():
    equi = make_test_equirect(W, H)
    kw = dict(cx=450.0, cy=450.0, radius=420.0, fov_deg=220.0)
    A, B = LensParams(base_yaw=0.0, **kw), LensParams(base_yaw=180.0, **kw)
    fa = render_fisheye_from_equirect(equi, A, (900, 900))
    fb = render_fisheye_from_equirect(equi, B, (900, 900))
    L = warp_preview_layers(fa, fb, A, B, ViewParams(), W, H)
    return equi, A, B, fa, fb, L


def test_blend_mode_rebuilds_panorama(world):
    equi, A, B, fa, fb, L = world
    out = compose_preview(L, A, B, BlendParams())
    assert out.dtype == np.uint8 and out.shape == (H, W, 3)
    assert np.abs(out.astype(int) - equi.astype(int))[8:-8].mean() < 2.5


def test_layer_modes_blank_outside_their_lens(world):
    equi, A, B, fa, fb, L = world
    a = compose_preview(L, A, B, BlendParams(), mode="a_only")
    b = compose_preview(L, A, B, BlendParams(), mode="b_only")
    assert a[~L.valid_a].max() == 0 and b[~L.valid_b].max() == 0
    assert a[L.valid_a].max() > 100 and b[L.valid_b].max() > 100


def test_flicker_changes_only_the_overlap(world):
    equi, A, B, fa, fb, L = world
    p0 = compose_preview(L, A, B, BlendParams(gain_b=0.7), mode="flicker", flicker_phase=0).astype(int)
    p1 = compose_preview(L, A, B, BlendParams(gain_b=0.7), mode="flicker", flicker_phase=1).astype(int)
    ov = L.valid_a & L.valid_b
    diff = np.abs(p0 - p1).sum(axis=2)
    assert diff[~ov].max() == 0
    assert diff[ov].mean() > 5


def test_difference_is_near_zero_when_aligned_and_large_when_not(world):
    equi, A, B, fa, fb, L = world
    ov = L.valid_a & L.valid_b
    good = compose_preview(L, A, B, BlendParams(), mode="difference")
    assert np.all(good[ov][:, 0] == good[ov][:, 1])          # gray inside the overlap
    # Judge the middle of the overlap; the rim of each fisheye is soft.
    mid = ov & (np.abs(L.tha - 90) < 12)
    assert good[mid].mean() < 12
    B_bad = LensParams(cx=450.0, cy=450.0, radius=420.0, base_yaw=180.0, trim_pitch=3.0)
    L2 = warp_preview_layers(fa, fb, A, B_bad, ViewParams(), W, H)
    bad = compose_preview(L2, A, B_bad, BlendParams(), mode="difference")
    mid2 = (L2.valid_a & L2.valid_b) & (np.abs(L2.tha - 90) < 12)
    assert bad[mid2].mean() > 3 * good[mid].mean()


def test_seam_and_overlap_overlays(world):
    equi, A, B, fa, fb, L = world
    plain = compose_preview(L, A, B, BlendParams())
    seam = compose_preview(L, A, B, BlendParams(), show_seam=True)
    changed = np.any(plain != seam, axis=2)
    assert 500 < changed.sum() < 20000
    # The seam sits where theta_A is 90 degrees: near +/-90 degrees of longitude.
    cols = np.where(changed.any(axis=0))[0]
    assert cols.min() < W * 0.30 and cols.max() > W * 0.70
    assert np.all(np.abs(L.tha[changed] - 90) < 25)
    tint = compose_preview(L, A, B, BlendParams(), show_overlap=True)
    ov = L.valid_a & L.valid_b
    same = np.all(plain == tint, axis=2)
    assert same[~ov].all() and (~same[ov]).mean() > 0.9


def test_hard_cut_has_no_feather_lines(world):
    equi, A, B, fa, fb, L = world
    feather = compose_preview(L, A, B, BlendParams(feather_deg=30), show_seam=True)
    hard = compose_preview(L, A, B, BlendParams(hard_cut=True), show_seam=True)
    plain = compose_preview(L, A, B, BlendParams(hard_cut=True))
    assert (np.any(feather != compose_preview(L, A, B, BlendParams(feather_deg=30)), axis=2).sum()
            > np.any(hard != plain, axis=2).sum())


def test_unknown_mode_rejected(world):
    equi, A, B, fa, fb, L = world
    with pytest.raises(ValueError):
        compose_preview(L, A, B, BlendParams(), mode="nope")


def test_proxy_preview_matches_full_size_preview(world):
    equi, A, B, fa, fb, L = world
    pa, sa = make_proxy(fa, max_dim=450)
    pb, sb = make_proxy(fb, max_dim=450)
    assert pa.shape[0] == 450 and abs(sa[0] - 0.5) < 1e-9
    L2 = warp_preview_layers(pa, pb, scale_lens(A, sa), scale_lens(B, sb), ViewParams(), W, H)
    full = compose_preview(L, A, B, BlendParams()).astype(int)
    prox = compose_preview(L2, scale_lens(A, sa), scale_lens(B, sb), BlendParams()).astype(int)
    assert np.abs(full - prox)[8:-8].mean() < 3.0


def test_16bit_sources_preview_on_same_scale(world):
    equi, A, B, fa, fb, L = world
    L16 = warp_preview_layers(fa.astype(np.uint16) * 257, fb.astype(np.uint16) * 257, A, B, ViewParams(), W, H)
    assert abs(float(L16.la.max()) - float(L.la.max())) < 1.0


def test_view_for_center_lands_the_clicked_point_in_the_middle():
    rng = np.random.default_rng(3)
    for _ in range(50):
        v = ViewParams(rng.uniform(-180, 180), rng.uniform(-60, 60), rng.uniform(-45, 45))
        sx, sy = rng.uniform(0, 2048), rng.uniform(100, 924)
        lon, lat = (sx / 2048 - 0.5) * 2 * np.pi, (0.5 - sy / 1024) * np.pi
        d_out = np.array([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)])
        clicked_world = rotation(v.yaw, v.pitch, v.roll) @ d_out
        nv = view_for_center(v, sx, sy, 2048, 1024)
        center_world = rotation(nv.yaw, nv.pitch, nv.roll) @ np.array([0, 0, 1.0])
        assert np.allclose(clicked_world, center_world, atol=1e-9) and nv.roll == v.roll


def test_perspective_viewer_looks_where_told():
    equi = make_test_equirect(W, H)
    def center(yaw, pitch):
        im = render_perspective(equi, yaw, pitch, 60, 101, 61)
        return [int(c) for c in im[30, 50]]
    r, g, b = center(0, 0);   assert r > 200 and g < 60 and b < 60       # front blob is red
    r, g, b = center(90, 0);  assert g > 200 and r < 60 and b < 60       # right blob is green
    r, g, b = center(180, 0); assert b > 200 and r < 60 and g < 60       # back blob is blue
    r, g, b = center(0, 60);  assert r > 200 and g > 200 and b < 80      # up blob is yellow
    assert render_perspective(equi, 10, 20, 90, 320, 200).shape == (200, 320, 3)
