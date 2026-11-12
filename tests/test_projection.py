import numpy as np
import pytest

from spherejoin.projection import (
    MODELS, LensParams, ViewParams, direction_to_source, equirect_directions, lens_rotation,
    radius_from_theta, rotation, source_to_direction, theta_from_radius, build_maps, render_layer,
)
from spherejoin.synth import make_test_equirect, render_fisheye_from_equirect


def lens(model="equidistant", **kw):
    base = dict(cx=1000.0, cy=900.0, radius=800.0, fov_deg=220.0, model=model)
    if model == "polynomial":
        base["poly"] = (1.0, 0.08, -0.03)
    base.update(kw)
    return LensParams(**base)


@pytest.mark.parametrize("model", MODELS)
def test_radius_theta_roundtrip_and_edge(model):
    L = lens(model)
    th = np.linspace(0, L.theta_max, 200)
    r = radius_from_theta(th, L)
    assert abs(r[-1] - L.radius) < 1e-6          # rim lands exactly on the circle
    assert np.all(np.diff(r) > 0)                 # monotonic
    assert np.allclose(theta_from_radius(r, L), th, atol=1e-6)


@pytest.mark.parametrize("model", MODELS)
def test_pixel_vector_pixel_roundtrip(model):
    L = lens(model, trim_yaw=0.3)
    rng = np.random.default_rng(1)
    ang = rng.uniform(0, 2 * np.pi, 5000)
    rad = rng.uniform(0, 0.999 * L.radius, 5000)
    x = L.cx + rad * np.cos(ang)
    y = L.cy + rad * np.sin(ang)
    d = source_to_direction(x, y, L)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1.0)
    x2, y2, _ = direction_to_source(d, L)
    assert np.allclose(x, x2, atol=1e-6) and np.allclose(y, y2, atol=1e-6)


@pytest.mark.parametrize("model", MODELS)
def test_vector_pixel_vector_roundtrip(model):
    L = lens(model)
    rng = np.random.default_rng(2)
    v = rng.normal(size=(3000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    x, y, theta = direction_to_source(v, L)
    ok = np.isfinite(x)
    assert ok.sum() > 1000
    assert np.all(theta[ok] <= L.theta_max + 1e-12) and np.all(theta[~ok] > L.theta_max)
    v2 = source_to_direction(x[ok], y[ok], L)
    assert np.allclose(v[ok], v2, atol=1e-6)


def test_known_equidistant_points():
    L = lens()
    # Straight ahead lands on the circle center.
    x, y, _ = direction_to_source(np.array([0.0, 0.0, 1.0]), L)
    assert abs(x - L.cx) < 1e-9 and abs(y - L.cy) < 1e-9
    # 90 degrees up is above center by R * 90/110 (image up = smaller row).
    x, y, _ = direction_to_source(np.array([0.0, 1.0, 0.0]), L)
    assert abs(x - L.cx) < 1e-9 and abs(y - (L.cy - 800.0 * 90 / 110)) < 1e-6
    # 90 degrees right is right of center.
    x, y, _ = direction_to_source(np.array([1.0, 0.0, 0.0]), L)
    assert abs(y - L.cy) < 1e-9 and abs(x - (L.cx + 800.0 * 90 / 110)) < 1e-6
    # Directly behind is outside the field of view.
    x, _, _ = direction_to_source(np.array([0.0, 0.0, -1.0]), L)
    assert np.isnan(x)


def test_rotation_is_orthonormal_and_points_where_told():
    R = rotation(30, 20, 15)
    assert np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-12
    z = rotation(30, 20, 99) @ np.array([0, 0, 1.0])  # roll must not move the axis
    yaw, pitch = np.radians([30, 20])
    assert np.allclose(z, [np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])


def test_back_lens_faces_backward_with_up_kept():
    B = lens(base_yaw=180.0)
    R = lens_rotation(B)
    assert np.allclose(R @ [0, 0, 1], [0, 0, -1], atol=1e-12)
    assert np.allclose(R @ [0, 1, 0], [0, 1, 0], atol=1e-12)
    assert np.allclose(R @ [1, 0, 0], [-1, 0, 0], atol=1e-12)


def test_panorama_center_follows_view_yaw():
    A, B = lens(), lens(base_yaw=180.0)
    cx_out = 512
    for view, expect_lens in ((ViewParams(0, 0, 0), A), (ViewParams(180, 0, 0), B)):
        mx, my, th = build_maps(1024, 512, expect_lens, view, 255, 257)
        assert abs(mx[0, cx_out] - expect_lens.cx) < 2 and abs(my[0, cx_out] - expect_lens.cy) < 2
        assert th[0, cx_out] < 1.0
    # Lens B does not see the panorama center when the view is not turned.
    mx, _, _ = build_maps(1024, 512, B, ViewParams(), 255, 257)
    assert mx[0, cx_out] == -1


def test_two_lenses_cover_the_whole_sphere():
    A, B = lens(), lens(base_yaw=180.0)
    _, _, tha = build_maps(720, 360, A, ViewParams())
    _, _, thb = build_maps(720, 360, B, ViewParams())
    assert np.all((tha <= 110) | (thb <= 110))
    # Overlap band exists (220 degrees > 180) and 90 degrees is the natural seam.
    assert np.any((tha <= 110) & (thb <= 110))
    assert np.allclose(tha + thb, 180.0, atol=1e-3)


def test_tiled_render_matches_single_pass(tmp_path):
    equi = make_test_equirect(512, 256)
    L = lens(cx=300.0, cy=300.0, radius=280.0)
    src = render_fisheye_from_equirect(equi, L, (600, 600))
    a, _ = render_layer(src, L, ViewParams(), 400, 200, tile_rows=33)
    b, _ = render_layer(src, L, ViewParams(), 400, 200, tile_rows=200)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("model", MODELS)
def test_synthetic_grid_rebuilt_without_seam_gap(model):
    equi = make_test_equirect(2048, 1024)
    kw = dict(cx=800.0, cy=800.0, radius=760.0, model=model)
    if model == "polynomial":
        kw["poly"] = (1.0, 0.08, -0.03)
    A = LensParams(fov_deg=220.0, base_yaw=0.0, **kw)
    B = LensParams(fov_deg=220.0, base_yaw=180.0, **kw)
    fa = render_fisheye_from_equirect(equi, A, (1600, 1600))
    fb = render_fisheye_from_equirect(equi, B, (1600, 1600))
    la, tha = render_layer(fa, A, ViewParams(), 2048, 1024)
    lb, thb = render_layer(fb, B, ViewParams(), 2048, 1024)
    assert np.all((tha <= 110) | (thb <= 110)), "gap on the sphere"
    out = np.where((tha <= 90)[..., None], la, lb)
    err = np.abs(out.astype(np.int16) - equi.astype(np.int16))
    # Ignore a 2 degree band next to each pole, where the fisheye is stretched most.
    rows = slice(int(1024 * 2 / 180), int(1024 * (1 - 2 / 180)))
    mean_err = err[rows].mean()
    p99 = np.percentile(err[rows], 99)
    print(model, "mean abs err", round(float(mean_err), 3), "p99", float(p99))
    assert mean_err < 2.5
    assert p99 < 30
