import numpy as np
import pytest

from spherejoin.blend import BlendParams, compose
from spherejoin.export import ExportSettings, default_output_path, export_panorama
from spherejoin.preview import view_for_center
from spherejoin.projection import LensParams, ViewParams, equirect_directions, rotation
from spherejoin.reproject import (
    EQUIRECT, KINDS, MAX_LAT, Projection, lonlat_to_pixel, output_directions, output_size, pixel_to_lonlat,
)
from spherejoin.synth import make_test_equirect, render_fisheye_from_equirect

MERC, GALL = Projection.default("mercator"), Projection.default("gall")


def test_equirect_layout_is_identical_to_the_original_grid():
    for w, h in ((64, 32), (200, 100)):
        assert np.allclose(output_directions(EQUIRECT, w, h), equirect_directions(w, h), atol=1e-12)
        assert output_size(EQUIRECT, w) == (w, h)


@pytest.mark.parametrize("kind", KINDS)
def test_pixel_lonlat_roundtrip(kind):
    p = Projection.default(kind)
    w, h = output_size(p, 1200)
    rng = np.random.default_rng(0)
    px, py = rng.uniform(0, w, 500), rng.uniform(0, h, 500)
    lon, lat = pixel_to_lonlat(p, px, py, w, h)
    x2, y2 = lonlat_to_pixel(p, lon, lat, w, h)
    assert np.allclose(px, x2, atol=1e-6) and np.allclose(py, y2, atol=1e-6)
    top, bot = np.radians(p.lat_top_deg), np.radians(p.lat_bot_deg)
    if not p.is_planet:                      # the planet has no latitude range
        assert lat.max() <= top + 1e-9 and lat.min() >= bot - 1e-9


def test_sizes_match_what_we_measured_in_the_experiment():
    assert output_size(MERC, 2000) == (2000, 1065)
    assert output_size(GALL, 2000) == (2000, 1242)


def test_horizon_is_the_straight_line_at_zero_latitude():
    for p in (MERC, GALL):
        w, h = output_size(p, 1000)
        _, lat = pixel_to_lonlat(p, np.array([0.0, 500.0, 999.0]), np.full(3, 0.0), w, h)
        # Row of latitude 0: find it, then check every column on that row is level.
        _, py = lonlat_to_pixel(p, 0.0, 0.0, w, h)
        lon, lat = pixel_to_lonlat(p, np.linspace(0, w, 7), np.full(7, py), w, h)
        assert np.allclose(lat, 0.0, atol=1e-9)


def test_mercator_is_conformal_small_circles_stay_round():
    """At any latitude a small step east and a same-size step north on the sphere cover the same number of
    pixels in Mercator, so circles (star trails around the pole) stay round. Equirectangular does not."""
    def aspect(p, lat_deg):
        w, h = output_size(p, 4000)
        lat = np.radians(lat_deg)
        d = 1e-4
        x0, y0 = lonlat_to_pixel(p, 0.0, lat, w, h)
        xe, _ = lonlat_to_pixel(p, d / np.cos(lat), lat, w, h)     # d radians of arc to the east
        _, yn = lonlat_to_pixel(p, 0.0, lat + d, w, h)             # d radians of arc to the north
        return abs(xe - x0) / abs(yn - y0)                          # 1.0 means undistorted
    for lat in (0, 30, 50, 70):
        assert abs(aspect(MERC, lat) - 1.0) < 1e-3
    assert abs(aspect(EQUIRECT, 50) - 1.0) > 0.4       # the ovals we saw
    assert abs(aspect(GALL, 50) - 1.0) < abs(aspect(EQUIRECT, 50) - 1.0)


def test_bad_latitude_ranges_are_rejected():
    for bad in (Projection("mercator", 90.0, -60.0), Projection("mercator", 40.0, 50.0),
                Projection("gall", 80.0, -MAX_LAT - 1), Projection("equirectangular", 100.0, -90.0)):
        with pytest.raises(ValueError):
            output_size(bad, 1000)


KW = dict(cx=450.0, cy=450.0, radius=420.0, fov_deg=220.0)


@pytest.fixture(scope="module")
def world():
    equi = make_test_equirect(2048, 1024)
    A, B = LensParams(base_yaw=0.0, **KW), LensParams(base_yaw=180.0, **KW)
    return (equi, A, B, render_fisheye_from_equirect(equi, A, (900, 900)),
            render_fisheye_from_equirect(equi, B, (900, 900)))


@pytest.mark.parametrize("p", [MERC, GALL])
def test_projected_stitch_matches_resampling_the_true_sphere(world, p):
    """Compose straight from the fisheyes in the new layout, and compare with sampling the original
    equirectangular test pattern at the same longitude and latitude."""
    equi, A, B, fa, fb = world
    w, h = output_size(p, 1000)
    out = compose(fa, fb, A, B, ViewParams(), BlendParams(), w, proj=p)
    assert out.shape == (h, w, 3)
    import cv2
    px, py = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    lon, lat = pixel_to_lonlat(p, px, py, w, h)
    ex = ((lon / (2 * np.pi) + 0.5) * 2048 - 0.5).astype(np.float32)
    ey = ((0.5 - lat / np.pi) * 1024 - 0.5).astype(np.float32)
    expect = cv2.remap(equi, ex, ey, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    err = np.abs(out.astype(int) - expect.astype(int))
    print(p.kind, "mean abs err", round(float(err.mean()), 2))
    assert err.mean() < 3.0


def test_view_for_center_works_in_every_layout():
    rng = np.random.default_rng(5)
    for p in (EQUIRECT, MERC, GALL):
        w, h = output_size(p, 2048)
        for _ in range(30):
            v = ViewParams(rng.uniform(-180, 180), rng.uniform(-50, 50), rng.uniform(-30, 30))
            sx, sy = rng.uniform(0, w), rng.uniform(0.15 * h, 0.85 * h)
            lon, lat = pixel_to_lonlat(p, sx, sy, w, h)
            d_out = np.array([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)])
            clicked = rotation(v.yaw, v.pitch, v.roll) @ d_out
            nv = view_for_center(v, sx, sy, w, h, p)
            assert np.allclose(clicked, rotation(nv.yaw, nv.pitch, nv.roll) @ np.array([0, 0, 1.0]), atol=1e-9)


def test_export_in_mercator_has_the_right_size_name_and_metadata_note(world, tmp_path):
    equi, A, B, fa, fb = world
    out = tmp_path / "m.tif"
    res = export_panorama(str(out), fa, fb, A, B, ViewParams(), BlendParams(),
                          ExportSettings(width=1000, fmt="tiff", save_layers=True, spherical_metadata=True), proj=MERC)
    import tifffile
    assert tifffile.imread(res.paths[0]).shape[:2] == (output_size(MERC, 1000)[1], 1000)
    assert tifffile.imread(res.paths[1]).shape[:2] == (output_size(MERC, 1000)[1], 1000)
    assert any("equirectangular" in m for m in res.messages)
    assert default_output_path("/x/a.jpg", "png", "mercator").endswith("a_spherejoin_mercator.png")
    assert default_output_path("/x/a.jpg", "png").endswith("a_spherejoin_equirect.png")
