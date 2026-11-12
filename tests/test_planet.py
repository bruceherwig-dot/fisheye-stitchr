"""Tiny planet layout: round trip, square size, center, and the control wiring."""
import numpy as np

from spherejoin.reproject import Projection, lonlat_to_pixel, output_directions, output_size, pixel_to_lonlat


def test_square_and_center():
    p = Projection.default("planet")
    assert output_size(p, 800) == (800, 800)
    lon, lat = pixel_to_lonlat(p, 400.0, 400.0, 800, 800)
    assert abs(float(lon)) < 1e-9 and abs(float(lat)) < 1e-9


def test_round_trip():
    p = Projection("planet", 150.0, 0.0)
    px = np.array([100.5, 400.5, 700.5, 250.5, 33.5])
    py = np.array([100.5, 400.5, 250.5, 690.5, 780.5])
    lon, lat = pixel_to_lonlat(p, px, py, 800, 800)
    bx, by = lonlat_to_pixel(p, lon, lat, 800, 800)
    assert np.allclose(bx, px, atol=1e-6) and np.allclose(by, py, atol=1e-6)


def test_edge_reaches_the_set_angle():
    p = Projection("planet", 120.0, 0.0)
    d = output_directions(p, 800, 800, 400, 401)[0]          # row through the middle
    right_edge = d[-1]
    assert abs(np.degrees(np.arccos(right_edge[2])) - 120.0) < 0.5


def test_controls_switch_and_restore(app):
    from spherejoin.controls import ControlsPanel
    c = ControlsPanel()
    c.set_view(10.0, 5.0, 0.0)
    c.proj_kind.setCurrentIndex(c.proj_kind.findData("planet"))
    assert c.projection().is_planet and c.view().pitch == -90.0
    assert c.lat_top.maximum() >= 170
    state = c.state_dict()
    c.proj_kind.setCurrentIndex(c.proj_kind.findData("mercator"))
    assert c.view().pitch == 5.0 and c.lat_top.maximum() < 90
    c.load_state(state)
    assert c.projection().is_planet and abs(c.projection().lat_top_deg - 150.0) < 1e-9


def test_planet_tunnel_slider_follows_pitch(app):
    from spherejoin.controls import ControlsPanel
    c = ControlsPanel()
    c.proj_kind.setCurrentIndex(c.proj_kind.findData("planet"))
    assert c.planet_tilt.value() == -90.0 and not c.planet_tilt.isHidden()
    c.planet_tilt.setValue(90.0)
    assert c.view().pitch == 90.0                       # slider drives the view
    c.set_view(0.0, 20.0)
    assert c.planet_tilt.value() == 20.0                # and the view drives the slider
    c.proj_kind.setCurrentIndex(c.proj_kind.findData("mercator"))
    assert c.planet_tilt.isHidden()
