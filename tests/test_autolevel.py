"""Line up, level and face the pole: checked on synthetic fisheye pictures with a known camera tilt and pole."""
import numpy as np

from spherejoin.autolevel import auto_level, euler_trim
from spherejoin.projection import LensParams, rotation, source_to_direction

SIZE = 900


def synth(base_yaw, trim, pole_world, fov=220.0):
    """A fisheye picture of a world made of a level horizon (bright sky, dark ground) and concentric star-trail rings
    around pole_world. trim = (yaw, pitch, roll) is how the camera really sat."""
    lens = LensParams(cx=SIZE / 2, cy=SIZE / 2, radius=SIZE * 0.46, fov_deg=fov, base_yaw=base_yaw)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    d_lens = source_to_direction(xx, yy, lens)
    d_world = d_lens @ (rotation(base_yaw, 0, 0) @ rotation(*trim)).T
    inside = np.hypot(xx - lens.cx, yy - lens.cy) < lens.radius
    sky = d_world[..., 1] > 0.0
    ang = np.arccos(np.clip(d_world @ pole_world, -1, 1))
    lon = np.arctan2(d_world[..., 0], d_world[..., 2])
    lat = np.arcsin(np.clip(d_world[..., 1], -1, 1))
    ground = 45.0 + 25.0 * np.sin(11.0 * lon) * np.sin(9.0 * lat) + 15.0 * np.sin(23.0 * lon + 5.0 * lat)  # features
    img = np.where(sky, 120.0 + 25.0 * np.sin(ang * 70.0), ground)
    img = np.where(inside, img, 200.0)
    return np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, axis=2), lens


def pole_dir(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.array([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)])


def test_euler_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        t = (rng.uniform(-170, 170), rng.uniform(-60, 60), rng.uniform(-170, 170))
        assert np.abs(rotation(*t) - rotation(*euler_trim(rotation(*t)))).max() < 1e-9


def test_lines_up_levels_and_faces_the_pole():
    pole = pole_dir(-20.0, -33.0)                        # the pole below the horizon, 20 degrees left of lens A's axis
    img_a, la = synth(0.0, (0.0, 5.0, -2.0), pole)
    img_b, lb = synth(180.0, (-15.0, 8.0, 3.0), pole)
    res = auto_level(img_a, la, img_b, lb)
    assert res.trim_a is not None, res.notes
    # both lenses leveled: pitch and roll match how the cameras really sat
    assert abs(res.trim_a[1] - 5.0) < 1.0 and abs(res.trim_a[2] + 2.0) < 1.0
    assert abs(res.trim_b[1] - 8.0) < 1.0 and abs(res.trim_b[2] - 3.0) < 1.0
    assert abs(res.trim_b[0] - res.trim_a[0] + 15.0) < 1.5          # B turned 15 degrees relative to A
    assert abs((res.view_yaw - res.trim_a[0]) + 20.0) < 1.5         # the pole, measured from lens A's own heading
    assert abs(res.fov_deg - 220.0) < 8.0


def test_finds_the_true_field_of_view():
    pole = pole_dir(-20.0, -33.0)
    img_a, la = synth(0.0, (0.0, 0.0, 0.0), pole, fov=230.0)
    img_b, lb = synth(180.0, (-12.0, 4.0, 2.0), pole, fov=230.0)
    res = auto_level(img_a, replace_fov(la, 220.0), img_b, replace_fov(lb, 220.0))
    # the field of view is only loosely pinned down (the horizon-step term pulls it a little): within 8 degrees
    assert res.fov_deg is not None and abs(res.fov_deg - 230.0) < 8.0, res.fov_deg


def replace_fov(lens, fov):
    from dataclasses import replace
    return replace(lens, fov_deg=fov)


def test_no_trails_changes_nothing():
    flat = np.full((SIZE, SIZE, 3), 90, np.uint8)
    lens = LensParams(cx=SIZE / 2, cy=SIZE / 2, radius=SIZE * 0.46)
    lens_b = LensParams(cx=SIZE / 2, cy=SIZE / 2, radius=SIZE * 0.46, base_yaw=180.0)
    res = auto_level(flat, lens, flat, lens_b)
    assert res.trim_a is None and res.trim_b is None and res.fov_deg is None and res.view_yaw is None
    assert res.notes


def test_button_sets_fov_trims_and_view(app):
    from spherejoin.workspace import Workspace
    ws = Workspace()
    pole = pole_dir(-20.0, -33.0)
    img_a, la = synth(0.0, (0.0, 5.0, -2.0), pole)
    img_b, lb = synth(180.0, (-15.0, 8.0, 3.0), pole)
    ws.pane_a.load_array(img_a, "a.png", "a.png", (la.cx, la.cy, la.radius))
    ws.pane_b.load_array(img_b, "b.png", "b.png", (lb.cx, lb.cy, lb.radius))
    ws.controls.level_btn.click()
    assert abs(ws.controls.trim_b[1].value() - 8.0) < 1.0
    assert abs(ws.controls.trim_b[0].value() - ws.controls.trim_a[0].value() + 15.0) < 1.5
    assert abs((ws.controls.view().yaw - ws.controls.trim_a[0].value()) + 20.0) < 1.5
