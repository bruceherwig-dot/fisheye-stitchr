"""Fisheye projection math and equirectangular remapping.

World frame:  x = right, y = up, z = forward.  Lens frame: z = optical axis,
x = image right, y = image UP (so image row decreases as y grows).

Equirectangular output: longitude 0 is the center column, latitude +90 is the top row.
Lens A faces world +z (forward). Lens B faces world -z (a 180 degree yaw), image up stays up.

Pixel convention is cv2.remap's: the CENTER of pixel (i, j) is at x=i, y=j.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MODELS = ("equidistant", "equisolid", "stereographic", "polynomial")


@dataclass
class LensParams:
    cx: float
    cy: float
    radius: float                      # circle radius in source pixels (where theta = half FOV)
    fov_deg: float = 220.0             # full field of view across the circle
    model: str = "equidistant"
    poly: tuple = (1.0, 0.0, 0.0)      # polynomial model: coefficients of u, u^3, u^5
    trim_yaw: float = 0.0              # small alignment trims, degrees
    trim_pitch: float = 0.0
    trim_roll: float = 0.0
    base_yaw: float = 0.0              # 0 for the front lens, 180 for the back lens

    @property
    def theta_max(self) -> float:
        return np.radians(self.fov_deg) / 2.0


@dataclass
class ViewParams:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


# ---------------------------------------------------------------- rotations
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Rotation that sends the +z axis to the direction (yaw right, pitch up), then rolls about it.

    Column vectors: d_world = R @ d_local.
    """
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    return rot_y(y) @ rot_x(-p) @ rot_z(r)


def lens_rotation(lens: LensParams) -> np.ndarray:
    """Lens frame -> world frame."""
    base = rotation(lens.base_yaw, 0.0, 0.0)
    trim = rotation(lens.trim_yaw, lens.trim_pitch, lens.trim_roll)
    return base @ trim


# ---------------------------------------------------------------- lens models
def _poly_g(u, poly):
    k1, k2, k3 = poly
    return k1 * u + k2 * u ** 3 + k3 * u ** 5


def radius_from_theta(theta, lens: LensParams):
    """Source-pixel distance from the circle center for an angle theta off the lens axis."""
    R, tm = lens.radius, lens.theta_max
    m = lens.model
    if m == "equidistant":
        return R * theta / tm
    if m == "equisolid":
        return R * np.sin(theta / 2.0) / np.sin(tm / 2.0)
    if m == "stereographic":
        return R * np.tan(theta / 2.0) / np.tan(tm / 2.0)
    if m == "polynomial":
        return R * _poly_g(theta / tm, lens.poly) / _poly_g(1.0, lens.poly)
    raise ValueError(f"Unknown lens model: {m}")


def theta_from_radius(r, lens: LensParams):
    """Inverse of radius_from_theta. Inputs beyond the circle radius give theta beyond theta_max."""
    R, tm = lens.radius, lens.theta_max
    m = lens.model
    if m == "equidistant":
        return r / R * tm
    if m == "equisolid":
        return 2.0 * np.arcsin(np.clip(r / R * np.sin(tm / 2.0), -1.0, 1.0))
    if m == "stereographic":
        return 2.0 * np.arctan(r / R * np.tan(tm / 2.0))
    if m == "polynomial":
        # Numerical inverse by table lookup; the curve must rise steadily with u.
        u = np.linspace(0.0, 1.5, 3001)
        rr = R * _poly_g(u, lens.poly) / _poly_g(1.0, lens.poly)
        if not np.all(np.diff(rr) > 0):
            raise ValueError("Polynomial lens curve is not strictly increasing")
        return np.interp(r, rr, u * tm)
    raise ValueError(f"Unknown lens model: {m}")


# ---------------------------------------------------------------- pixel <-> direction in lens frame
def source_to_direction(x, y, lens: LensParams):
    """Source pixel -> unit direction in the lens frame. Returns (..., 3)."""
    dx = np.asarray(x, dtype=np.float64) - lens.cx
    dy = lens.cy - np.asarray(y, dtype=np.float64)  # image up is +y in the lens frame
    r = np.hypot(dx, dy)
    theta = theta_from_radius(r, lens)
    phi = np.arctan2(dy, dx)
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1)


def direction_to_source(d, lens: LensParams):
    """Unit direction in the lens frame -> (x, y, theta). x, y are NaN beyond the lens FOV."""
    d = np.asarray(d, dtype=np.float64)
    theta = np.arctan2(np.hypot(d[..., 0], d[..., 1]), d[..., 2])
    phi = np.arctan2(d[..., 1], d[..., 0])
    valid = theta <= lens.theta_max
    th = np.where(valid, theta, 0.0)
    r = radius_from_theta(th, lens)
    x = lens.cx + r * np.cos(phi)
    y = lens.cy - r * np.sin(phi)
    x = np.where(valid, x, np.nan)
    y = np.where(valid, y, np.nan)
    return x, y, theta


# ---------------------------------------------------------------- equirectangular grid
def equirect_directions(out_w: int, out_h: int, y0: int = 0, y1: int | None = None) -> np.ndarray:
    """Unit directions (world/panorama frame) for output rows y0..y1. Shape (rows, out_w, 3)."""
    if y1 is None:
        y1 = out_h
    lon = ((np.arange(out_w) + 0.5) / out_w - 0.5) * 2.0 * np.pi
    lat = (0.5 - (np.arange(y0, y1) + 0.5) / out_h) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    cl = np.cos(lat)
    return np.stack([cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)], axis=-1)


def direction_to_equirect(d, out_w: int, out_h: int):
    """World direction -> equirectangular pixel (x, y) in cv2 convention."""
    d = np.asarray(d, dtype=np.float64)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    x = (lon / (2.0 * np.pi) + 0.5) * out_w - 0.5
    y = (0.5 - lat / np.pi) * out_h - 0.5
    return x, y


def output_to_lens_matrix(lens: LensParams, view: ViewParams) -> np.ndarray:
    """Row-vector matrix M so that d_lens = d_panorama @ M."""
    r_view = rotation(view.yaw, view.pitch, view.roll)  # panorama -> world
    r_lens = lens_rotation(lens)                       # lens -> world
    return r_view.T @ r_lens


def build_maps(out_w, out_h, lens: LensParams, view: ViewParams, y0=0, y1=None, proj=None):
    """float32 remap maps for output rows y0..y1, plus theta (degrees, float32, beyond FOV kept).

    proj is a reproject.Projection; None means the full equirectangular layout."""
    if proj is None:
        d = equirect_directions(out_w, out_h, y0, y1)
    else:
        from .reproject import output_directions
        d = output_directions(proj, out_w, out_h, y0, y1)
    d_lens = d @ output_to_lens_matrix(lens, view)
    x, y, theta = direction_to_source(d_lens, lens)
    valid = np.isfinite(x)
    map_x = np.where(valid, x, -1.0).astype(np.float32)
    map_y = np.where(valid, y, -1.0).astype(np.float32)
    return map_x, map_y, np.degrees(theta).astype(np.float32)


INTERP = {"cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4, "linear": cv2.INTER_LINEAR}


def render_layer(src: np.ndarray, lens: LensParams, view: ViewParams, out_w: int, out_h: int | None = None,
                 tile_rows: int = 256, interp: str = "lanczos"):
    """Warp one fisheye into the equirectangular frame, in row tiles to limit memory.

    Returns (layer, theta_deg). layer has the source dtype and is zero outside the lens FOV.
    theta_deg is the angle off this lens's axis for every output pixel (float32).
    """
    if out_h is None:
        out_h = out_w // 2
    layer = np.zeros((out_h, out_w, src.shape[2]), dtype=src.dtype)
    theta = np.empty((out_h, out_w), dtype=np.float32)
    flag = INTERP[interp]
    for y0 in range(0, out_h, tile_rows):
        y1 = min(out_h, y0 + tile_rows)
        mx, my, th = build_maps(out_w, out_h, lens, view, y0, y1)
        layer[y0:y1] = cv2.remap(src, mx, my, flag, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        theta[y0:y1] = th
    return layer, theta
