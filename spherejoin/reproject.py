"""Output layouts for the finished image.

All three keep longitude on the horizontal axis and differ in how latitude maps to the vertical axis:
  equirectangular  y = lat                      full sphere, 2:1. Pole rings stretch into ovals.
  mercator         y = ln(tan(45deg + lat/2))   conformal: small circles stay round. Cannot reach the pole.
  gall             y = (1 + sqrt2/2) tan(lat/2) cylindrical stereographic: a gentler middle ground.
  planet           "tiny planet": a square picture. The output view direction is the middle of the picture and
                   the sky curls out around it (stereographic). lat_top_deg is how far from the middle, in
                   degrees, the edge of the picture reaches; lat_bot_deg is not used.

Rows and columns use pixel CENTERS, the same convention as projection.equirect_directions, so the
equirectangular layout with the full latitude range is identical to the original output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINDS = ("equirectangular", "mercator", "gall", "planet")
DEFAULT_KIND = "mercator"      # what a fresh session starts with
KIND_LABELS = {"equirectangular": "Equirectangular (360 x 180)", "mercator": "Mercator (round pole rings)",
               "gall": "Gall stereographic (gentler)", "planet": "Tiny planet (square)"}
DEFAULT_RANGE = {"equirectangular": (90.0, -90.0), "mercator": (75.0, -60.0), "gall": (85.0, -70.0), "planet": (150.0, 0.0)}
PLANET_REACH = (60.0, 175.0)     # allowed range of the planet reach, degrees from the middle to the edge
MAX_LAT = 88.0          # Mercator and Gall stretch without limit toward the pole
_GALL_K = 1.0 + np.sqrt(2.0) / 2.0


@dataclass(frozen=True)
class Projection:
    kind: str = "equirectangular"
    lat_top_deg: float = 90.0
    lat_bot_deg: float = -90.0

    @staticmethod
    def default(kind: str) -> "Projection":
        top, bot = DEFAULT_RANGE[kind]
        return Projection(kind, top, bot)

    @property
    def is_equirect(self) -> bool:
        return self.kind == "equirectangular"

    @property
    def is_planet(self) -> bool:
        return self.kind == "planet"


EQUIRECT = Projection()


def _y_of_lat(kind: str, lat):
    if kind == "equirectangular":
        return lat
    if kind == "mercator":
        return np.log(np.tan(np.pi / 4.0 + lat / 2.0))
    if kind == "gall":
        return _GALL_K * np.tan(lat / 2.0)
    raise ValueError(f"Unknown projection: {kind}")


def _lat_of_y(kind: str, y):
    if kind == "equirectangular":
        return y
    if kind == "mercator":
        return 2.0 * np.arctan(np.exp(y)) - np.pi / 2.0
    if kind == "gall":
        return 2.0 * np.arctan(y / _GALL_K)
    raise ValueError(f"Unknown projection: {kind}")


def _x_span(kind: str) -> float:
    return 2.0 * np.pi / np.sqrt(2.0) if kind == "gall" else 2.0 * np.pi


def _range(proj: Projection) -> tuple[float, float]:
    """(y_top, y_bottom) in the projection's own units, after checking the latitude range."""
    top, bot = proj.lat_top_deg, proj.lat_bot_deg
    if proj.is_planet:
        raise ValueError("The tiny planet layout has no latitude range")
    limit = 90.0 if proj.is_equirect else MAX_LAT
    if not (-limit <= bot < top <= limit):
        raise ValueError(f"Latitude range must satisfy -{limit:g} <= bottom < top <= {limit:g}")
    return float(_y_of_lat(proj.kind, np.radians(top))), float(_y_of_lat(proj.kind, np.radians(bot)))


def output_size(proj: Projection, width: int) -> tuple[int, int]:
    """(width, height) in pixels for a given width."""
    if proj.is_planet:
        return int(width), int(width)
    yt, yb = _range(proj)
    return int(width), max(1, int(round(width * (yt - yb) / _x_span(proj.kind))))


def _planet_scale(proj: Projection) -> float:
    """tan(reach/2): how far out (in stereographic units) the edge of the picture sits."""
    return float(np.tan(np.radians(np.clip(proj.lat_top_deg, *PLANET_REACH)) / 2.0))


def _planet_to_lonlat(proj, px, py, out_w, out_h):
    """Square picture -> direction. The middle of the picture is the straight-ahead direction (lon 0, lat 0);
    the picture's half width is the planet reach."""
    u = (px / out_w - 0.5) * 2.0
    v = (0.5 - py / out_h) * 2.0
    r = np.hypot(u, v)
    psi = 2.0 * np.arctan(r * _planet_scale(proj))                 # angle away from the middle
    k = np.where(r > 1e-12, np.sin(psi) / np.maximum(r, 1e-12), 0.0)
    dx, dy, dz = u * k, v * k, np.cos(psi)
    return np.arctan2(dx, dz), np.arcsin(np.clip(dy, -1.0, 1.0))


def _lonlat_to_planet(proj, lon, lat, out_w, out_h):
    dx, dy, dz = np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
    psi = np.arccos(np.clip(dz, -1.0, 1.0))
    s = np.sin(psi)
    r = np.tan(psi / 2.0) / _planet_scale(proj)
    k = np.where(s > 1e-12, r / np.maximum(s, 1e-12), 0.0)
    u, v = dx * k, dy * k
    return (u / 2.0 + 0.5) * out_w, (0.5 - v / 2.0) * out_h


def pixel_to_lonlat(proj: Projection, px, py, out_w: int, out_h: int):
    """Output position -> (lon, lat) in radians. px, py have pixel EDGES at whole numbers, so the
    middle of the first pixel is 0.5. Works on arrays."""
    px, py = np.asarray(px, dtype=np.float64), np.asarray(py, dtype=np.float64)
    if proj.is_planet:
        return _planet_to_lonlat(proj, px, py, out_w, out_h)
    yt, yb = _range(proj)
    lon = (px / out_w - 0.5) * 2.0 * np.pi
    lat = _lat_of_y(proj.kind, yt - py / out_h * (yt - yb))
    return lon, lat


def lonlat_to_pixel(proj: Projection, lon, lat, out_w: int, out_h: int):
    """Inverse of pixel_to_lonlat (same edge convention)."""
    lon, lat = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    if proj.is_planet:
        return _lonlat_to_planet(proj, lon, lat, out_w, out_h)
    yt, yb = _range(proj)
    px = (lon / (2.0 * np.pi) + 0.5) * out_w
    py = (yt - _y_of_lat(proj.kind, lat)) / (yt - yb) * out_h
    return px, py


def output_directions(proj: Projection, out_w: int, out_h: int, y0: int = 0, y1: int | None = None) -> np.ndarray:
    """Unit directions (panorama frame) for output rows y0..y1, pixel centers. Shape (rows, out_w, 3)."""
    if y1 is None:
        y1 = out_h
    cols = np.arange(out_w) + 0.5
    rows = np.arange(y0, y1) + 0.5
    px, py = np.meshgrid(cols, rows)
    lon, lat = pixel_to_lonlat(proj, px, py, out_w, out_h)
    cl = np.cos(lat)
    return np.stack([cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)], axis=-1)
