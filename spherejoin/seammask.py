"""Painted seam mask.

The mask lives on the sphere, not on the screen, so it stays in the right place when you change the
output layout (equirectangular, Mercator, Gall) or turn the view. It is a 2048 x 1024 map of the whole
sky in world directions (lens A's frame, before the view rotation).

Each cell holds a number from -1 to +1:
    +1  always use image A here          0  let the automatic blend decide          -1  always use image B here
Values in between pull the blend part of the way. The brush is a circle in the PREVIEW, so what you
paint is what you see, and it is mapped onto the sphere exactly.
"""
from __future__ import annotations

import cv2
import numpy as np

from .projection import rotation
from .reproject import EQUIRECT, MAX_LAT, Projection, lonlat_to_pixel, output_directions, pixel_to_lonlat

CANVAS_W, CANVAS_H = 2048, 1024
MODES = ("a", "b", "erase")
_TARGET = {"a": 1.0, "b": -1.0, "erase": 0.0}
MAX_UNDO = 15


def _dirs_from_lonlat(lon, lat):
    cl = np.cos(lat)
    return np.stack([cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)], axis=-1)


class SeamMask:
    def __init__(self, w: int = CANVAS_W, h: int = CANVAS_H):
        self.w, self.h = w, h
        self.paint = np.zeros((h, w), np.float32)
        self._undo: list[np.ndarray] = []

    # ---- state ----
    @property
    def is_empty(self) -> bool:
        return not bool(np.any(self.paint))

    def push_undo(self):
        self._undo.append(self.paint.astype(np.float16))
        del self._undo[:-MAX_UNDO]

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.paint = self._undo.pop().astype(np.float32)
        return True

    def clear(self):
        self.push_undo()
        self.paint[:] = 0.0

    # ---- painting ----
    def stamp(self, view, proj: Projection, out_w: int, out_h: int, cx: float, cy: float, r_px: float,
              mode: str, softness: float = 0.5, opacity: float = 1.0):
        """Paint one round dab. cx, cy, r_px are in preview pixels (edges at whole numbers)."""
        if mode not in _TARGET:
            raise ValueError(f"Unknown paint mode: {mode}")
        target = _TARGET[mode]
        rot = rotation(view.yaw, view.pitch, view.roll)          # d_world = rot @ d_out

        def out_to_world(px, py):
            lon, lat = pixel_to_lonlat(proj, px, py, out_w, out_h)
            return _dirs_from_lonlat(lon, lat) @ rot.T

        dc = out_to_world(cx, cy)
        ang = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        dp = out_to_world(cx + r_px * np.cos(ang), cy + r_px * np.sin(ang))
        rho = float(np.max(np.arccos(np.clip(dp @ dc, -1.0, 1.0)))) * 1.25 + 1e-4
        lat_c = float(np.arcsin(np.clip(dc[1], -1.0, 1.0)))
        lon_c = float(np.arctan2(dc[0], dc[2]))

        lat0, lat1 = max(-np.pi / 2, lat_c - rho), min(np.pi / 2, lat_c + rho)
        edge = max(abs(lat0), abs(lat1))
        full = edge >= np.pi / 2 - 1e-3 or rho >= np.pi / 2
        j0 = int(np.clip(np.floor((0.5 - lat1 / np.pi) * self.h), 0, self.h - 1))
        j1 = int(np.clip(np.ceil((0.5 - lat0 / np.pi) * self.h), 0, self.h - 1))
        if full:
            cols_unwrapped = np.arange(self.w)
        else:
            dlon = min(np.pi, rho / max(np.cos(edge), 1e-3))
            i0 = int(np.floor(((lon_c - dlon) / (2 * np.pi) + 0.5) * self.w))
            i1 = int(np.ceil(((lon_c + dlon) / (2 * np.pi) + 0.5) * self.w))
            cols_unwrapped = np.arange(i0, i1 + 1)
            if len(cols_unwrapped) >= self.w:
                cols_unwrapped = np.arange(self.w)
        rows = np.arange(j0, j1 + 1)
        cols = cols_unwrapped % self.w
        lon = ((cols_unwrapped + 0.5) / self.w - 0.5) * 2 * np.pi
        lat = (0.5 - (rows + 0.5) / self.h) * np.pi
        lon_g, lat_g = np.meshgrid(lon, lat)
        d_out = _dirs_from_lonlat(lon_g, lat_g) @ rot            # canvas cell -> output direction
        lon_o = np.arctan2(d_out[..., 0], d_out[..., 2])
        lat_o = np.arcsin(np.clip(d_out[..., 1], -1.0, 1.0))
        if not proj.is_equirect and not proj.is_planet:
            lim = np.radians(MAX_LAT)
            lat_o = np.clip(lat_o, -lim, lim)
        px, py = lonlat_to_pixel(proj, lon_o, lat_o, out_w, out_h)
        dx = px - cx
        if not proj.is_planet:
            dx = (dx + out_w / 2.0) % out_w - out_w / 2.0           # the picture wraps left to right
        dist = np.hypot(dx, py - cy) / max(r_px, 1e-6)
        soft = float(np.clip(softness, 1e-3, 1.0))
        t = np.clip((dist - (1.0 - soft)) / soft, 0.0, 1.0)
        weight = np.where(dist <= 1.0, 0.5 * (1.0 + np.cos(np.pi * t)), 0.0).astype(np.float32) * float(opacity)
        ix = np.ix_(rows, cols)
        block = self.paint[ix]
        block += (target - block) * weight
        self.paint[ix] = block

    def stroke(self, view, proj, out_w, out_h, points, r_px, mode, softness=0.5, opacity=1.0):
        """Stamp along a polyline of preview points, close enough together to leave a smooth line."""
        step = max(r_px * 0.25, 1.0)
        last = None
        for x, y in points:
            if last is None:
                self.stamp(view, proj, out_w, out_h, x, y, r_px, mode, softness, opacity)
            else:
                dist = float(np.hypot(x - last[0], y - last[1]))
                n = max(1, int(np.ceil(dist / step)))
                for k in range(1, n + 1):
                    f = k / n
                    self.stamp(view, proj, out_w, out_h, last[0] + (x - last[0]) * f, last[1] + (y - last[1]) * f,
                               r_px, mode, softness, opacity)
            last = (x, y)

    # ---- reading ----
    def maps(self, view, proj: Projection | None, out_w: int, out_h: int, y0: int = 0, y1: int | None = None):
        """Remap maps (float32) that look the mask up for output rows y0..y1."""
        d = output_directions(proj or EQUIRECT, out_w, out_h, y0, y1)
        d_world = d @ rotation(view.yaw, view.pitch, view.roll).T
        lon = np.arctan2(d_world[..., 0], d_world[..., 2])
        lat = np.arcsin(np.clip(d_world[..., 1], -1.0, 1.0))
        mx = ((lon / (2 * np.pi) + 0.5) * self.w - 0.5).astype(np.float32)
        my = ((0.5 - lat / np.pi) * self.h - 0.5).astype(np.float32)
        return mx, my

    def sample_maps(self, mx, my) -> np.ndarray:
        return cv2.remap(self.paint, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def sample(self, view, proj, out_w: int, out_h: int, y0: int = 0, y1: int | None = None) -> np.ndarray:
        mx, my = self.maps(view, proj, out_w, out_h, y0, y1)
        return self.sample_maps(mx, my)

    # ---- saving ----
    def to_uint8(self) -> np.ndarray:
        """128 = automatic, 255 = always A, 1 = always B."""
        return (np.rint(np.clip(self.paint, -1, 1) * 127.0) + 128.0).astype(np.uint8)

    def save_png(self, path: str):
        ok, buf = cv2.imencode(".png", self.to_uint8())
        if not ok:
            raise IOError("Could not encode the seam mask")
        with open(path, "wb") as f:
            f.write(buf.tobytes())

    @classmethod
    def load_png(cls, path: str) -> "SeamMask":
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read the seam mask: {path}")
        h, w = img.shape
        m = cls(w, h)
        m.paint = ((img.astype(np.float32) - 128.0) / 127.0).clip(-1.0, 1.0)
        return m
