"""Synthetic test data: an equirectangular test pattern and a fisheye renderer using the SAME math.

Used to prove the stitcher rebuilds a known panorama. Not used by the app itself.
"""
from __future__ import annotations

import cv2
import numpy as np

from .projection import (LensParams, direction_to_equirect, lens_rotation, source_to_direction)


def make_test_equirect(w: int = 2048, h: int = 1024) -> np.ndarray:
    """Smooth colors, soft grid lines every 15 degrees, and colored blobs that reveal any flip."""
    lon = ((np.arange(w) + 0.5) / w - 0.5) * 360.0
    lat = (0.5 - (np.arange(h) + 0.5) / h) * 180.0
    lon, lat = np.meshgrid(lon, lat)
    img = np.empty((h, w, 3), np.float32)
    img[..., 0] = 110 + 70 * np.sin(np.radians(lon * 2.0)) * np.cos(np.radians(lat))
    img[..., 1] = 110 + 70 * np.cos(np.radians(lon * 1.0 + 30)) * np.cos(np.radians(lat * 2.0))
    img[..., 2] = 110 + 70 * np.sin(np.radians(lat * 3.0)) 
    lines = np.zeros((h, w), np.float32)
    lines[:, np.round((np.arange(-180, 180, 15) + 180) / 360.0 * w).astype(int) % w] = 1.0
    lines[np.round((90 - np.arange(-75, 90, 15)) / 180.0 * h).astype(int), :] = 1.0
    lines = cv2.GaussianBlur(lines, (0, 0), 3.0)
    lines /= lines.max()
    img = img * (1 - 0.6 * lines[..., None]) + 255 * 0.6 * lines[..., None]
    # Orientation blobs (R, G, B order): front red, right green, back blue, up yellow, down magenta.
    spots = [(0, 0, (255, 0, 0)), (90, 0, (0, 255, 0)), (180, 0, (0, 0, 255)),
             (0, 60, (255, 255, 0)), (0, -60, (255, 0, 255)), (-90, 0, (0, 255, 255))]
    for lo, la, col in spots:
        d = np.hypot((lon - lo + 180) % 360 - 180, lat - la)
        wgt = np.exp(-(d / 6.0) ** 2)[..., None]
        img = img * (1 - wgt) + np.array(col, np.float32) * wgt
    return np.clip(img, 0, 255).astype(np.uint8)


def render_fisheye_from_equirect(equi: np.ndarray, lens: LensParams, size: tuple[int, int]) -> np.ndarray:
    """Render what a lens with these parameters would record of the panorama. size = (width, height)."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    d_lens = source_to_direction(xx, yy, lens)
    d_world = d_lens @ lens_rotation(lens).T
    ex, ey = direction_to_equirect(d_world, equi.shape[1], equi.shape[0])
    out = cv2.remap(equi, ex.astype(np.float32), ey.astype(np.float32), cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_WRAP)
    inside = np.hypot(xx - lens.cx, yy - lens.cy) <= lens.radius
    out[~inside] = 0
    return out
