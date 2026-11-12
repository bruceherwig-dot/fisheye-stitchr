"""Fisheye circle helpers.

Coordinate convention (same as cv2.remap): the CENTER of pixel (i, j) is at x=i, y=j.
So the circle center (cx, cy) is in pixel-center units, and the viewer draws it
at scene position (cx + 0.5, cy + 0.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .imageio_utils import to_display_uint8


@dataclass
class Circle:
    cx: float
    cy: float
    r: float


def _fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    """Least-squares circle through Nx2 points (algebraic fit)."""
    x, y = points[:, 0], points[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    (ca, cb, cc), *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = ca / 2.0, cb / 2.0
    r = float(np.sqrt(max(cc + cx * cx + cy * cy, 0.0)))
    return float(cx), float(cy), r


def auto_detect_circle(array: np.ndarray) -> Circle | None:
    """Find the fisheye disc: a bright disc on a dark surround, or (as in Star Trail CleanR's pictures) a darker
    disc on a bright gray surround. A starting guess only; always editable."""
    return _bright_disc_on_dark(array) or _dark_disc_on_bright(array)


def _small_gray(array: np.ndarray):
    """Blurred, reduced gray copy (so stars and noise do not matter) and the reduction factor."""
    disp = to_display_uint8(array)
    gray = cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = max(1, max(h, w) // 1200)
    small = cv2.resize(gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (0, 0), 3), scale


def _dark_disc_on_bright(array: np.ndarray) -> Circle | None:
    """The disc is clearly darker than a bright, even surround (the margins left and right of it)."""
    small, scale = _small_gray(array)
    sh, sw = small.shape
    m = max(2, int(sw * 0.04))
    bg = float(np.median(np.concatenate([small[:, :m].ravel(), small[:, -m:].ravel()])))
    if bg < 100.0:                                       # a dark surround is the other detector's case
        return None
    mask = (small < bg - max(30.0, 0.2 * bg)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    big = max(contours, key=cv2.contourArea)
    if cv2.contourArea(big) < 0.02 * sh * sw:
        return None
    pts = big.reshape(-1, 2).astype(np.float64)
    edge = 3
    keep = (pts[:, 0] > edge) & (pts[:, 0] < sw - 1 - edge) & (pts[:, 1] > edge) & (pts[:, 1] < sh - 1 - edge)
    use = pts[keep] if keep.sum() >= 50 else pts
    cx, cy, r = _fit_circle(use)
    return Circle(cx=(cx + 0.5) * scale - 0.5, cy=(cy + 0.5) * scale - 0.5, r=r * scale)


def _bright_disc_on_dark(array: np.ndarray) -> Circle | None:
    """Find the bright fisheye disc on the dark surround.

    Works on a blurred, reduced copy so stars and noise do not matter. Returns None if
    no clear disc is found.
    """
    disp = to_display_uint8(array)
    gray = cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = max(1, max(h, w) // 1200)
    small = cv2.resize(gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    sh, sw = small.shape

    # Background level from the left and right margins (outside the disc).
    m = max(2, int(sw * 0.04))
    bg = float(np.median(np.concatenate([small[:, :m].ravel(), small[:, -m:].ravel()])))
    thresh = bg * 1.5 + 2.0
    mask = (small > thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    big = max(contours, key=cv2.contourArea)
    if cv2.contourArea(big) < 0.02 * sh * sw:
        return None
    pts = big.reshape(-1, 2).astype(np.float64)
    # Points on the frame border come from the sensor cutting the disc, not from its rim.
    edge = 3
    keep = (pts[:, 0] > edge) & (pts[:, 0] < sw - 1 - edge) & (pts[:, 1] > edge) & (pts[:, 1] < sh - 1 - edge)
    use = pts[keep]
    if len(use) < 50:
        use = pts
    cx, cy, r = _fit_circle(use)
    # Small-image pixel i has its center at (i + 0.5) * scale - 0.5 in full-size pixels.
    return Circle(cx=(cx + 0.5) * scale - 0.5, cy=(cy + 0.5) * scale - 0.5, r=r * scale)
