"""Automatic brightness and color matching between the two lenses of a one-photo (single exposure) shot.

Both lenses photograph the same scene along the seam, so the strip of sky and ground that both can see should
look the same. We measure each color channel in that strip (trimmed mean, so stars and noise do not matter) and
nudge the two lenses toward each other by the same amount in opposite directions. Nothing for the user to set.
"""
from __future__ import annotations

import cv2
import numpy as np

from .blend import world_lat_deg
from .projection import INTERP, ViewParams, build_maps

RING_LO, RING_HI = 80.0, 100.0        # degrees off each lens's axis: the strip both lenses see well
SKY_MIN_LAT = 3.0                     # measure the sky only: the ground is too dark at night to measure well
MAX_STEP = 2.0                        # never change a lens by more than this factor in total
SAMPLE_W = 1024                       # width of the small panorama used for the measurement


def _trimmed_mean(v):
    """Per-channel mean without the brightest and darkest 2 percent (stars, hot pixels, noise). A plain median
    is too coarse on a dark 8-bit sky, where nearly every pixel is 3."""
    lo, hi = np.percentile(v, [2, 98], axis=0)
    return np.array([v[:, c][(v[:, c] >= lo[c]) & (v[:, c] <= hi[c])].mean() for c in range(v.shape[1])])


def gains_from_layers(la, lb, tha, thb, lens_a, lens_b, lat=None):
    """Per-channel multipliers (gain_a, gain_b), each shape (3,), that make the shared strip match.

    la, lb: float HxWx3 layers on a 0..255 scale. tha, thb: angle off each lens's axis in degrees.
    lat: optional real-world latitude of each pixel; when given only the sky (above the horizon) is measured."""
    top_a = np.degrees(lens_a.theta_max) - 1.0
    top_b = np.degrees(lens_b.theta_max) - 1.0
    ring = ((tha >= RING_LO) & (tha <= min(RING_HI, top_a)) & (thb >= RING_LO) & (thb <= min(RING_HI, top_b)))
    if lat is not None:
        ring &= lat >= SKY_MIN_LAT
    one = np.ones(3, np.float32)
    if ring.sum() < 2000:
        return one, one
    a = _trimmed_mean(la[ring])
    b = _trimmed_mean(lb[ring])
    if min(a.min(), b.min()) < 1.0:                     # too dark to measure: leave things alone
        return one, one
    r = np.clip(a / b, 1.0 / MAX_STEP ** 2, MAX_STEP ** 2)       # how much brighter A is than B
    return (r ** -0.5).astype(np.float32), (r ** 0.5).astype(np.float32)


def estimate_gains(src_a, src_b, lens_a, lens_b):
    """Gains for two source pictures (already scaled to match their lens descriptions)."""
    w, h = SAMPLE_W, SAMPLE_W // 2
    layers, thetas = [], []
    for src, lens in ((src_a, lens_a), (src_b, lens_b)):
        mx, my, th = build_maps(w, h, lens, ViewParams(), 0, None, None)
        lay = cv2.remap(src, mx, my, INTERP["linear"], borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        top = 255.0 if src.dtype == np.uint8 else 65535.0
        layers.append(lay.astype(np.float32) * (255.0 / top))
        thetas.append(th)
    lat = world_lat_deg(ViewParams(), None, w, h)
    return gains_from_layers(layers[0], layers[1], thetas[0], thetas[1], lens_a, lens_b, lat)
