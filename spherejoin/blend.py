"""Combine the two warped fisheye layers into one equirectangular panorama.

Seam model: weights are driven by u, which is theta_A (the angle off lens A's axis) when the lenses face exactly
opposite ways, and otherwise puts the seam where a point is the same angle off both lenses (see seam_coordinate).
  u <= seam - feather/2  -> only A
  u >= seam + feather/2  -> only B
  in between             -> smooth fade (smoothstep). hard_cut makes the fade zero width.
Where only one lens sees a direction, that lens wins regardless of the weights.

The feather is narrow in the sky and wider on the ground. Star trails from the two photos do not match,
so blending them only ghosts them; the ground shifts a little between the shots and likes a soft blend.
Sky versus ground is decided by the real latitude of each direction (above or below the horizon), not by
where it lands in the picture, so turning the output view does not move it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .projection import LensParams, ViewParams, build_maps, INTERP, rotation
from .reproject import EQUIRECT, output_directions

SKY_LAT_LO = 1.0     # degrees above the horizon where the blend starts narrowing
SKY_LAT_HI = 7.0     # ... and where it has fully narrowed (clears low hills along the horizon)


@dataclass
class BlendParams:
    seam_deg: float = 90.0            # theta_A where the two sides meet
    feather_deg: float = 20.0         # width of the fade on the ground, centered on the seam
    sky_feather_deg: float = 7.0      # width of the fade in the sky
    hard_cut: bool = False
    gain_a: float = 1.0               # brightness multiplier per image
    gain_b: float = 1.0
    wb_a: tuple = (1.0, 1.0, 1.0)     # per-channel (R, G, B) multipliers per image
    wb_b: tuple = (1.0, 1.0, 1.0)
    sky_gain_a: tuple | None = None   # per-channel (R, G, B) multipliers applied to the sky only (fading in across the
    sky_gain_b: tuple | None = None   # horizon); set automatically to even out the two lenses' sky brightness


def world_lat_deg(view: ViewParams, proj, out_w: int, out_h: int, y0: int = 0, y1: int | None = None) -> np.ndarray:
    """Latitude (degrees, float32) of the real-world direction behind each output pixel, for rows y0..y1."""
    d = output_directions(proj or EQUIRECT, out_w, out_h, y0, y1)
    d_world = d @ rotation(view.yaw, view.pitch, view.roll).T
    return np.degrees(np.arcsin(np.clip(d_world[..., 1], -1.0, 1.0))).astype(np.float32)


def sky_factor(gain, lat_deg):
    """Per-pixel, per-channel multiplier: 1 on the ground, the full gain in the sky, a smooth change across the
    horizon band (the same band the sky/ground blend width uses)."""
    s = np.clip((np.asarray(lat_deg, dtype=np.float32) - SKY_LAT_LO) / (SKY_LAT_HI - SKY_LAT_LO), 0.0, 1.0)
    s = (s * s * (3.0 - 2.0 * s))[..., None]
    return 1.0 + (np.asarray(gain, dtype=np.float32) - 1.0) * s


def feather_array(bp: BlendParams, lat_deg=None):
    """Feather width per pixel: the ground width below the horizon band, the sky width above it, and a
    smooth change between. A scalar when no latitude is given (then it is just the ground width)."""
    if bp.hard_cut:
        return 0.0
    ground = max(bp.feather_deg, 0.0)
    if lat_deg is None:
        return ground
    sky = max(bp.sky_feather_deg, 0.0)
    s = np.clip((lat_deg - SKY_LAT_LO) / (SKY_LAT_HI - SKY_LAT_LO), 0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)
    return ground * (1.0 - s) + sky * s


def seam_coordinate(theta_a, theta_b):
    """Where a point sits between the two lenses, as an angle that equals theta_a when the lenses face exactly
    opposite ways. When lens B is turned or tilted relative to A, the seam still falls where the point is the same
    angle off both axes, in the middle of each overlap, instead of drifting toward one lens's glowing rim."""
    return (np.asarray(theta_a, dtype=np.float32) - np.asarray(theta_b, dtype=np.float32)) / 2.0 + 90.0


def blend_weight_a(theta_a, theta_b, lens_a: LensParams, lens_b: LensParams, bp: BlendParams, paint=None,
                   lat_deg=None):
    """Weight of lens A (0..1) per output pixel. Returns (w_a, any_valid).

    paint is the optional painted seam mask for these pixels: +1 forces image A, -1 forces image B,
    values between pull the automatic blend part of the way. Where only one lens sees a direction,
    that lens still wins, whatever was painted."""
    tmax_a, tmax_b = np.degrees(lens_a.theta_max), np.degrees(lens_b.theta_max)
    valid_a = theta_a <= tmax_a
    valid_b = theta_b <= tmax_b
    half = np.asarray(feather_array(bp, lat_deg), dtype=np.float32) / 2.0
    safe = np.maximum(half, 1e-6)
    u = seam_coordinate(theta_a, theta_b)
    t = np.clip((u - (bp.seam_deg - half)) / (2.0 * safe), 0.0, 1.0)
    smooth = 1.0 - t * t * (3.0 - 2.0 * t)                    # smoothstep, 1 -> 0
    w = np.where(half <= 1e-6, (u <= bp.seam_deg).astype(np.float32), smooth).astype(np.float32)
    if paint is not None:
        pos = np.clip(paint, 0.0, 1.0)
        neg = np.clip(-paint, 0.0, 1.0)
        w = (w + (1.0 - w) * pos) * (1.0 - neg)
    w = np.where(valid_a & ~valid_b, 1.0, np.where(~valid_a & valid_b, 0.0, w)).astype(np.float32)
    return w, (valid_a | valid_b)


def _channel_gain(gain: float, wb: tuple, channels: int) -> np.ndarray:
    g = np.full(channels, gain, dtype=np.float32)
    g[: min(3, channels)] *= np.asarray(wb[: min(3, channels)], dtype=np.float32)
    return g


def _warp_tile(src, lens, view, out_w, out_h, y0, y1, interp, proj=None):
    mx, my, th = build_maps(out_w, out_h, lens, view, y0, y1, proj)
    layer = cv2.remap(src, mx, my, INTERP[interp], borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return layer, th


def _dtype_max(dtype) -> float:
    return 255.0 if dtype == np.uint8 else 65535.0


def compose(src_a: np.ndarray, src_b: np.ndarray, lens_a: LensParams, lens_b: LensParams,
            view: ViewParams, bp: BlendParams, out_w: int, out_h: int | None = None,
            tile_rows: int = 256, interp: str = "lanczos",
            progress: Callable[[float], None] | None = None, proj=None, mask=None) -> np.ndarray:
    """Full blended panorama, built in row tiles. Output keeps the source bit depth.

    proj is a reproject.Projection (None = equirectangular, 2:1). With a projection and no out_h, the
    height follows from the width and the projection's latitude range.
    progress, if given, is called with the fraction done after each tile (it may raise to cancel)."""
    if out_h is None:
        if proj is None:
            out_h = out_w // 2
        else:
            from .reproject import output_size
            out_h = output_size(proj, out_w)[1]
    if src_a.dtype != src_b.dtype or src_a.shape[2] != src_b.shape[2]:
        raise ValueError("Both images must have the same bit depth and channel count")
    ch = src_a.shape[2]
    top = _dtype_max(src_a.dtype)
    ga = _channel_gain(bp.gain_a, bp.wb_a, ch)
    gb = _channel_gain(bp.gain_b, bp.wb_b, ch)
    out = np.zeros((out_h, out_w, ch), dtype=src_a.dtype)
    for y0 in range(0, out_h, tile_rows):
        y1 = min(out_h, y0 + tile_rows)
        la, tha = _warp_tile(src_a, lens_a, view, out_w, out_h, y0, y1, interp, proj)
        lb, thb = _warp_tile(src_b, lens_b, view, out_w, out_h, y0, y1, interp, proj)
        paint = None
        if mask is not None and not mask.is_empty:
            paint = mask.sample(view, proj, out_w, out_h, y0, y1)
        lat = world_lat_deg(view, proj, out_w, out_h, y0, y1)
        w, any_valid = blend_weight_a(tha, thb, lens_a, lens_b, bp, paint, lat)
        w = w[..., None]
        la_f, lb_f = la.astype(np.float32), lb.astype(np.float32)
        if bp.sky_gain_a is not None:
            la_f = la_f * sky_factor(bp.sky_gain_a, lat)
        if bp.sky_gain_b is not None:
            lb_f = lb_f * sky_factor(bp.sky_gain_b, lat)
        mix = la_f * ga * w + lb_f * gb * (1.0 - w)
        mix[~any_valid] = 0.0
        out[y0:y1] = np.clip(np.rint(mix), 0, top).astype(src_a.dtype)
        if progress is not None:
            progress(y1 / out_h)
    return out


def overlap_info(lens_a: LensParams, lens_b: LensParams, view: ViewParams, out_w: int, out_h: int | None = None):
    """Overlap tools for the preview: (theta_a, theta_b, overlap_mask). Both lens angles in degrees."""
    if out_h is None:
        out_h = out_w // 2
    _, _, tha = build_maps(out_w, out_h, lens_a, view)
    _, _, thb = build_maps(out_w, out_h, lens_b, view)
    overlap = (tha <= np.degrees(lens_a.theta_max)) & (thb <= np.degrees(lens_b.theta_max))
    return tha, thb, overlap


def overlap_difference(src_a, src_b, lens_a, lens_b, view, bp: BlendParams, out_w, out_h=None, interp="linear"):
    """Absolute per-pixel difference of the two gain-adjusted layers inside the overlap, as float32 HxW
    (mean over channels, in source units). Zero outside the overlap. For judging alignment by eye."""
    if out_h is None:
        out_h = out_w // 2
    ch = src_a.shape[2]
    ga = _channel_gain(bp.gain_a, bp.wb_a, ch)
    gb = _channel_gain(bp.gain_b, bp.wb_b, ch)
    la, tha = _warp_tile(src_a, lens_a, view, out_w, out_h, 0, out_h, interp)
    lb, thb = _warp_tile(src_b, lens_b, view, out_w, out_h, 0, out_h, interp)
    ov = (tha <= np.degrees(lens_a.theta_max)) & (thb <= np.degrees(lens_b.theta_max))
    d = np.abs(la.astype(np.float32) * ga - lb.astype(np.float32) * gb).mean(axis=2)
    d[~ov] = 0.0
    return d
