"""Preview rendering (no GUI). Geometry is warped once into float layers; showing a different
mode, gain, feather or overlay only re-composes those layers, which is cheap."""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .blend import (BlendParams, _channel_gain, _dtype_max, blend_weight_a, feather_array, seam_coordinate, sky_factor,
                    world_lat_deg)
from .projection import INTERP, LensParams, ViewParams, build_maps, rotation
from .reproject import EQUIRECT, Projection, output_size, pixel_to_lonlat

MODES = ("blend", "a_only", "b_only", "flicker", "difference")
SEAM_COLOR = np.array([255, 230, 0], np.float32)
FEATHER_COLOR = np.array([255, 130, 0], np.float32)
OVERLAP_COLOR = np.array([255, 0, 255], np.float32)
MASK_A_COLOR = np.array([40, 230, 80], np.float32)
MASK_B_COLOR = np.array([60, 140, 255], np.float32)


# ---------------------------------------------------------------- proxies (shrunk sources)
def make_proxy(array: np.ndarray, max_dim: int = 3400):
    """Shrunk copy for fast previews. Returns (array, (sx, sy)) with the scale per axis."""
    h, w = array.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    if s >= 1.0:
        return array, (1.0, 1.0)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = cv2.resize(array, (nw, nh), interpolation=cv2.INTER_AREA)
    return small, (nw / w, nh / h)


def scale_lens(lens: LensParams, scale: tuple[float, float]) -> LensParams:
    """Move a lens description onto a shrunk copy of its image."""
    sx, sy = scale
    return replace(lens, cx=(lens.cx + 0.5) * sx - 0.5, cy=(lens.cy + 0.5) * sy - 0.5,
                   radius=lens.radius * (sx + sy) / 2.0)


# ---------------------------------------------------------------- warping
@dataclass
class PreviewLayers:
    la: np.ndarray        # float32 HxWx3, 0..255 scale, zero outside lens A
    lb: np.ndarray
    tha: np.ndarray       # float32 HxW, degrees off axis
    thb: np.ndarray
    valid_a: np.ndarray   # bool HxW
    valid_b: np.ndarray
    view: object = None   # what these layers were warped with; the seam mask needs it
    proj: object = None
    mask_maps: object = None   # cached look-up maps into the seam mask, filled on first use
    lat: object = None         # float32 HxW: real-world latitude of each pixel, for the sky/ground blend


def warp_preview_layers(proxy_a, proxy_b, lens_a: LensParams, lens_b: LensParams, view: ViewParams,
                        out_w: int, out_h: int | None = None, interp: str = "linear",
                        proj=None) -> PreviewLayers:
    if out_h is None:
        out_h = out_w // 2 if proj is None else output_size(proj, out_w)[1]
    layers, thetas = [], []
    for src, lens in ((proxy_a, lens_a), (proxy_b, lens_b)):
        mx, my, th = build_maps(out_w, out_h, lens, view, 0, None, proj)
        lay = cv2.remap(src, mx, my, INTERP[interp], borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        layers.append(lay.astype(np.float32) * (255.0 / _dtype_max(src.dtype)))
        thetas.append(th)
    va = thetas[0] <= np.degrees(lens_a.theta_max)
    vb = thetas[1] <= np.degrees(lens_b.theta_max)
    return PreviewLayers(layers[0], layers[1], thetas[0], thetas[1], va, vb, view=view, proj=proj,
                         lat=world_lat_deg(view, proj, out_w, out_h))


# ---------------------------------------------------------------- composing
def _edge_mask(inside: np.ndarray) -> np.ndarray:
    e = np.zeros(inside.shape, bool)
    e[:, 1:] |= inside[:, 1:] != inside[:, :-1]
    e[1:, :] |= inside[1:, :] != inside[:-1, :]
    return cv2.dilate(e.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)


def compose_preview(L: PreviewLayers, lens_a: LensParams, lens_b: LensParams, bp: BlendParams,
                    mode: str = "blend", show_seam: bool = False, show_overlap: bool = False,
                    flicker_phase: int = 0, mask=None, show_mask: bool = False) -> np.ndarray:
    """uint8 RGB preview from warped layers."""
    if mode not in MODES:
        raise ValueError(f"Unknown preview mode: {mode}")
    ga = _channel_gain(bp.gain_a, bp.wb_a, 3)
    gb = _channel_gain(bp.gain_b, bp.wb_b, 3)
    A, B = L.la * ga, L.lb * gb
    if L.lat is not None:
        if bp.sky_gain_a is not None:
            A = A * sky_factor(bp.sky_gain_a, L.lat)
        if bp.sky_gain_b is not None:
            B = B * sky_factor(bp.sky_gain_b, L.lat)
    paint = None
    if mask is not None and (show_mask or not mask.is_empty):
        h, wd = L.tha.shape
        if L.mask_maps is None:
            L.mask_maps = mask.maps(L.view, L.proj, wd, h)
        paint = mask.sample_maps(*L.mask_maps)
    w, anyv = blend_weight_a(L.tha, L.thb, lens_a, lens_b, bp, paint, L.lat)
    w3 = w[..., None]
    blend = A * w3 + B * (1.0 - w3)
    blend[~anyv] = 0.0
    ov = L.valid_a & L.valid_b

    if mode == "blend":
        out = blend
    elif mode == "a_only":
        out = np.where(L.valid_a[..., None], A, 0.0)
    elif mode == "b_only":
        out = np.where(L.valid_b[..., None], B, 0.0)
    elif mode == "flicker":
        src = A if flicker_phase % 2 == 0 else B
        out = np.where(ov[..., None], src, blend)
    else:  # difference
        d = np.abs(A - B).mean(axis=2) * 4.0
        out = blend * 0.25
        out[ov] = d[ov][:, None]
    out = np.clip(out, 0.0, 255.0)

    if show_mask and paint is not None:
        a = np.clip(paint, 0.0, 1.0)[..., None] * 0.45        # painted "use A": green
        b = np.clip(-paint, 0.0, 1.0)[..., None] * 0.45       # painted "use B": blue
        out = out * (1.0 - a - b) + MASK_A_COLOR * a + MASK_B_COLOR * b
    if show_overlap:
        out[ov] = out[ov] * 0.72 + OVERLAP_COLOR * 0.28
    if show_seam:
        seam_u = seam_coordinate(L.tha, L.thb)
        half = np.asarray(feather_array(bp, L.lat), dtype=np.float32) / 2.0
        if np.any(half > 0):
            for edge in (bp.seam_deg - half, bp.seam_deg + half):
                out[_edge_mask(seam_u <= edge)] = FEATHER_COLOR
        out[_edge_mask(seam_u <= bp.seam_deg)] = SEAM_COLOR
    return out.astype(np.uint8)


# ---------------------------------------------------------------- "make this point the center"
def view_for_center(view: ViewParams, sx: float, sy: float, out_w: int, out_h: int,
                    proj: Projection = EQUIRECT) -> ViewParams:
    """New view so the clicked preview point becomes the panorama center.

    sx, sy are positions in preview pixels with pixel EDGES at whole numbers (a click on the
    middle of the first pixel is 0.5, 0.5). proj is the layout being shown. Roll is kept.
    """
    lon, lat = pixel_to_lonlat(proj, sx, sy, out_w, out_h)
    lon, lat = float(lon), float(lat)
    d_out = np.array([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)])
    d_world = rotation(view.yaw, view.pitch, view.roll) @ d_out
    yaw = float(np.degrees(np.arctan2(d_world[0], d_world[2])))
    pitch = float(np.degrees(np.arcsin(np.clip(d_world[1], -1.0, 1.0))))
    return ViewParams(yaw=yaw, pitch=pitch, roll=view.roll)
