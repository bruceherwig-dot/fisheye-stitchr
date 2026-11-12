"""One photo that holds both fisheye circles side by side (Insta360 .insp, or any 2:1 dual-fisheye picture).

The photo is cut down the middle: the left half becomes picture A, the right half picture B. For cameras
we have measured, a preset supplies the circles and the lens settings so the join works straight away.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraPreset:
    name: str
    circle_radius_frac: float      # circle radius as a fraction of the half-photo's width
    fov_deg: float
    model: str
    trim_a: tuple                  # yaw, pitch, roll trim for lens A, degrees
    trim_b: tuple                  # yaw, pitch, roll trim for lens B, degrees
    center_a: tuple                # circle center of A as a fraction of the half's width, from the half's middle
    center_b: tuple


# Insta360 X4 .insp (11904 x 5952, each lens in its own 5952-pixel square). Fitted by lining our single-lens
# renders up with Insta360's own stitched frame of the same scene (horizon, trees and ground), using a separate
# circle center per lens. Lens A is the reference, so the trims keep only how B sits relative to A: the camera's
# own tilt is not baked in. Circle offsets are 5952-pixel units divided by 5952.
PRESETS = {
    "insta360 x4": CameraPreset(
        "Insta360 X4", 2890.0 / 5952.0, 202.1, "equidistant",
        trim_a=(0.0, 0.0, 0.0), trim_b=(0.1165, 1.376, 0.7455),
        center_a=(-0.6 / 5952.0, 0.0), center_b=(-6.9 / 5952.0, -5.9 / 5952.0)),
}


def camera_model(path: str) -> str:
    """The camera model written in the photo's EXIF, or an empty string."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return str(im.getexif().get(0x0110, "") or "")
    except Exception:
        return ""


# Exact photo sizes that identify a camera when the file has lost its EXIF camera tag (Star Trail CleanR's
# stacked and cleaned pictures are written without it). Only used when the tag is empty, never against a
# different named camera.
SIZE_HINTS = {(11904, 5952): "insta360 x4"}


def preset_for(path: str, size: tuple | None = None) -> CameraPreset | None:
    """The camera preset for a photo: by the EXIF model, or (tag missing) by exact (width, height)."""
    model = camera_model(path).strip().lower()
    for key, preset in PRESETS.items():
        if key in model:
            return preset
    if not model and size is not None and tuple(size) in SIZE_HINTS:
        return PRESETS[SIZE_HINTS[tuple(size)]]
    return None


def split_halves(arr: np.ndarray):
    """(left, right) halves of the photo, each its own contiguous copy."""
    h, w = arr.shape[:2]
    mid = w // 2
    return np.ascontiguousarray(arr[:, :mid]), np.ascontiguousarray(arr[:, mid:mid * 2])


def circle_for_half(half: np.ndarray, preset: CameraPreset | None, which: str = "a"):
    """Circle (cx, cy, r) for the A or B half-photo from a camera preset, or None when there is no preset."""
    if preset is None:
        return None
    h, w = half.shape[:2]
    dx, dy = preset.center_a if which == "a" else preset.center_b
    return (w / 2.0 + dx * w, h / 2.0 + dy * w, preset.circle_radius_frac * w)


def looks_two_to_one(arr: np.ndarray) -> bool:
    h, w = arr.shape[:2]
    return abs(w / max(h, 1) - 2.0) < 0.08
