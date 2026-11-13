"""Image loading. Returns RGB arrays, uint8 or uint16, with the file's own bit depth kept."""
from __future__ import annotations

import os

import cv2
import numpy as np
import tifffile

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".insp")   # .insp is an Insta360 JPEG


def open_failure_text(path: str, err: Exception) -> str:
    """Plain-English message for a photo that would not open, with the likely reason first."""
    name = os.path.basename(path) or path
    if not os.path.exists(path):
        why = "The file was not found. It may have been moved or renamed."
    elif isinstance(err, PermissionError) or not os.access(path, os.R_OK):
        why = ("The app is not allowed to read this file. If it is in Desktop, Documents or Downloads, "
               "allow Fisheye StitchR under System Settings > Privacy & Security > Files and Folders.")
    elif os.path.splitext(path)[1].lower() not in SUPPORTED_EXTENSIONS:
        why = "This kind of file is not supported. Use JPEG, PNG, TIFF or Insta360 .insp."
    else:
        why = "The file may be damaged or not a normal picture."
    detail = str(err).strip()
    detail = f"\n\nDetails: {detail}" if detail and detail != path else ""
    return f"Could not open {name}.\n\n{why}{detail}"


def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """Make any channel layout into HxWx3 RGB, dropping alpha and expanding gray."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError(f"Unsupported channel layout: shape {arr.shape}")
    return np.ascontiguousarray(arr)


def _normalize_dtype(arr: np.ndarray) -> np.ndarray:
    """Keep uint8 and uint16 as they are. Floats are scaled into uint16."""
    if arr.dtype in (np.uint8, np.uint16):
        return arr
    if arr.dtype == np.int16:
        return np.clip(arr.astype(np.int32) + 32768, 0, 65535).astype(np.uint16)
    if np.issubdtype(arr.dtype, np.floating):
        top = float(arr.max()) if arr.size else 1.0
        scale = 65535.0 if top <= 1.0 else 65535.0 / top
        return np.clip(arr * scale, 0, 65535).astype(np.uint16)
    raise ValueError(f"Unsupported pixel type: {arr.dtype}")


def load_image(path: str) -> np.ndarray:
    """Load JPG, PNG or TIFF as an RGB array. 16 bit stays 16 bit.

    EXIF rotation is NOT applied, so pixel coordinates always match the stored file.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    if ext in (".tif", ".tiff"):
        arr = tifffile.imread(path)
        # Some TIFFs store channels first, or hold several pages. Take the first page.
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 4:
            arr = arr[0]
        arr = _normalize_dtype(arr)
        return _to_rgb(arr)

    # JPG and PNG go through OpenCV. Reading the bytes ourselves keeps odd paths safe.
    data = np.fromfile(path, dtype=np.uint8)
    flags = cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION
    arr = cv2.imdecode(data, flags)
    if arr is None:
        raise ValueError(f"Could not decode image: {path}")
    arr = _normalize_dtype(arr)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, [2, 1, 0] + list(range(3, arr.shape[2]))]  # BGR(A) -> RGB(A)
    return _to_rgb(arr)


def to_display_uint8(arr: np.ndarray) -> np.ndarray:
    """8 bit copy for the screen. 16 bit drops its low byte, which is exact for display."""
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray((arr >> 8).astype(np.uint8))
