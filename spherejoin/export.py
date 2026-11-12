"""Full-resolution export: the panorama, the two warped layers, color profile, spherical metadata."""
from __future__ import annotations

import io
import os
import shutil
import struct
import subprocess
import zlib
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
import tifffile
from PIL import Image

from .blend import BlendParams, _channel_gain, _dtype_max, compose
from .imageio_utils import to_display_uint8
from .projection import INTERP, LensParams, ViewParams, build_maps, radius_from_theta
from .reproject import EQUIRECT, Projection, output_size

FORMATS = ("tiff", "png", "jpg")
EXTENSIONS = {"tiff": ".tif", "png": ".png", "jpg": ".jpg"}
Progress = Callable[[float, str], None]


class ExportCancelled(Exception):
    pass


@dataclass
class ExportSettings:
    width: int = 8192
    fmt: str = "jpg"
    sixteen_bit: bool = False        # write 16 bit even when the source is 8 bit (headroom for editing)
    interp: str = "lanczos"
    jpg_quality: int = 95
    save_layers: bool = False
    spherical_metadata: bool = False
    use_mask: bool = True            # apply the painted seam mask if there is one


@dataclass
class ExportResult:
    paths: list = field(default_factory=list)
    messages: list = field(default_factory=list)


# ---------------------------------------------------------------- sizing
def recommended_width(lens_a: LensParams, lens_b: LensParams) -> int:
    """Width that keeps the sharpest part of the fisheye at full detail.

    Pixels per radian at the lens axis is dr/dtheta there; the panorama has width/(2 pi) pixels per
    radian, so width = 2 pi * dr/dtheta. Averaged over the two lenses, rounded to a multiple of 16.
    """
    eps = 1e-4
    f = [float(radius_from_theta(eps, l)) / eps for l in (lens_a, lens_b)]
    w = 2.0 * np.pi * (sum(f) / 2.0)
    return int(max(1024, round(w / 16.0) * 16))


def estimate_memory_mb(width: int, bytes_per_sample: int, layers: bool, height_over_width: float = 0.5) -> float:
    one = width * (width * height_over_width) * 3 * bytes_per_sample
    return one * (3.0 if layers else 1.6) / 1e6


_TAGS = {"equirectangular": "equirect", "mercator": "mercator", "gall": "gall", "planet": "tinyplanet"}


def default_output_path(path_a: str | None, fmt: str, kind: str = "equirectangular") -> str:
    base = os.path.splitext(os.path.basename(path_a))[0] if path_a else "panorama"
    folder = os.path.dirname(path_a) if path_a else os.path.expanduser("~")
    return os.path.join(folder, f"{base}_spherejoin_{_TAGS.get(kind, kind)}{EXTENSIONS[fmt]}")


# ---------------------------------------------------------------- color profile
def read_icc(path: str | None) -> bytes | None:
    if not path:
        return None
    try:
        with Image.open(path) as im:
            return im.info.get("icc_profile")
    except Exception:
        return None


def inject_icc_png(png: bytes, icc: bytes) -> bytes:
    """Insert an iCCP chunk right after IHDR."""
    data = b"ICC Profile\x00\x00" + zlib.compress(icc)
    chunk = struct.pack(">I", len(data)) + b"iCCP" + data + struct.pack(">I", zlib.crc32(b"iCCP" + data) & 0xFFFFFFFF)
    cut = 8 + 25  # signature + IHDR chunk
    return png[:cut] + chunk + png[cut:]


# ---------------------------------------------------------------- writing one file
def _depth(arr: np.ndarray, want16: bool) -> np.ndarray:
    if want16 and arr.dtype == np.uint8:
        return arr.astype(np.uint16) * 257
    return arr


def write_image(path: str, arr: np.ndarray, fmt: str, icc: bytes | None = None, jpg_quality: int = 95):
    """arr is HxWx3 or HxWx4, uint8 or uint16. JPEG drops to 8 bit; it cannot hold alpha."""
    if fmt == "tiff":
        extra = ["unassalpha"] if arr.shape[2] == 4 else None
        tifffile.imwrite(path, arr, photometric="rgb", compression="zlib", compressionargs={"level": 3},
                         iccprofile=icc, extrasamples=extra)
    elif fmt == "png":
        if arr.dtype == np.uint16:
            bgr = arr[:, :, [2, 1, 0] + ([3] if arr.shape[2] == 4 else [])]
            ok, buf = cv2.imencode(".png", np.ascontiguousarray(bgr), [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                raise IOError("PNG encoding failed")
            data = buf.tobytes()
            if icc:
                data = inject_icc_png(data, icc)
            with open(path, "wb") as f:
                f.write(data)
        else:
            Image.fromarray(arr).save(path, format="PNG", icc_profile=icc, compress_level=3)
    elif fmt == "jpg":
        if arr.shape[2] == 4:
            raise ValueError("JPEG cannot hold transparency")
        Image.fromarray(to_display_uint8(arr)).save(path, format="JPEG", quality=int(jpg_quality),
                                                    subsampling=0, icc_profile=icc)
    else:
        raise ValueError(f"Unknown format: {fmt}")


# ---------------------------------------------------------------- layers with transparency
def render_layer_rgba(src: np.ndarray, lens: LensParams, view: ViewParams, gain: float, wb: tuple,
                      out_w: int, out_h: int | None = None, interp: str = "lanczos", rim_soft_deg: float = 2.0,
                      tile_rows: int = 256, progress: Callable[[float], None] | None = None,
                      proj=None) -> np.ndarray:
    """One warped lens as RGBA. Alpha is 1 inside the lens field of view and fades to 0 over the last
    rim_soft_deg degrees, so the soft edge of the circle does not leave a fringe. Gain and white
    balance are applied so the layer looks the way it does in the blend. proj is the output layout
    (None = equirectangular)."""
    if out_h is None:
        out_h = out_w // 2 if proj is None else output_size(proj, out_w)[1]
    top = _dtype_max(src.dtype)
    g = _channel_gain(gain, wb, src.shape[2])
    out = np.zeros((out_h, out_w, 4), dtype=src.dtype)
    tmax = float(np.degrees(lens.theta_max))
    soft = max(rim_soft_deg, 1e-3)
    for y0 in range(0, out_h, tile_rows):
        y1 = min(out_h, y0 + tile_rows)
        mx, my, th = build_maps(out_w, out_h, lens, view, y0, y1, proj)
        lay = cv2.remap(src, mx, my, INTERP[interp], borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        t = np.clip((tmax - th) / soft, 0.0, 1.0)
        alpha = t * t * (3.0 - 2.0 * t)
        rgb = np.clip(lay.astype(np.float32) * g, 0, top)
        out[y0:y1, :, :3] = np.rint(rgb).astype(src.dtype)
        out[y0:y1, :, 3] = np.rint(alpha * top).astype(src.dtype)
        if progress is not None:
            progress(y1 / out_h)
    return out


# ---------------------------------------------------------------- spherical photo metadata
def exiftool_path() -> str | None:
    return shutil.which("exiftool")


def add_spherical_metadata(path: str, width: int, height: int) -> tuple[bool, str]:
    """Mark the file as an equirectangular spherical photo (XMP-GPano) using exiftool if installed."""
    exe = exiftool_path()
    if exe is None:
        return False, "exiftool is not installed, so spherical photo metadata was not written."
    cmd = [exe, "-overwrite_original", "-XMP-GPano:ProjectionType=equirectangular",
           "-XMP-GPano:UsePanoramaViewer=True",
           f"-XMP-GPano:FullPanoWidthPixels={width}", f"-XMP-GPano:FullPanoHeightPixels={height}",
           f"-XMP-GPano:CroppedAreaImageWidthPixels={width}", f"-XMP-GPano:CroppedAreaImageHeightPixels={height}",
           "-XMP-GPano:CroppedAreaLeftPixels=0", "-XMP-GPano:CroppedAreaTopPixels=0", os.path.abspath(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except Exception as e:
        return False, f"exiftool failed: {e}"
    if r.returncode != 0:
        return False, f"exiftool reported an error: {r.stderr.strip() or r.stdout.strip()}"
    return True, "Spherical photo metadata written."


# ---------------------------------------------------------------- the whole export
def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.realpath(a) == os.path.realpath(b)


def export_panorama(path: str, src_a: np.ndarray, src_b: np.ndarray, lens_a: LensParams, lens_b: LensParams,
                    view: ViewParams, bp: BlendParams, settings: ExportSettings, icc: bytes | None = None,
                    protect: tuple = (), progress: Progress | None = None,
                    proj: Projection = EQUIRECT, mask=None) -> ExportResult:
    """Write the full-resolution image (and optionally the two layers). Never overwrites `protect`.

    proj picks the layout (equirectangular, Mercator or Gall). The height follows from the width and the
    layout's latitude range."""
    if settings.fmt not in FORMATS:
        raise ValueError(f"Unknown format: {settings.fmt}")
    if settings.width < 64 or settings.width % 2:
        raise ValueError("Export width must be an even number of at least 64")
    out_path = os.path.abspath(path)
    for p in protect:
        if p and os.path.exists(p) and os.path.exists(out_path) and _same_file(p, out_path):
            raise ValueError("Refusing to overwrite a source image. Choose a different file name.")
        if p and os.path.abspath(p) == out_path:
            raise ValueError("Refusing to overwrite a source image. Choose a different file name.")
    folder = os.path.dirname(out_path) or "."
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Output folder does not exist: {folder}")

    w, h = output_size(proj, settings.width)
    result = ExportResult()
    layers = settings.save_layers
    main_share = 0.6 if layers else 0.92

    def report(frac, text):
        if progress is not None:
            progress(min(max(frac, 0.0), 1.0), text)

    report(0.0, f"Building the {w} x {h} image...")
    pano = compose(src_a, src_b, lens_a, lens_b, view, bp, w, h, interp=settings.interp,
                   progress=lambda f: report(f * main_share, f"Building the {w} x {h} image..."), proj=proj,
                   mask=mask if settings.use_mask else None)
    want16 = settings.sixteen_bit and settings.fmt != "jpg"
    pano = _depth(pano, want16)
    report(main_share, "Writing the file...")
    write_image(out_path, pano, settings.fmt, icc=icc, jpg_quality=settings.jpg_quality)
    result.paths.append(out_path)
    del pano

    if settings.spherical_metadata:
        if proj.is_equirect:
            ok, msg = add_spherical_metadata(out_path, w, h)
            result.messages.append(msg)
        else:
            result.messages.append("Spherical photo metadata only applies to the equirectangular layout, "
                                   "so it was not written.")

    if layers:
        lay_fmt = "png" if settings.fmt == "jpg" else settings.fmt   # JPEG has no transparency
        if settings.fmt == "jpg":
            result.messages.append("Layers were saved as PNG because JPEG cannot hold transparency.")
        stem = os.path.splitext(out_path)[0]
        for i, (name, src, lens, gain, wb) in enumerate((("A", src_a, lens_a, bp.gain_a, bp.wb_a),
                                                          ("B", src_b, lens_b, bp.gain_b, bp.wb_b))):
            lo = main_share + i * (1 - main_share) / 2
            span = (1 - main_share) / 2
            rgba = render_layer_rgba(src, lens, view, gain, wb, w, h, interp=settings.interp,
                                     progress=lambda f, lo=lo, span=span, name=name:
                                     report(lo + f * span * 0.9, f"Building layer {name}..."), proj=proj)
            rgba = _depth(rgba, want16)
            lp = f"{stem}_layer{name}{EXTENSIONS[lay_fmt]}"
            write_image(lp, rgba, lay_fmt, icc=icc)
            result.paths.append(lp)
    report(1.0, "Done")
    return result
