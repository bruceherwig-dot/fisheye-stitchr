import io
import json
import os

import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from spherejoin.blend import BlendParams
from spherejoin.export import (
    ExportCancelled, ExportSettings, add_spherical_metadata, default_output_path, estimate_memory_mb,
    exiftool_path, export_panorama, inject_icc_png, read_icc, recommended_width, render_layer_rgba, write_image,
)
from spherejoin.projection import LensParams, ViewParams
from spherejoin.synth import make_test_equirect, render_fisheye_from_equirect

KW = dict(cx=450.0, cy=450.0, radius=420.0, fov_deg=220.0)


@pytest.fixture(scope="module")
def world():
    equi = make_test_equirect(1024, 512)
    A, B = LensParams(base_yaw=0.0, **KW), LensParams(base_yaw=180.0, **KW)
    return (equi, A, B, render_fisheye_from_equirect(equi, A, (900, 900)),
            render_fisheye_from_equirect(equi, B, (900, 900)))


def srgb_profile() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def test_recommended_width_matches_center_detail(world):
    equi, A, B, fa, fb = world
    # Equidistant, radius 420 at 110 degrees: f = 420/1.9199 px/rad, width = 2 pi f.
    expect = 2 * np.pi * 420.0 / np.radians(110)
    assert abs(recommended_width(A, B) - expect) <= 16
    assert recommended_width(A, B) % 16 == 0
    big = LensParams(cx=0, cy=0, radius=1994.0, fov_deg=220.0)
    assert 6400 <= recommended_width(big, big) <= 6700


def test_default_name_sits_next_to_image_a_and_follows_format():
    p = default_output_path("/x/y/photo.jpg", "tiff")
    assert p == "/x/y/photo_spherejoin_equirect.tif"
    assert default_output_path("/x/y/photo.jpg", "png").endswith(".png")
    assert estimate_memory_mb(8192, 2, False) < estimate_memory_mb(8192, 2, True)


@pytest.mark.parametrize("fmt,suffix", [("tiff", ".tif"), ("png", ".png"), ("jpg", ".jpg")])
def test_export_each_format_matches_expected_size(world, tmp_path, fmt, suffix):
    equi, A, B, fa, fb = world
    out = tmp_path / f"pano{suffix}"
    res = export_panorama(str(out), fa, fb, A, B, ViewParams(), BlendParams(),
                          ExportSettings(width=1024, fmt=fmt))
    assert res.paths == [str(out)]
    if fmt == "tiff":
        back = tifffile.imread(str(out))
    else:
        back = np.array(Image.open(out).convert("RGB"))
    assert back.shape == (512, 1024, 3) and back.dtype == np.uint8
    tol = 4.0 if fmt == "jpg" else 2.5
    assert np.abs(back.astype(int) - equi.astype(int))[8:-8].mean() < tol


def test_16bit_source_stays_16bit_in_tiff_and_png(world, tmp_path):
    equi, A, B, fa, fb = world
    fa16, fb16 = fa.astype(np.uint16) * 257, fb.astype(np.uint16) * 257
    for fmt in ("tiff", "png"):
        out = tmp_path / f"p16.{ 'tif' if fmt == 'tiff' else 'png'}"
        export_panorama(str(out), fa16, fb16, A, B, ViewParams(), BlendParams(), ExportSettings(width=512, fmt=fmt))
        back = tifffile.imread(str(out)) if fmt == "tiff" else cv2.imread(str(out), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
        assert back.dtype == np.uint16 and back.max() > 255 * 200


def test_save_8bit_source_as_16bit_scales_correctly(world, tmp_path):
    equi, A, B, fa, fb = world
    out8, out16 = tmp_path / "e8.tif", tmp_path / "e16.tif"
    export_panorama(str(out8), fa, fb, A, B, ViewParams(), BlendParams(), ExportSettings(width=512, fmt="tiff"))
    export_panorama(str(out16), fa, fb, A, B, ViewParams(), BlendParams(), ExportSettings(width=512, fmt="tiff", sixteen_bit=True))
    a8, a16 = tifffile.imread(str(out8)), tifffile.imread(str(out16))
    assert a16.dtype == np.uint16 and np.array_equal(a16, a8.astype(np.uint16) * 257)


def test_jpg_from_16bit_source_is_8bit(world, tmp_path):
    equi, A, B, fa, fb = world
    out = tmp_path / "p.jpg"
    export_panorama(str(out), fa.astype(np.uint16) * 257, fb.astype(np.uint16) * 257, A, B, ViewParams(),
                    BlendParams(), ExportSettings(width=512, fmt="jpg", sixteen_bit=True))
    assert Image.open(out).mode == "RGB"


def test_never_overwrites_a_source_image(world, tmp_path):
    equi, A, B, fa, fb = world
    src = tmp_path / "source.png"
    cv2.imwrite(str(src), fa[:, :, ::-1])
    before = src.read_bytes()
    with pytest.raises(ValueError):
        export_panorama(str(src), fa, fb, A, B, ViewParams(), BlendParams(), ExportSettings(width=256, fmt="png"),
                        protect=(str(src),))
    assert src.read_bytes() == before
    with pytest.raises(FileNotFoundError):
        export_panorama(str(tmp_path / "nope" / "x.tif"), fa, fb, A, B, ViewParams(), BlendParams(), ExportSettings(width=256))
    with pytest.raises(ValueError):
        export_panorama(str(tmp_path / "x.tif"), fa, fb, A, B, ViewParams(), BlendParams(), ExportSettings(width=255))


def test_layers_have_transparency_and_gain(world, tmp_path):
    equi, A, B, fa, fb = world
    out = tmp_path / "pano.tif"
    res = export_panorama(str(out), fa, fb, A, B, ViewParams(), BlendParams(gain_b=0.5),
                          ExportSettings(width=1024, fmt="tiff", save_layers=True))
    assert len(res.paths) == 3 and res.paths[1].endswith("_layerA.tif") and res.paths[2].endswith("_layerB.tif")
    la, lb = tifffile.imread(res.paths[1]), tifffile.imread(res.paths[2])
    assert la.shape == (512, 1024, 4)
    # A looks forward: opaque in the middle column, clear at the far back.
    assert la[256, 512, 3] == 255 and la[256, 0, 3] == 0
    assert lb[256, 0, 3] == 255 and lb[256, 512, 3] == 0
    # Gain on B shows up in the layer (center of B's region is darker than the raw warp).
    raw_b, _ = render_layer_rgba(fb, B, ViewParams(), 1.0, (1, 1, 1), 1024, 512)[..., :3], None
    assert lb[256, 0, :3].astype(int).sum() <= raw_b[256, 0].astype(int).sum() * 0.55 + 6
    # Soft rim: alpha falls off smoothly instead of a hard step.
    col = la[256, :, 3].astype(int)
    assert 0 < col[(col > 0) & (col < 255)].size


def test_layers_from_jpg_choice_become_png(world, tmp_path):
    equi, A, B, fa, fb = world
    out = tmp_path / "pano.jpg"
    res = export_panorama(str(out), fa, fb, A, B, ViewParams(), BlendParams(),
                          ExportSettings(width=512, fmt="jpg", save_layers=True))
    assert res.paths[1].endswith("_layerA.png")
    assert Image.open(res.paths[1]).mode == "RGBA"
    assert any("PNG" in m for m in res.messages)


def test_progress_is_monotonic_and_can_cancel(world, tmp_path):
    equi, A, B, fa, fb = world
    seen = []
    export_panorama(str(tmp_path / "a.tif"), fa, fb, A, B, ViewParams(), BlendParams(),
                    ExportSettings(width=512, save_layers=True), progress=lambda f, t: seen.append(f))
    assert seen[0] == 0.0 and seen[-1] == 1.0 and all(b >= a - 1e-9 for a, b in zip(seen, seen[1:]))

    def stop(f, t):
        if f > 0.2:
            raise ExportCancelled()
    with pytest.raises(ExportCancelled):
        export_panorama(str(tmp_path / "c.tif"), fa, fb, A, B, ViewParams(), BlendParams(),
                        ExportSettings(width=512), progress=stop)
    assert not (tmp_path / "c.tif").exists()      # nothing half-written


def test_icc_profile_is_carried_through_every_format(world, tmp_path):
    equi, A, B, fa, fb = world
    icc = srgb_profile()
    for fmt, ext in (("tiff", ".tif"), ("png", ".png"), ("jpg", ".jpg")):
        for sixteen in ((False, True) if fmt != "jpg" else (False,)):
            out = tmp_path / f"icc_{fmt}_{sixteen}{ext}"
            export_panorama(str(out), fa, fb, A, B, ViewParams(), BlendParams(),
                            ExportSettings(width=256, fmt=fmt, sixteen_bit=sixteen), icc=icc)
            if fmt == "tiff":
                with tifffile.TiffFile(str(out)) as tf:
                    got = tf.pages[0].tags[34675].value
                got = bytes(got)
            else:
                got = Image.open(out).info.get("icc_profile")
            assert got == icc, (fmt, sixteen)


def test_read_icc_and_png16_injection(tmp_path):
    icc = srgb_profile()
    p = tmp_path / "t.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(p, icc_profile=icc)
    assert read_icc(str(p)) == icc
    plain = tmp_path / "plain.jpg"
    Image.new("RGB", (8, 8)).save(plain)
    assert read_icc(str(plain)) is None and read_icc(None) is None and read_icc("/no/such") is None


def test_spherical_metadata_without_exiftool_is_graceful(world, tmp_path, monkeypatch):
    monkeypatch.setattr("spherejoin.export.shutil.which", lambda name: None)
    ok, msg = add_spherical_metadata(str(tmp_path / "x.jpg"), 1024, 512)
    assert not ok and "exiftool" in msg


@pytest.mark.skipif(exiftool_path() is None, reason="exiftool not installed")
def test_spherical_metadata_written(world, tmp_path):
    import subprocess
    equi, A, B, fa, fb = world
    out = tmp_path / "s.jpg"
    export_panorama(str(out), fa, fb, A, B, ViewParams(), BlendParams(),
                    ExportSettings(width=512, fmt="jpg", spherical_metadata=True))
    got = subprocess.run([exiftool_path(), "-s3", "-XMP-GPano:ProjectionType", str(out)], capture_output=True, text=True)
    assert got.stdout.strip() == "equirectangular"


def test_export_applies_the_painted_mask_only_when_asked(world, tmp_path):
    from spherejoin.seammask import SeamMask
    equi, A, B, fa, fb = world
    m = SeamMask()
    m.paint[:] = 1.0                                           # always image A wherever both see it
    bp = BlendParams(feather_deg=0.0, gain_b=0.5)
    kw = dict(src_a=fa, src_b=fb, lens_a=A, lens_b=B, view=ViewParams(), bp=bp)
    outs = {}
    for name, mask, use in (("auto", None, True), ("masked", m, True), ("ignored", m, False)):
        p = tmp_path / f"{name}.tif"
        export_panorama(str(p), **kw, settings=ExportSettings(width=1024, fmt="tiff", use_mask=use), mask=mask)
        outs[name] = tifffile.imread(str(p))
    assert np.array_equal(outs["auto"], outs["ignored"])
    assert not np.array_equal(outs["auto"], outs["masked"])
    x = int(1024 * (0.5 + 90 / 360))
    sl = (slice(236, 276), slice(x - 20, x + 20))
    assert outs["masked"][sl].mean() > outs["auto"][sl].mean()   # A (full brightness) replaced B (half) in the overlap
