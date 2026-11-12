import cv2
import numpy as np
import tifffile

from spherejoin.imageio_utils import load_image, to_display_uint8


def test_png8_roundtrip(tmp_path):
    rgb = np.random.randint(0, 256, (20, 30, 3), dtype=np.uint8)
    p = tmp_path / "a b.png"  # space in the name on purpose
    cv2.imwrite(str(p), rgb[:, :, ::-1])
    out = load_image(str(p))
    assert out.dtype == np.uint8 and out.shape == (20, 30, 3)
    assert np.array_equal(out, rgb)


def test_png16_stays_16bit(tmp_path):
    rgb = np.random.randint(0, 65536, (10, 12, 3), dtype=np.uint16)
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), rgb[:, :, ::-1])
    out = load_image(str(p))
    assert out.dtype == np.uint16
    assert np.array_equal(out, rgb)


def test_tiff16_stays_16bit(tmp_path):
    rgb = np.random.randint(0, 65536, (10, 12, 3), dtype=np.uint16)
    p = tmp_path / "a.tif"
    tifffile.imwrite(str(p), rgb)
    out = load_image(str(p))
    assert out.dtype == np.uint16 and np.array_equal(out, rgb)


def test_tiff_alpha_and_gray(tmp_path):
    rgba = np.random.randint(0, 256, (8, 8, 4), dtype=np.uint8)
    tifffile.imwrite(str(tmp_path / "rgba.tif"), rgba, photometric="rgb", extrasamples=[2])
    assert load_image(str(tmp_path / "rgba.tif")).shape == (8, 8, 3)
    gray = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "g.png"), gray)
    assert load_image(str(tmp_path / "g.png")).shape == (8, 8, 3)


def test_display_conversion():
    a = np.full((2, 2, 3), 65535, dtype=np.uint16)
    assert to_display_uint8(a).max() == 255
    assert to_display_uint8(a).dtype == np.uint8


def test_bad_extension(tmp_path):
    p = tmp_path / "x.gif"
    p.write_bytes(b"x")
    try:
        load_image(str(p))
    except ValueError:
        return
    raise AssertionError("expected ValueError")
