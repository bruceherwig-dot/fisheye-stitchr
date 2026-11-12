import numpy as np

from spherejoin.circle import auto_detect_circle


def _synthetic(cx, cy, r, w=1200, h=800, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 5, (h, w, 3)).astype(np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - cx, yy - cy)
    inside = d <= r
    shade = (30 + 90 * (1 - np.clip(d / r, 0, 1))).astype(np.uint8)
    for c in range(3):
        img[:, :, c][inside] = shade[inside]
    # Stars outside the disc.
    for _ in range(80):
        x, y = rng.integers(0, w), rng.integers(0, h)
        if np.hypot(x - cx, y - cy) > r + 10:
            img[y, x] = 255
    return img


def test_detects_known_disc():
    img = _synthetic(610.0, 395.0, 380.0)
    c = auto_detect_circle(img)
    assert c is not None
    assert abs(c.cx - 610.0) < 3 and abs(c.cy - 395.0) < 3 and abs(c.r - 380.0) < 4


def test_detects_off_center_disc():
    img = _synthetic(560.0, 420.0, 340.0)
    c = auto_detect_circle(img)
    assert c is not None
    assert abs(c.cx - 560.0) < 3 and abs(c.cy - 420.0) < 3 and abs(c.r - 340.0) < 4


def test_blank_image_returns_none():
    img = np.zeros((400, 600, 3), np.uint8)
    assert auto_detect_circle(img) is None


def test_16bit_input():
    img = _synthetic(610.0, 395.0, 380.0).astype(np.uint16) * 257
    c = auto_detect_circle(img)
    assert c is not None and abs(c.r - 380.0) < 4


def test_dark_disc_on_bright_gray_surround():
    """Star Trail CleanR's pictures: a darker disc on a light gray, noisy surround."""
    import cv2
    import numpy as np
    from spherejoin.circle import auto_detect_circle
    rng = np.random.default_rng(3)
    img = np.clip(rng.normal(175, 12, (1200, 1800, 3)), 0, 255).astype(np.uint8)
    cv2.circle(img, (900, 600), 520, (70, 60, 50), -1)
    c = auto_detect_circle(img)
    assert c is not None and abs(c.cx - 900) < 6 and abs(c.cy - 600) < 6 and abs(c.r - 520) < 8
