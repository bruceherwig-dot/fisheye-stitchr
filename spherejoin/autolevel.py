"""Line up the two lenses from the star trails, then level the pair from the horizon and face the pole.

Star trails are circles around the celestial pole. That gives three things to measure:
  1. Relative alignment. Where both lenses see the same sky, the trails must run the same direction in both,
     whenever each picture was taken. We turn lens B (and fit the field of view) until the trail directions in
     the overlap agree as well as they can. This is what makes the sky join without a kink at the seam.
  2. Level. The horizon edge in one lens is rendered as a strip of panorama; if the camera was tilted it wobbles
     with longitude. A tilt applied to both lenses together (so their alignment is untouched) makes it run straight.
  3. Facing. Every trail direction is perpendicular to the pole axis, so the axis is the direction all the trails
     agree on. The output is turned to face the pole that lies below the horizon, so the other pole sits at the
     two edges of the panorama.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .preview import make_proxy, scale_lens
from .projection import LensParams, ViewParams, build_maps, rotation, source_to_direction

HORIZON_LON = 75.0          # degrees either side of the lens axis used to judge the horizon
HORIZON_BAND = 14.0         # degrees above and below the equator searched for the horizon edge
MAX_HORIZON_SCATTER = 0.5   # a horizon that wanders more than this (degrees) is too ragged to level by
MIN_POLE_RATIO = 3.0        # how clearly the trails agree on one axis, to trust it
HORIZON_STEP_WEIGHT = 0.15   # score lost per degree the horizon steps across the seam
GROUND_WEIGHT = 0.3         # how much the ground match counts against the trail match when lining up
LENS_MODELS = ("stereographic", "equidistant", "equisolid")   # fisheye projections tried, in this order
FOV_RANGE = (195.0, 250.0)  # field of view search limits, degrees


@dataclass
class AutoLevelResult:
    fov_deg: float | None     # fitted field of view; None when the alignment could not be measured
    model: str | None         # lens projection that fitted best (equidistant, equisolid or stereographic)
    trim_a: tuple | None      # yaw, pitch, roll for lens A and lens B; None when the alignment could not be measured
    trim_b: tuple | None
    view_yaw: float | None    # output view yaw that faces the pole below the horizon
    notes: list


def _gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return g * (255.0 / 65535.0) if img.dtype == np.uint16 else g


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def euler_trim(r: np.ndarray) -> tuple:
    """(yaw, pitch, roll) in degrees with rotation(yaw, pitch, roll) == r."""
    sa = float(np.clip(-r[1, 2], -1.0, 1.0))
    a = np.arcsin(sa)
    roll = np.arctan2(r[1, 0], r[1, 1])
    yaw = np.arctan2(r[0, 2], r[2, 2])
    return float(np.degrees(yaw)), float(-np.degrees(a)), float(np.degrees(roll))


# ---------------------------------------------------------------- the pole
def pole_axis(img: np.ndarray, lens: LensParams, max_dim: int = 1800):
    """(axis, ratio): the axis the star trails circle around, as a unit vector in the lens frame, and how clearly
    the trails agree on it. (None, 0.0) when there are not enough trails."""
    g = _gray(img)
    h, w = g.shape
    s = min(1.0, max_dim / max(h, w))
    if s < 1.0:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = g.shape
    cx, cy, r = lens.cx * s, lens.cy * s, lens.radius * s
    gs = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 3)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 3)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 3)
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (tr + 1e-6)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)               # gradient direction; a trail runs across it
    ys, xs = np.mgrid[0:h, 0:w]
    sky = (np.hypot(xs - cx, ys - cy) < r * 0.93) & (ys < cy - 0.07 * r)
    if sky.sum() < 1000:
        return None, 0.0
    sel = sky & (coh > 0.6) & (tr > np.percentile(tr[sky], 70))
    idx = np.flatnonzero(sel.ravel())
    if len(idx) < 1500:
        return None, 0.0
    idx = np.random.default_rng(1).choice(idx, size=min(40000, len(idx)), replace=False)
    py, px = np.divmod(idx, w)
    t_ = theta.ravel()[idx]
    tx, ty = -np.sin(t_), np.cos(t_)
    e = 2.0
    d1 = source_to_direction((px + e * tx) / s, (py + e * ty) / s, lens)
    d2 = source_to_direction((px - e * tx) / s, (py - e * ty) / s, lens)
    t = d1 - d2
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    _, sv, vt = np.linalg.svd(t, full_matrices=False)
    return vt[-1], float(sv[-2] / max(sv[-1], 1e-9))


def _pole_lon(lens: LensParams, trim: tuple, axis: np.ndarray) -> float:
    """World longitude of the pole that lies below the horizon, for this lens with these trims."""
    w = rotation(lens.base_yaw, 0.0, 0.0) @ rotation(*trim) @ axis
    if w[1] > 0:
        w = -w
    return float(np.degrees(np.arctan2(w[0], w[2])))


# ---------------------------------------------------------------- relative alignment from trail directions + ground
def _layer_fields(proxy: np.ndarray, lens: LensParams, width: int):
    """Warp one lens into a panorama strip and measure it: band-passed gray (for matching the ground), trail
    direction, how strongly directional it is, how strong the texture is, and the angle off the lens axis."""
    h = width // 2
    mx, my, th = build_maps(width, h, lens, ViewParams())
    layer = cv2.remap(proxy, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    gray = _gray(layer)
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 3)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 3)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 3)
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (tr + 1e-6)
    bp = cv2.GaussianBlur(gray, (0, 0), 1.2) - cv2.GaussianBlur(gray, (0, 0), 6.0)
    vert = cv2.Sobel(cv2.GaussianBlur(gray, (0, 0), 1.5), cv2.CV_32F, 0, 1, ksize=5)
    return bp, 0.5 * np.arctan2(2 * jxy, jxx - jyy), coh, tr, th, vert


def overlap_agreement(proxy_a, lens_a: LensParams, proxy_b, lens_b: LensParams, width: int = 1000):
    """(trails, ground, horizon step): how well the two lenses agree in the strip both see, each -1..1.
    Trails: the trail directions agree (true whenever each picture was taken). Ground: the ground and horizon
    match, which also pins down the turn about the pole axis that the symmetric trails cannot."""
    h = width // 2
    lat = (0.5 - (np.arange(h) + 0.5) / h) * 180.0
    ba, a1, c1, t1, th1, v1 = _layer_fields(proxy_a, lens_a, width)
    bb, a2, c2, t2, th2, v2 = _layer_fields(proxy_b, lens_b, width)
    ring = (th1 >= 72) & (th1 <= 104) & (th2 >= 72) & (th2 <= 104)
    sky = ring & (lat[:, None] > 6.0) & (c1 > 0.5) & (c2 > 0.5)
    ground = ring & (lat[:, None] >= -45.0) & (lat[:, None] <= 8.0)
    trails = 0.0
    if sky.sum() >= 500:
        w = np.sqrt(t1[sky] * t2[sky]) * np.sqrt(c1[sky] * c2[sky])
        trails = float((w * np.cos(2.0 * (a1[sky] - a2[sky]))).sum() / w.sum())
    gnd = 0.0
    if ground.sum() >= 500:
        x, y = ba[ground], bb[ground]
        gnd = float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-9))
    return trails, gnd, _horizon_mismatch(ring, v1, v2, lat)


def _horizon_mismatch(ring, v1, v2, lat) -> float:
    """How far apart (degrees of latitude) the two lenses put the horizon edge, column by column, in the strip both
    see. A step in the horizon across the seam is the first thing the eye catches."""
    rows = np.flatnonzero((lat >= -12.0) & (lat <= 14.0))
    diffs, strengths = [], []
    for x in range(0, ring.shape[1], 2):
        ok = rows[ring[rows, x]]
        if len(ok) < 8:
            continue
        i1, i2 = int(np.argmin(v1[ok, x])), int(np.argmin(v2[ok, x]))
        diffs.append(abs(lat[ok[i1]] - lat[ok[i2]]))
        strengths.append(max(v1[ok[i1], x], v2[ok[i2], x]))        # both negative: the weaker edge counts
    if len(diffs) < 15:
        return 0.0
    d, st = np.array(diffs), np.array(strengths)
    return float(np.median(d[st <= np.percentile(st, 50)]))


def trail_agreement(proxy_a, lens_a: LensParams, proxy_b, lens_b: LensParams, width: int = 1000) -> float:
    return overlap_agreement(proxy_a, lens_a, proxy_b, lens_b, width)[0]


def ground_agreement(proxy_a, lens_a: LensParams, proxy_b, lens_b: LensParams, width: int = 1000) -> float:
    return overlap_agreement(proxy_a, lens_a, proxy_b, lens_b, width)[1]


def _pattern_max(f, x0, steps, min_step):
    x = np.array(x0, float)
    st = np.array(steps, float)
    best = f(x)
    while st.max() > min_step:
        improved = False
        for i in range(len(x)):
            for sgn in (1, -1):
                y = x.copy()
                y[i] += sgn * st[i]
                v = f(y)
                if v > best + 1e-5:
                    x, best, improved = y, v, True
                    break
        if not improved:
            st = st / 2.0
    return x, best


# ---------------------------------------------------------------- the horizon
def horizon_tilt(proxy: np.ndarray, lens: LensParams, width: int = 1400):
    """Fit the horizon edge seen by one lens as lat = c + A cos(lon) + B sin(lon) around the lens axis.
    Returns (A, B, points, scatter): A and B are degrees of tilt that pitch and roll would remove (c, the horizon's
    own height, cannot be changed by rotating, so it is fitted and ignored); scatter is how far the edge points
    stray from the fitted curve, in degrees, which says how clean a horizon this is."""
    h = width // 2
    mx, my, th = build_maps(width, h, lens, ViewParams(yaw=lens.base_yaw))
    layer = cv2.remap(proxy, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    g = cv2.GaussianBlur(_gray(layer), (0, 0), 1.5)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
    lat_row = (0.5 - (np.arange(h) + 0.5) / h) * 180.0
    lon_col = ((np.arange(width) + 0.5) / width - 0.5) * 360.0
    rows = np.flatnonzero(np.abs(lat_row) <= HORIZON_BAND)
    pts = []
    for x in np.flatnonzero(np.abs(lon_col) <= HORIZON_LON)[::3]:
        ok = rows[th[rows, x] <= 80.0]
        if len(ok) < 10:
            continue
        col = gy[ok, x]
        j = int(np.argmin(col))                            # strongest bright-above, dark-below edge
        pts.append((lon_col[x], lat_row[ok[j]], col[j]))
    if len(pts) < 20:
        return 0.0, 0.0, 0, 99.0
    p = np.array(pts)
    p = p[p[:, 2] < np.percentile(p[:, 2], 55)]                 # keep the strongest edges
    lon, lat = np.radians(p[:, 0]), p[:, 1]
    a = np.c_[np.ones(len(lat)), np.cos(lon), np.sin(lon)]
    keep = np.ones(len(lat), bool)
    c = np.zeros(3)
    res = lat
    for _ in range(10):
        if keep.sum() < 10:
            return 0.0, 0.0, 0, 99.0
        c, *_ = np.linalg.lstsq(a[keep], lat[keep], rcond=None)
        res = lat - a @ c
        keep = np.abs(res) < max(2.5 * np.median(np.abs(res[keep])), 0.6)
    return float(c[1]), float(c[2]), int(keep.sum()), float(np.median(np.abs(res[keep])))


def _tilted(lens: LensParams, trim: tuple, g: np.ndarray) -> LensParams:
    """lens with the world rotation g applied on top of its trims."""
    base = rotation(lens.base_yaw, 0.0, 0.0)
    y, p, r = euler_trim(base.T @ g @ base @ rotation(*trim))
    return replace(lens, trim_yaw=y, trim_pitch=p, trim_roll=r)


def level_rotation(proxy: np.ndarray, lens: LensParams, trim: tuple):
    """World tilt (pitch, roll in degrees) that makes this lens's horizon run straight when applied to the whole
    pair, or None when its horizon is missing or too ragged to trust."""
    def meas(v):
        return np.array(horizon_tilt(proxy, _tilted(lens, trim, rotation(0.0, v[0], v[1])))[:2])

    x = np.zeros(2)
    first = horizon_tilt(proxy, _tilted(lens, trim, np.eye(3)))
    if first[2] < 20:
        return None
    m = meas(x)
    best_x, best_m = x.copy(), float(np.abs(m).max())
    for _ in range(6):
        if best_m < 0.2:
            break
        jac = np.zeros((2, 2))
        for k in range(2):
            y = x.copy()
            y[k] += 1.0
            jac[:, k] = meas(y) - m
        if abs(np.linalg.det(jac)) < 1e-6:
            break
        x = x - np.clip(np.linalg.solve(jac, m), -12.0, 12.0)
        m = meas(x)
        if float(np.abs(m).max()) < best_m:                  # near the answer the edge detection can jump: keep the best
            best_x, best_m = x.copy(), float(np.abs(m).max())
        else:
            x, m = best_x.copy(), meas(best_x)
    final = horizon_tilt(proxy, _tilted(lens, trim, rotation(0.0, best_x[0], best_x[1])))
    if final[2] < 20 or final[3] > MAX_HORIZON_SCATTER or best_m > 0.3:
        return None
    return float(best_x[0]), float(best_x[1])


# ---------------------------------------------------------------- putting it together
def auto_level(img_a, lens_a: LensParams, img_b, lens_b: LensParams) -> AutoLevelResult:
    """Pictures and lens descriptions (circles in the pictures' own pixels, field of view as a starting value).
    Trims in the lenses are ignored. Each common fisheye projection is tried and the one that makes the two lenses
    agree best is kept: a wrong projection shows up as a kink or a horizon step at one seam."""
    best = None
    for model in LENS_MODELS:
        res, score = _auto_level_for(img_a, replace(lens_a, model=model), img_b, replace(lens_b, model=model))
        if res.trim_a is None:
            return res                                            # no pole: the model cannot matter
        if best is None or score > best[1]:
            best = (res, score)
    res = best[0]
    res.notes.append(f"Lens projection: {res.model}.")
    return res


def _auto_level_for(img_a, lens_a: LensParams, img_b, lens_b: LensParams):
    """One projection model: (result, how well the two lenses agree)."""
    notes = []
    pa, sa = make_proxy(img_a, 2000)
    pb, sb = make_proxy(img_b, 2000)
    zero = dict(trim_yaw=0.0, trim_pitch=0.0, trim_roll=0.0)

    def scaled(lens, s, fov, trim=(0.0, 0.0, 0.0)):
        return scale_lens(replace(lens, fov_deg=float(fov), trim_yaw=trim[0], trim_pitch=trim[1], trim_roll=trim[2]), s)

    fov0 = float(np.clip(lens_a.fov_deg, *FOV_RANGE))
    axis_a, ratio_a = pole_axis(pa, scaled(lens_a, sa, fov0))
    axis_b, ratio_b = pole_axis(pb, scaled(lens_b, sb, fov0))
    if axis_a is None or axis_b is None or min(ratio_a, ratio_b) < MIN_POLE_RATIO:
        notes.append("No star-trail pole found in both pictures, so nothing was lined up. "
                     "This button needs star trails in both.")
        return AutoLevelResult(None, None, None, None, None, notes), -9.0

    # 1. relative alignment. Two starting guesses: (a) turn B the smallest amount that points its pole where A's
    #    points; (b) level each lens on its own horizon where it has a clean one, then turn B about the vertical so
    #    the poles agree. Keep the one that agrees better in the overlap, then refine on trails and ground together.
    base_a, base_b = rotation(lens_a.base_yaw, 0, 0), rotation(lens_b.base_yaw, 0, 0)
    wa, wb = base_a @ axis_a, base_b @ axis_b
    if wa @ wb < 0:
        wb = -wb
    v = np.cross(wb, wa)
    sinang, cosang = np.linalg.norm(v), float(np.clip(wb @ wa, -1.0, 1.0))
    q = np.eye(3)
    if sinang > 1e-9:
        k = v / sinang
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        q = np.eye(3) + sinang * kx + (1.0 - cosang) * (kx @ kx)       # Rodrigues: rotates wb onto wa
    cand_a = euler_trim(base_b.T @ q @ base_b)

    la0, lb0 = scaled(lens_a, sa, fov0), scaled(lens_b, sb, fov0)
    lv_a, lv_b = level_rotation(pa, la0, (0.0, 0.0, 0.0)), level_rotation(pb, lb0, (0.0, 0.0, 0.0))
    ga = rotation(0.0, *lv_a) if lv_a else np.eye(3)
    gb = rotation(0.0, *lv_b) if lv_b else np.eye(3)
    ra_c = base_a @ rotation(*euler_trim(base_a.T @ ga @ base_a))
    tb_level = euler_trim(base_b.T @ gb @ base_b)
    yaw_fix = _wrap(_pole_lon(la0, euler_trim(base_a.T @ ga @ base_a), axis_a)
                    - _pole_lon(lb0, tb_level, axis_b))
    rb_c = base_b @ rotation(tb_level[0] + yaw_fix, tb_level[1], tb_level[2])
    cand_b = euler_trim(base_b.T @ base_a @ ra_c.T @ rb_c) if lv_a or lv_b else cand_a

    def agree(x):
        fov = float(np.clip(x[0], *FOV_RANGE))
        la_, lb_ = scaled(lens_a, sa, fov), scaled(lens_b, sb, fov, (x[1], x[2], x[3]))
        trails, ground, step = overlap_agreement(pa, la_, pb, lb_)
        return trails + GROUND_WEIGHT * ground - HORIZON_STEP_WEIGHT * min(step, 6.0)

    starts = [[fov0, *cand_a], [fov0, *cand_b]]
    start = max(starts, key=agree)
    x, score = _pattern_max(agree, start, [4.0, 2.0, 2.0, 2.0], 0.05)
    fov = float(np.clip(x[0], *FOV_RANGE))
    rel_b = (float(x[1]), float(x[2]), float(x[3]))
    if score < 0.8 + GROUND_WEIGHT * 0.1:
        notes.append("The overlap agrees only loosely after lining up; check the seam by eye.")

    # 2. level the pair from the horizon. Each lens is leveled on its own first (that works from the start, where the
    #    tilt is still unknown); the tilt that does it, applied to the whole pair, keeps the alignment just found.
    zero = (0.0, 0.0, 0.0)
    lb = scaled(lens_b, sb, fov)
    la = scaled(lens_a, sa, fov)
    g = None
    lv_b = level_rotation(pb, lb, zero)
    if lv_b is not None:
        g = rotation(0.0, *lv_b) @ base_b @ rotation(*rel_b).T @ base_b.T
    else:
        lv_a = level_rotation(pa, la, zero)
        if lv_a is not None:
            g = rotation(0.0, *lv_a)
    if g is None:
        g = np.eye(3)
        notes.append("No clean horizon in either picture, so the pair was not leveled.")
    trim_a = euler_trim(base_a.T @ g @ base_a)
    trim_b = euler_trim(base_b.T @ g @ base_b @ rotation(*rel_b))

    # 3. face the pole below the horizon (the axis depends on the field of view, so measure it again)
    axis_a2, ratio_a2 = pole_axis(pa, scaled(lens_a, sa, fov))
    view_yaw = None
    if axis_a2 is not None and ratio_a2 >= MIN_POLE_RATIO:
        view_yaw = _wrap(_pole_lon(scaled(lens_a, sa, fov), trim_a, axis_a2))
    return AutoLevelResult(fov, lens_a.model, trim_a, trim_b, view_yaw, notes), score
