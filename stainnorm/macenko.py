import json
import random

import numpy as np
from PIL import Image, ImageCms

IO, ALPHA, BETA, PCT = 240, 1.0, 0.15, 99.0
SRGB = ImageCms.createProfile("sRGB")


def rgb_to_od(rgb, io=IO):
    return -np.log((np.asarray(rgb, dtype=np.float64).reshape(-1, 3) + 1) / io)


def tissue(od, beta=BETA):
    return od[~np.any(od < beta, axis=1)]


def stain_vectors(od, alpha=ALPHA):
    _, eig = np.linalg.eigh(np.cov(od.T))
    plane = eig[:, 1:3]
    proj = od @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    vlo = plane @ np.array([np.cos(lo), np.sin(lo)])
    vhi = plane @ np.array([np.cos(hi), np.sin(hi)])
    he = np.stack([vlo, vhi], 1) if vlo[0] > vhi[0] else np.stack([vhi, vlo], 1)
    return he * np.where(he.sum(0) < 0, -1.0, 1.0)


def amounts(od, he):
    return np.linalg.lstsq(he, od.T, rcond=None)[0]


def fit_od(od, alpha=ALPHA, beta=BETA, pct=PCT):
    t = tissue(od, beta)
    he = stain_vectors(t, alpha)
    return he, np.percentile(amounts(t, he), pct, axis=1)


def fit_tiles(tiles):
    return fit_od(rgb_to_od(np.vstack([t.reshape(-1, 3) for t in tiles])))


def apply(rgb, src_he, src_max, ref_he, ref_max, io=IO, beta=BETA):
    rgb = np.asarray(rgb)
    od = rgb_to_od(rgb, io)
    c = amounts(od, src_he) * (np.asarray(ref_max) / np.asarray(src_max))[:, None]
    w = np.clip(od.max(1) / beta, 0, 1)
    out = w * io * np.exp(-np.asarray(ref_he) @ c) + (1 - w) * rgb.reshape(-1, 3).T
    return np.clip(out, 0, 255).T.reshape(rgb.shape).astype(np.uint8)


def load_reference(path, setting):
    full = json.load(open(path))
    ref = {**full[{"stored": "before", "icc": "after"}[setting]], "amount_pcts": full["amount_pcts"], "rgb_pcts": full["rgb_pcts"]}
    return np.array(ref["median"]["he"]), np.array(ref["median"]["max"]), ref


def icc_transform(slide):
    prof = slide.color_profile
    if prof is None:
        return None, ""
    name = (prof.profile.profile_description or "").strip()[:80]
    return ImageCms.buildTransform(prof, SRGB, "RGB", "RGB", ImageCms.getDefaultIntent(prof), 0), name


def slide_mpp(slide, default=0.25):
    try:
        return float(slide.properties["openslide.mpp-x"])
    except (KeyError, ValueError):
        return default


def tile_grid(slide, mpp=0.5, tile=512, n=30, min_tissue=0.5, seed=42, thumb=2000, beta=BETA):
    w0, h0 = slide.dimensions
    size0 = int(round(tile * mpp / slide_mpp(slide)))
    th = np.asarray(slide.get_thumbnail((thumb, thumb)).convert("RGB"))
    f = th.shape[1] / w0
    mask = ~np.any(rgb_to_od(th).reshape(th.shape) < beta, axis=2)
    step = max(1, int(size0 * f))
    cells = [(x, y) for y in range(0, h0 - size0 + 1, size0) for x in range(0, w0 - size0 + 1, size0)
             if mask[int(y * f):int(y * f) + step, int(x * f):int(x * f) + step].mean() >= min_tissue]
    random.Random(seed).shuffle(cells)
    return [(x, y, size0) for x, y in cells[:n]], tile


def read_tile(slide, x, y, size0, tile, icc=None):
    lvl = slide.get_best_level_for_downsample(size0 / tile)
    d = slide.level_downsamples[lvl]
    img = slide.read_region((x, y), lvl, (int(round(size0 / d)),) * 2).convert("RGB")
    if img.size != (tile, tile):
        img = img.resize((tile, tile), Image.BOX)
    if icc is not None:
        img = ImageCms.applyTransform(img, icc)
    return np.asarray(img)
