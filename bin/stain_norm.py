#!/usr/bin/env python3
import argparse
import csv
import json

import numpy as np
import openslide
from PIL import Image, ImageCms, ImageDraw, ImageFont

from stainnorm import macenko as m
from stainnorm import metrics as mt


def w1(px, tq, setting):
    return float(np.mean([np.abs(np.percentile(px[:, j], tq["q"]) - tq[setting]["quantiles"][c]).mean() for j, c in enumerate("rgb")]))


def measure(tiles, setting, opts):
    he, mx = m.fit_tiles(tiles)
    rhe, rmx, ref = m.load_reference(opts.reference, setting)
    px = np.vstack([t.reshape(-1, 3) for t in tiles])
    od = m.rgb_to_od(px)
    keep = ~np.any(od < m.BETA, axis=1)
    row = {"clipped_pct": 100 * float(np.mean(np.any(px[keep] <= 2, axis=1)))}
    for i, s in enumerate("he"):
        row.update({f"{s}_{c}": he[j, i] for j, c in enumerate("rgb")})
        row[f"{s}_p99"] = mx[i]
        amt = np.percentile(m.amounts(od[keep], he)[i], ref["amount_pcts"])
        row.update({f"{s}_ratio_p{p}": r / a for p, r, a in zip(ref["amount_pcts"], ref["amount_pcts_median"][i], amt)})

    sub = tiles[:opts.metric_tiles]
    norm = [m.apply(t, he, mx, rhe, rmx) for t in sub]
    masks = [mt.tissue_mask(t) for t in sub]
    tq = json.load(open(opts.quantiles))
    row["w1_orig"] = w1(np.vstack([t[k] for t, k in zip(sub, masks)]), tq, setting)
    row["w1_norm"] = w1(np.vstack([t[k] for t, k in zip(norm, masks)]), tq, setting)
    row["de_norm_vs_orig"] = float(np.median([mt.delta_e2000(t, n, k) for t, n, k in zip(sub, norm, masks)]))
    row["ssim_norm_vs_orig"] = float(np.median([mt.ssim(t, n) for t, n in zip(sub, norm)]))
    return {f"{k}_{setting}": v for k, v in row.items()}, (he, mx, rhe, rmx), norm


def label(img, text, px=None):
    im = Image.fromarray(img)
    if px:
        im = im.resize((px, px), Image.BOX)
    size = max(12, im.width // 40)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, size + 6], fill="white")
    d.text((4, 3), text, fill="black", font=ImageFont.load_default(size))
    return im


def grid_image(cells, path):
    w, h = max(c.width for r in cells for c in r) + 6, max(c.height for r in cells for c in r) + 6
    im = Image.new("RGB", (len(cells[0]) * w, len(cells) * h), "white")
    for i, r in enumerate(cells):
        for j, c in enumerate(r):
            im.paste(c, (j * w, i * h))
    im.save(path)


def run(opts):
    s = openslide.OpenSlide(opts.slide)
    icc, profile = m.icc_transform(s)
    grid, size = m.tile_grid(s, n=opts.tiles)
    if not grid:
        raise ValueError("no tissue tiles")
    tiles = {"stored": [m.read_tile(s, x, y, s0, size) for x, y, s0 in grid]}
    if icc is not None:
        tiles["icc"] = [np.asarray(ImageCms.applyTransform(Image.fromarray(t), icc)) for t in tiles["stored"]]

    row = {"name": opts.name, "error": "", "icc_profile": profile, "mpp": m.slide_mpp(s, float("nan")), "tiles": len(grid)}
    thumb = np.asarray(s.get_thumbnail((opts.thumb, opts.thumb)).convert("RGB"))
    thumbs = {"stored": thumb}
    if icc is not None:
        thumbs["icc"] = np.asarray(ImageCms.applyTransform(Image.fromarray(thumb), icc))
    fits, norm, slide_rows = {"name": opts.name, "icc_profile": profile, "reference": "tcga median"}, {}, []
    for k, tl in tiles.items():
        r, fit, norm[k] = measure(tl, k, opts)
        row.update(r)
        fits[k] = {"he": fit[0].round(6).tolist(), "max": fit[1].round(6).tolist()}
        slide_rows.append([label(thumbs[k], k), label(m.apply(thumbs[k], *fit), f"{k}, normalised")])
    cols = [("stored", tiles["stored"]), ("stored, normalised", norm["stored"])]
    if icc is not None:
        masks = [mt.tissue_mask(t) for t in tiles["stored"][:opts.metric_tiles]]
        row["de_stored_vs_icc"] = float(np.median([mt.delta_e2000(a, b, k) for a, b, k in zip(tiles["stored"], tiles["icc"], masks)]))
        row["de_norm_stored_vs_icc"] = float(np.median([mt.delta_e2000(a, b, k) for a, b, k in zip(norm["stored"], norm["icc"], masks)]))
        cols += [("icc", tiles["icc"]), ("icc, normalised", norm["icc"])]
    json.dump(fits, open(f"{opts.name}_fit.json", "w"), indent=1)
    grid_image(slide_rows, f"{opts.name}_slide.png")
    grid_image([[label(t[i], text, opts.px) for text, t in cols] for i in range(min(opts.show, len(norm["stored"])))],
               f"{opts.name}_tiles.png")
    return row


def main():
    ap = argparse.ArgumentParser(description="Slide level Macenko fit as stored and with icc, normalised to the TCGA median reference")
    ap.add_argument("--slide", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--quantiles", required=True)
    ap.add_argument("--tiles", type=int, default=30)
    ap.add_argument("--metric_tiles", type=int, default=10)
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--thumb", type=int, default=1200)
    ap.add_argument("--px", type=int, default=256)
    opts = ap.parse_args()
    try:
        row = run(opts)
    except Exception as ex:
        msg = f"{type(ex).__name__}: {ex}".replace(opts.slide, "<slide>").replace(opts.name, "<slide>")
        row = {"name": opts.name, "error": msg[:120]}
    with open(f"{opts.name}.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, list(row), delimiter="\t")
        w.writeheader()
        w.writerow({k: f"{v:.5g}" if isinstance(v, float) else v for k, v in row.items()})


if __name__ == "__main__":
    main()
