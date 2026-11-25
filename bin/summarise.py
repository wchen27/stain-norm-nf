#!/usr/bin/env python3
import csv
import glob
import json
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MEASURES = [
    ("w1_orig", "distance to TCGA colour, original", "{:.1f}"),
    ("w1_norm", "distance to TCGA colour, normalised", "{:.1f}"),
    ("de_norm_vs_orig", "ΔE normalised vs original", "{:.2f}"),
    ("ssim_norm_vs_orig", "SSIM normalised vs original", "{:.3f}"),
    ("h_p99", "H 99th percentile", "{:.2f}"),
    ("e_p99", "E 99th percentile", "{:.2f}"),
    ("clipped_pct", "% tissue px with a channel <= 2", "{:.2f}"),
]
GROUPS = [("tcga", "stored", "TCGA as stored", "#888780"), ("ge", "stored", "GE as stored", "#2a78d6"),
          ("tcga", "icc", "TCGA ICC", "#888780"), ("ge", "icc", "GE ICC", "#2a78d6")]
CSS = ("<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse;font-size:14px;margin:8px 0}"
       "td,th{padding:4px 12px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}"
       "th{border-bottom:2px solid #999}ul{max-width:900px;font-size:14px;color:#333}</style>")


def ge_values(rows, key, setting):
    return [float(r[f"{key}_{setting}"]) for r in rows if r.get(f"{key}_{setting}") not in (None, "")]


def values(data, src, key, setting):
    return data[src][setting].get(key, []) if src == "tcga" else ge_values(data["ge"], key, setting)


def median(v, fmt):
    return fmt.format(np.median(v)) if len(v) else "–"


def plot(data, path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes = axes.ravel()
    axes[-1].axis("off")
    rng = np.random.default_rng(0)
    for ax, (key, title, fmt) in zip(axes, MEASURES):
        for i, (src, s, _, col) in enumerate(GROUPS):
            v = values(data, src, key, s)
            if not v:
                continue
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=14 if src == "tcga" else 28, color=col,
                       alpha=0.6 if src == "tcga" else 0.9, edgecolor="white", linewidth=0.5, zorder=3)
            ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="#222", lw=2, zorder=4)
            ax.annotate(median(v, fmt), (i + 0.32, np.median(v)), va="center", fontsize=8, color="#222")
        ax.axvline(1.5, color="#ddd", lw=1)
        ax.set_xticks(range(len(GROUPS)), [g[2].replace(" ", "\n", 1) for g in GROUPS], fontsize=8)
        ax.set_xlim(-0.5, len(GROUPS) - 0.1)
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="y", color="#eee", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8, colors="#444")
    lo, hi = zip(*(ax.get_ylim() for ax in axes[:2]))
    for ax in axes[:2]:
        ax.set_ylim(0, max(hi))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def table(data):
    h = ["<table><tr><th></th>" + "".join(f"<th>{g[2]}</th>" for g in GROUPS) + "</tr>"]
    for key, title, fmt in MEASURES:
        h.append(f"<tr><td>{title}</td>" + "".join(f"<td>{median(values(data, src, key, s), fmt)}</td>" for src, s, _, _ in GROUPS) + "</tr>")
    for key, title in (("de_stored_vs_icc", "ΔE as stored vs ICC, original"), ("de_norm_stored_vs_icc", "ΔE as stored vs ICC, both normalised")):
        v = [float(r[key]) for r in data["ge"] if r.get(key)]
        h.append(f"<tr><td>{title}</td><td colspan=4 style='text-align:center'>GE {median(v, '{:.2f}')}</td></tr>")
    return "\n".join(h) + "</table>"


def main():
    baselines = json.load(open(sys.argv[1]))
    rows = []
    for p in sorted(glob.glob("tsv/*.tsv")):
        rows += list(csv.DictReader(open(p), delimiter="\t"))
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open("stain_norm_results.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", restval="")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if not r["error"]]
    scaling = ok[0]["scaling"] if ok else "p10-p90"
    baseline = baselines[scaling]
    data = {"ge": ok, "tcga": baseline}
    counts = (f"GE slides: {len(rows)}, fitted {len(ok)}, with an ICC profile {sum(1 for r in ok if r.get('h_r_icc'))}. "
              f"TCGA reference slides: {len(baseline['stored']['w1_orig'])} as stored, {len(baseline['icc']['w1_orig'])} with ICC")
    errors = [f"{e}: {n}" for e, n in Counter(r["error"] for r in rows if r["error"]).most_common()]
    lines = [counts] + errors + [""] + [f"{title}: " + ", ".join(f"{lab} {median(values(data, src, key, s), fmt)}" for src, s, lab, _ in GROUPS)
                                        for key, title, fmt in MEASURES]
    open("stain_norm_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if ok:
        plot(data, "stain_norm_summary.png")

    html = [f"<html><head>{CSS}</head><body><h2>Macenko normalisation to the TCGA median reference, {scaling} scaling</h2>", f"<p>{counts}</p>"]
    html += [f"<p>{e}</p>" for e in errors]
    html += [table(data),
             "<ul><li>each number: median over slides. GE: tissue pixels of 10 tiles per slide. TCGA: 50,000 of the tissue pixels kept "
             "from 30 tiles per reference slide, each normalised with its own fit</li>"
             "<li>distance to TCGA colour: Wasserstein distance per r g b channel to the pooled TCGA tissue pixels, mean over channels (0-255)</li>"
             "<li>ΔE: CIEDE2000, mean over tissue pixels. SSIM: grey image, 7 x 7 window, whole tile (needs whole tiles, not in the TCGA pixel cache)</li>"
             "<li>tissue pixels: od >= 0.15 in all channels</li></ul>"]
    if ok:
        html.append("<img src='stain_norm_summary.png' style='max-width:100%'>")
    html.append("<h3>slides</h3><p>whole slide as stored and with ICC, before and after normalising, then random tissue tiles "
                "(columns: stored, stored normalised, ICC, ICC normalised)</p>")
    for r in ok:
        html += [f"<h4>{r['name']}</h4>", f"<img src='previews/{r['name']}_slide.png' style='max-width:100%'>",
                 f"<img src='previews/{r['name']}_tiles.png' style='max-width:100%'>"]
    open("index.html", "w").write("\n".join(html) + "</body></html>\n")


if __name__ == "__main__":
    main()
