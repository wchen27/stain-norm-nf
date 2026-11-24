#!/usr/bin/env python3
import csv
import glob
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIRED = [
    ("clipped_pct", "% tissue px with a channel <= 2"),
    ("h_p99", "H 99th percentile"),
    ("e_p99", "E 99th percentile"),
    ("rgb_dist_orig", "distance to TCGA colour, original"),
    ("rgb_dist_norm", "distance to TCGA colour, normalised"),
    ("de_norm_vs_orig", "ΔE normalised vs original"),
    ("ssim_norm_vs_orig", "SSIM normalised vs original"),
]
SINGLE = [
    ("de_stored_vs_icc", "ΔE stored vs icc, original"),
    ("de_norm_stored_vs_icc", "ΔE stored vs icc, both normalised"),
]
COLOURS = {"stored": "#2a78d6", "icc": "#eb6834"}


def values(rows, key):
    return np.array([float(r[key]) for r in rows])


def style(ax, title):
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def plot(both, path):
    fig, axes = plt.subplots(2, 5, figsize=(13, 6.8))
    axes = axes.ravel()
    axes[-1].axis("off")
    for ax, (k, title) in zip(axes, PAIRED):
        a, b = values(both, f"{k}_stored"), values(both, f"{k}_icc")
        ax.plot([0, 1], [a, b], color="#c8c7c0", lw=0.8, zorder=1)
        for x, v, s in ((0, a, "stored"), (1, b, "icc")):
            ax.scatter(np.full(len(v), x), v, s=24, color=COLOURS[s], edgecolor="white", lw=0.8, zorder=3)
            ax.plot([x - 0.25, x + 0.25], [np.median(v)] * 2, color="#222", lw=2, zorder=4)
        ax.set_xticks([0, 1], ["stored", "icc"])
        ax.set_xlim(-0.5, 1.5)
        style(ax, title.replace(", ", ",\n"))
    for ax, (k, title) in zip(axes[len(PAIRED):], SINGLE):
        v = values(both, k)
        ax.scatter(np.random.default_rng(0).uniform(-0.1, 0.1, len(v)), v, s=24, color="#888780", edgecolor="white", lw=0.8, zorder=3)
        ax.plot([-0.25, 0.25], [np.median(v)] * 2, color="#222", lw=2, zorder=4)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(bottom=0)
        style(ax, title.replace(", ", ",\n"))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    rows = []
    for p in sorted(glob.glob("tsv/*.tsv")):
        rows += list(csv.DictReader(open(p), delimiter="\t"))
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open("stain_norm_results.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", restval="")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if not r["error"]]
    both = [r for r in ok if r.get("h_r_icc")]
    out = [f"slides: {len(rows)}, fitted: {len(ok)}, with an icc profile: {len(both)}"]
    out += [f"  {e}: {n}" for e, n in Counter(r["error"] for r in rows if r["error"]).most_common()]
    out += [f"icc profiles: {dict(Counter(r['icc_profile'] or 'none' for r in ok))}", ""]
    if both:
        out.append(f"medians over the {len(both)} slides with an icc profile, stored / icc (slides where icc is lower)")
        for k, title in PAIRED:
            a, b = values(both, f"{k}_stored"), values(both, f"{k}_icc")
            out.append(f"  {title}: {np.median(a):.3f} / {np.median(b):.3f} ({int((b < a).sum())} of {len(both)})")
        for k, title in SINGLE:
            out.append(f"  {title}: {np.median(values(both, k)):.3f}")
        pcts = [int(k[9:-7]) for k in both[0] if k.startswith("h_ratio_p") and k.endswith("_stored")]
        for s in ("stored", "icc"):
            out.append(f"  reference / slide amount at p{pcts}, {s}: " + ", ".join(
                f"{c} {[round(float(np.median(values(both, f'{c}_ratio_p{p}_{s}'))), 2) for p in pcts]}" for c in "he"))
        plot(both, "stain_norm_summary.png")
    text = "\n".join(out)
    print(text)
    open("stain_norm_summary.txt", "w").write(text + "\n")

    html = ["<html><body style='font-family:sans-serif'><h2>Macenko normalisation to the TCGA median reference</h2>",
            f"<pre>{text}</pre>"]
    if both:
        html.append("<img src='stain_norm_summary.png' style='max-width:100%'>")
    html.append("<p>columns: stored, stored normalised, icc, icc normalised</p>")
    html += [f"<p>{r['name']}</p><img src='previews/{r['name']}.png' style='max-width:100%'>" for r in ok]
    open("index.html", "w").write("\n".join(html) + "</body></html>\n")


if __name__ == "__main__":
    main()
