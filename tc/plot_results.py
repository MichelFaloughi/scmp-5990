"""Headline figure: max-wind MAE by lead time, raw AIFS vs GBM-corrected vs NHC OFCL.

Reads the first table of data/tc/RESULTS_tables.md (written by evaluate.py).
Usage: python tc/plot_results.py   -> out/tc_mae_by_lead.png
"""
import os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {"AIFS raw": "#2a78d6", "AIFS+GBM": "#eb6834", "NHC OFCL": "#1baf7a"}

rows = []
for line in open(os.path.join(ROOT, "data", "tc", "RESULTS_tables.md")):
    m = re.match(r"\|\s*(\d+)\s*\|\s*\d+\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
    if m:
        rows.append([float(x) for x in m.groups()])
    elif rows and line.startswith("| 12-168"):
        break
leads = [r[0] for r in rows]

fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for s in ax.spines.values():
    s.set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=9)
ax.grid(color=GRID, linewidth=0.5)
for i, (name, color) in enumerate(SERIES.items()):
    y = [r[i + 1] for r in rows]
    ax.plot(leads, y, color=color, linewidth=2, marker="o", markersize=5, label=name)
    ax.annotate(name, (leads[-1], y[-1]), textcoords="offset points", xytext=(8, -3),
                fontsize=9, color=INK2)
ax.set_xlim(0, 200)
ax.set_ylim(0, None)
ax.set_xticks(leads)
ax.set_xlabel("forecast lead time (h)", color=INK2)
ax.set_ylabel("max-wind MAE (kt)", color=INK2)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9, facecolor=SURFACE,
          edgecolor=GRID, labelcolor=INK2)
ax.set_title("Intensity error by lead time, homogeneous sample", fontsize=11, color=INK, loc="left")
out = os.path.join(ROOT, "out", "tc_mae_by_lead.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
print("wrote", out)
