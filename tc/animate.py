"""Animate one AIFS run: 10 m wind speed + MSLP contours through all lead times,
with the tracked storms drawn up to the current step.

Usage: python tc/animate.py 2026090200 [out.gif]     -> assets/aifs_<init>.gif
"""
import os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_tc import SURFACE, INK, INK2, MUTED, GRID, ORANGE, SEQ, style, coastlines, degree_axes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

init = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "assets", f"aifs_{init}.gif")
os.makedirs(os.path.dirname(out), exist_ok=True)

d = np.load(os.path.join(ROOT, "data", "aifs", f"{init}.npz"))
steps, lats, lons = d["steps"], d["lats"], d["lons"]
wspd = np.hypot(d["u10"].astype(np.float32), d["v10"].astype(np.float32))
msl = d["msl"] / 100.0

tr = pd.read_csv(os.path.join(ROOT, "data", "tc", "aifs_tracks.csv"), parse_dates=["init", "valid"])
tr = tr[tr.init == pd.Timestamp(f"{init[:8]} {init[8:]}:00")]

fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
style(ax)
vmax = float(np.ceil(wspd.max() / 5) * 5)
pm = ax.pcolormesh(lons, lats, wspd[0], cmap=SEQ, vmin=0, vmax=vmax, shading="auto")
coastlines(ax)
degree_axes(ax)
if len(tr):
    ax.set_xlim(tr.lon.min() - 12, tr.lon.max() + 12)
    ax.set_ylim(max(0, tr.lat.min() - 8), min(60, tr.lat.max() + 8))
fig.colorbar(pm, ax=ax, shrink=0.85, pad=0.02, label="10 m wind (m/s)")
title = ax.set_title("", fontsize=10, color=INK, loc="left")
contours, artists = None, []


def frame(i):
    global contours
    if contours is not None:
        contours.remove()
    for a in artists:
        a.remove()
    artists.clear()
    pm.set_array(wspd[i].ravel())
    contours = ax.contour(lons, lats, msl[i], levels=np.arange(900, 1050, 4),
                          colors=MUTED, linewidths=0.4)
    for sid, g in tr[tr.step <= steps[i]].groupby("sid"):
        g = g.sort_values("step")
        artists.extend(ax.plot(g.lon, g.lat, color=ORANGE, linewidth=1.5))
        artists.extend(ax.plot(g.lon.iloc[-1], g.lat.iloc[-1], "o", color=ORANGE,
                               markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5))
        artists.append(ax.annotate(g["name"].iloc[0], (g.lon.iloc[-1], g.lat.iloc[-1]),
                                   textcoords="offset points", xytext=(7, 7),
                                   fontsize=8, color=INK, fontweight="bold"))
    title.set_text(f"AIFS-Single  init {init[:8]} {init[8:]}Z  +{int(steps[i])}h")
    return [pm]


anim = FuncAnimation(fig, frame, frames=len(steps))
anim.save(out, writer=PillowWriter(fps=5), dpi=90)
print("wrote", out, f"({os.path.getsize(out) / 1e6:.1f} MB)")
