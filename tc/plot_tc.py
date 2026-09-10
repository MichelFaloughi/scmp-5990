"""Visualize the Mini AIFS-TC data.

Usage:
  python tc/plot_tc.py map 2026090100 48          # wind speed + MSLP from data/aifs/2026090100.npz at +48h
  python tc/plot_tc.py intensity LOWELL           # vmax vs time: best track + every AIFS forecast
Outputs land in out/.
"""
import os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"                      # categorical slots 1-2
SEQ = LinearSegmentedColormap.from_list("seq_blue", [   # sequential one-hue ramp, light->dark
    "#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.5)


def coastlines(ax):
    """Land outline from the model's land-sea mask (reduced grid, hence tricontour)."""
    try:
        import earthkit.data as ekd
        f = ekd.from_source("file", os.path.join(ROOT, "model", "lsm.grib"))[0]
        lats, lons = f.grid_points()
        lons = np.where(lons > 180, lons - 360, lons)
        keep = (lats > -5) & (lats < 65) & (lons > -185) & (lons < 5)
        import matplotlib.tri as tri
        t = tri.Triangulation(lons[keep], lats[keep])
        ax.tricontour(t, f.to_numpy(flatten=True)[keep], levels=[0.5],
                      colors=INK2, linewidths=0.5, alpha=0.6)
    except Exception as e:
        print(f"(no coastlines: {e})")


def plot_map(init, step):
    d = np.load(os.path.join(ROOT, "data", "aifs", f"{init}.npz"))
    i = int(np.where(d["steps"] == step)[0][0])
    wspd = np.hypot(d["u10"][i].astype(np.float32), d["v10"][i].astype(np.float32))
    msl = d["msl"][i] / 100.0
    lats, lons = d["lats"], d["lons"]

    tr = pd.read_csv(os.path.join(ROOT, "data", "tc", "aifs_tracks.csv"), parse_dates=["init", "valid"])
    tr = tr[tr.init == pd.Timestamp(f"{init[:8]} {init[8:]}:00")]

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
    style(ax)
    pm = ax.pcolormesh(lons, lats, wspd, cmap=SEQ, vmin=0,
                       vmax=max(20, float(np.ceil(wspd.max() / 5) * 5)), shading="auto")
    cs = ax.contour(lons, lats, msl, levels=np.arange(900, 1050, 4), colors=MUTED, linewidths=0.5)
    ax.clabel(cs, levels=cs.levels[::2], fontsize=7, colors=MUTED, fmt="%d")
    coastlines(ax)

    for sid, g in tr.groupby("sid"):
        g = g.sort_values("step")
        ax.plot(g.bt_lon, g.bt_lat, color=INK, linewidth=1.5, linestyle="--", label="Best track")
        ax.plot(g.lon, g.lat, color=ORANGE, linewidth=2, marker="o", markersize=3, label="AIFS track")
        here = g[g.step == step]
        if len(here):
            ax.plot(here.lon, here.lat, "o", color=ORANGE, markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(here["name"].iloc[0], (here.lon.iloc[0], here.lat.iloc[0]), textcoords="offset points",
                        xytext=(8, 8), fontsize=9, color=INK, fontweight="bold")
    if len(tr):
        h, l = ax.get_legend_handles_labels()
        by = dict(zip(l, h))
        ax.legend(by.values(), by.keys(), loc="upper left", fontsize=9,
                  framealpha=0.9, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK2)
        ax.set_xlim(tr.lon.min() - 12, tr.lon.max() + 12)
        ax.set_ylim(max(0, tr.lat.min() - 8), min(60, tr.lat.max() + 8))
    fig.colorbar(pm, ax=ax, shrink=0.85, pad=0.02, label="10 m wind speed (m/s)")
    ax.set_title(f"AIFS-Single  init {init[:8]} {init[8:]}Z  +{step}h   "
                 f"MSLP contours every 4 hPa", fontsize=11, color=INK, loc="left")
    out = os.path.join(ROOT, "out", f"tc_map_{init}_{step:03d}h.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", out)


def plot_intensity(name):
    tr = pd.read_csv(os.path.join(ROOT, "data", "tc", "aifs_tracks.csv"), parse_dates=["init", "valid"])
    tr = tr[tr["name"].str.upper() == name.upper()]
    if tr.empty:
        sys.exit(f"no tracks for {name}; names: {sorted(tr['name'].unique()) or 'run track.py first'}")
    bt = tr.drop_duplicates("valid").sort_values("valid")

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    style(ax)
    for j, (_, g) in enumerate(tr.groupby("init")):
        g = g.sort_values("valid")
        ax.plot(g.valid, g.vmax_kt, color=BLUE, alpha=0.45, linewidth=1.2,
                label="AIFS forecasts" if j == 0 else None)
    ax.plot(bt.valid, bt.bt_wind_kt, color=INK, linewidth=2.5, label="Best track")
    ymax = max(tr.vmax_kt.max(), bt.bt_wind_kt.max()) + 15
    for kt, lab in [(34, "TS"), (64, "Cat 1"), (96, "Cat 3"), (137, "Cat 5")]:
        if kt < ymax - 5:
            ax.axhline(kt, color=GRID, linewidth=0.8)
            ax.annotate(lab, (0.002, kt), xycoords=("axes fraction", "data"),
                        fontsize=7, color=MUTED, va="bottom")
    ax.set_ylim(0, ymax)
    ax.set_ylabel("vmax (kt)", color=INK2)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, facecolor=SURFACE,
              edgecolor=GRID, labelcolor=INK2)
    sid = tr.sid.iloc[0]
    ax.set_title(f"{name.upper()} ({sid})   best-track intensity vs raw AIFS forecasts",
                 fontsize=11, color=INK, loc="left")
    fig.autofmt_xdate()
    out = os.path.join(ROOT, "out", f"tc_intensity_{name.lower()}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "map":
        plot_map(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 48)
    elif len(sys.argv) >= 3 and sys.argv[1] == "intensity":
        plot_intensity(sys.argv[2])
    else:
        sys.exit(__doc__)
