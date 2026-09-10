"""Plot one field from an AIFS GRIB output.

Usage: python plot_aifs.py out/aifs_test.grib [param] [step_hours]
Defaults: param=2t, last step in the file.
"""
import sys

import earthkit.data as ekd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

path = sys.argv[1]
param = sys.argv[2] if len(sys.argv) > 2 else "2t"
step = int(sys.argv[3]) if len(sys.argv) > 3 else None

fs = ekd.from_source("file", path)
print("fields in file:", len(fs))
print("params:", sorted(set(fs.metadata("param"))))
print("steps:", sorted(set(fs.metadata("step"))))

sel = fs.sel(param=param)
if step is None:
    step = max(sel.metadata("step"))
f = sel.sel(step=step)[0]

lats, lons = f.grid_points()
lons = np.where(lons > 180, lons - 360, lons)   # 0..360 -> -180..180
vals = f.to_numpy(flatten=True)
if param == "2t":
    vals = vals - 273.15                          # K -> C

fig, ax = plt.subplots(figsize=(12, 6))
t = tri.Triangulation(lons, lats)
cs = ax.tricontourf(t, vals, levels=30, cmap="RdBu_r")
fig.colorbar(cs, ax=ax, shrink=0.8, label=f"{param}" + (" (°C)" if param == "2t" else ""))
ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.set_aspect("equal")
ax.set_title(f"AIFS-Single 2.0  {param}  base {f.metadata('date')} {f.metadata('time'):04d}Z  +{step}h")
out = f"out/{param}_step{step}.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("wrote", out)
