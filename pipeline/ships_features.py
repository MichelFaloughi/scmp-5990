"""Compute SHIPS-style environmental predictors from the AIFS pressure-level fields.

Input:  data/aifs_pl/{YYYYMMDDHH}.npz (from fetch_aifs_pl.py), data/tc/aifs_tracks_eta.csv (cases),
        data/tc/besttrack.csv (storm position at the init time, for the step-0 environment).
Output: data/tc/ships_features.csv, one row per case (init, sid, step), joinable on those keys.

Predictors, each area-averaged around the forecast storm centre (ring 200 to 800 km unless noted):
  shear_kt   magnitude of the ring-mean 200 minus 850 hPa wind vector
  shdir_deg  direction of that vector (meteorological, degrees)
  rh700, rh500, rhmid  relative humidity (%) at 700, 500 hPa and their mean
  t200_c     200 hPa temperature (deg C)
  div200     200 hPa divergence within 1000 km (1e-7 /s)
  vort850    850 hPa relative vorticity within 1000 km (1e-7 /s)
  u200_kt    ring-mean 200 hPa zonal wind (kt)
  sst_c      skin temperature within 200 km (deg C; includes land where present)
  mpi_kt     DeMaria and Kaplan (1994) maximum potential intensity from sst_c
  pot_kt     mpi_kt minus the current AIFS vmax at that step

Three versions per predictor: at the forecast step (suffix none), at the init (suffix _0, from the
step-0 field at the best-track position), and the mean along the forecast track from 0 to step
(suffix _m). SHIPS uses the along-track mean for most predictors.

Usage: python pipeline/ships_features.py [--limit N]
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PL = os.path.join(ROOT, "data", "aifs_pl")
TC = os.path.join(ROOT, "data", "tc")
R = 6371.0
MS2KT = 1.943844
WIN = 12.0            # degrees half-width of the working window around the centre

def haversine_km(lat0, lon0, lat, lon):
    p0, p = np.radians(lat0), np.radians(lat)
    dl = np.radians(lon - lon0)
    a = np.sin((p - p0) / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def rh(q, t, p_hpa):
    es = 6.112 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    qs = 0.622 * es / (p_hpa - 0.378 * es)
    return np.clip(100.0 * q / qs, 0, 100)

def mpi_dk94(sst_c):
    return 38.21 + 170.72 * np.exp(0.1909 * (sst_c - 30.0))

class Fields:
    def __init__(self, path):
        d = np.load(path)
        self.lats, self.lons, self.steps = d["lats"], d["lons"], list(d["steps"])
        self.f = {k: d[k] for k in d.files if k not in ("lats", "lons", "steps")}
        # grid spacing in metres for derivatives (lat descending in the files)
        self.dy = np.gradient(np.radians(self.lats)) * R * 1000.0
        self.dx = np.radians(np.gradient(self.lons))[None, :] * R * 1000.0 * np.cos(np.radians(self.lats))[:, None]

    def window(self, lat0, lon0):
        ia = np.where(np.abs(self.lats - lat0) <= WIN)[0]
        io = np.where(np.abs(self.lons - lon0) <= WIN)[0]
        return ia, io

    def predictors(self, step, lat0, lon0, vmax_kt):
        k = self.steps.index(step)
        ia, io = self.window(lat0, lon0)
        if len(ia) < 10 or len(io) < 10:
            return None
        LA, LO = np.meshgrid(self.lats[ia], self.lons[io], indexing="ij")
        dist = haversine_km(lat0, lon0, LA, LO)
        ring = (dist >= 200) & (dist <= 800); core = dist <= 200; wide = dist <= 1000
        g = lambda name: self.f[name][k][np.ix_(ia, io)]
        u850, v850, u200, v200 = g("u850"), g("v850"), g("u200"), g("v200")
        du, dv = u200[ring].mean() - u850[ring].mean(), v200[ring].mean() - v850[ring].mean()
        shear = np.hypot(du, dv) * MS2KT
        shdir = (np.degrees(np.arctan2(-du, -dv)) + 360) % 360
        rh7, rh5 = rh(g("q700"), g("t700"), 700.0)[ring].mean(), rh(g("q500"), g("t500"), 500.0)[ring].mean()
        dy, dx = self.dy[ia][:, None], self.dx[np.ix_(ia, io)]
        dudx = np.gradient(u200, axis=1) / dx; dvdy = np.gradient(v200, axis=0) / dy
        div = (dudx + dvdy)[wide].mean() * 1e7
        dvdx = np.gradient(v850, axis=1) / dx; dudy = np.gradient(u850, axis=0) / dy
        vort = (dvdx - dudy)[wide].mean() * 1e7
        sst = g("skt")[core].mean() - 273.15
        mpi = mpi_dk94(min(sst, 31.5))   # cap: skt over land can exceed 35 C and the formula explodes
        return dict(shear_kt=shear, shdir_deg=shdir, rh700=rh7, rh500=rh5, rhmid=(rh7 + rh5) / 2,
                    t200_c=g("t200")[ring].mean() - 273.15, div200=div, vort850=vort,
                    u200_kt=u200[ring].mean() * MS2KT, sst_c=sst, mpi_kt=mpi, pot_kt=mpi - vmax_kt)

def init_position(bt, sid, init):
    r = bt[(bt.SID == sid) & (bt.ISO_TIME == init)]
    return (float(r.USA_LAT.iloc[0]), float(r.USA_LON.iloc[0])) if len(r) else None

def main(limit=None):
    df = pd.read_csv(f"{TC}/aifs_tracks_eta.csv", keep_default_na=False, na_values=[""])
    bt = pd.read_csv(f"{TC}/besttrack.csv", keep_default_na=False, na_values=[""])
    bt["ISO_TIME"] = pd.to_datetime(bt.ISO_TIME).astype(str)
    out = []
    inits = sorted(df.init.unique())[:limit]
    for n, init in enumerate(inits):
        path = f"{PL}/{pd.Timestamp(init):%Y%m%d%H}.npz"
        if not os.path.exists(path):
            print(f"[{n + 1}/{len(inits)}] {init} missing npz, skipped", flush=True); continue
        F = Fields(path)
        for sid, g in df[df.init == init].groupby("sid"):
            g = g.sort_values("step")
            pos0 = init_position(bt, sid, init)
            if pos0 is None:           # extrapolate one step back from the first two forecast positions
                a, b = g.iloc[0], g.iloc[1] if len(g) > 1 else g.iloc[0]
                pos0 = (2 * a.lat - b.lat, 2 * a.lon - b.lon)
            v0 = float(g.aifs0_vmax.iloc[0])
            p0 = F.predictors(0, pos0[0], pos0[1], v0)
            along = [p0] if p0 else []
            for _, r in g.iterrows():
                p = F.predictors(int(r.step), float(r.lat), float(r.lon), float(r.vmax_kt))
                if p is None:
                    continue
                along.append(p)
                row = {"init": init, "sid": sid, "step": int(r.step)}
                row.update(p)
                if p0:
                    row.update({f"{k}_0": v for k, v in p0.items()})
                m = pd.DataFrame(along).mean()
                row.update({f"{k}_m": m[k] for k in p})
                out.append(row)
        print(f"[{n + 1}/{len(inits)}] {init} {len(out)} rows", flush=True)
    res = pd.DataFrame(out)
    res.to_csv(f"{TC}/ships_features.csv", index=False)
    print(f"wrote {len(res)} rows to data/tc/ships_features.csv")

if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[a.index("--limit") + 1]) if "--limit" in a else None)
