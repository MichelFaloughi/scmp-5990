"""Simple tropical-cyclone tracker for AIFS surface fields.

For each fetched AIFS run and each storm that exists in best track at init time:
  - seed at the best-track position at init
  - at each 6 h step, look for the sea-level-pressure minimum within SEARCH_KM of a first guess
    (previous position + previous motion), on a lightly smoothed field
  - intensity: pmin = that minimum (hPa); vmax = max 10 m wind within VMAX_KM (m/s and kt)
  - stop when the low is no longer distinct (pmin above WEAK_HPA and vmax below WEAK_MS)
Matches each forecast point to the best-track fix at the same valid time when available.

Output: data/tc/aifs_tracks.csv
Usage:  python tc/track.py [YYYYMMDDHH ...]      (default: every file in data/aifs)
"""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEARCH_KM, VMAX_KM = 450.0, 250.0
WEAK_HPA, WEAK_MS = 1010.0, 12.0
KT = 1.943844

def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

def track(run, seed_lat, seed_lon, start_step=0):
    lats, lons, steps = run["lats"], run["lons"], run["steps"]
    LON, LAT = np.meshgrid(lons, lats)
    msl, u, v = run["msl"] / 100.0, run["u10"].astype(np.float32), run["v10"].astype(np.float32)
    wind = np.hypot(u, v)
    out, lat, lon, vlat, vlon = [], seed_lat, seed_lon, 0.0, 0.0
    for k, step in enumerate(steps):
        if step < start_step:
            continue
        glat, glon = (lat + vlat, lon + vlon) if step > start_step else (lat, lon)
        dist = haversine_km(glat, glon, LAT, LON)
        sm = uniform_filter(msl[k], size=3, mode="nearest")
        cand = np.where(dist <= SEARCH_KM, sm, np.inf)
        j = np.unravel_index(np.argmin(cand), cand.shape)
        nlat, nlon, pmin = float(LAT[j]), float(LON[j]), float(msl[k][j])
        dv = haversine_km(nlat, nlon, LAT, LON)
        vmax = float(wind[k][dv <= VMAX_KM].max())
        if step > start_step and pmin > WEAK_HPA and vmax < WEAK_MS:
            break
        vlat, vlon = (nlat - lat, nlon - lon) if step > start_step else (0.0, 0.0)
        lat, lon = nlat, nlon
        out.append(dict(step=int(step), lat=lat, lon=lon, pmin_hpa=pmin, vmax_ms=vmax, vmax_kt=vmax * KT))
    return pd.DataFrame(out)

def main(files):
    bt = pd.read_csv(f"{ROOT}/data/tc/besttrack.csv", parse_dates=["ISO_TIME"])
    bt["USA_LON"] = np.where(bt.USA_LON > 180, bt.USA_LON - 360, bt.USA_LON)
    rows = []
    for f in files:
        init = pd.Timestamp(os.path.basename(f)[:8]) + pd.Timedelta(hours=int(os.path.basename(f)[8:10]))
        run = np.load(f)
        fixes = bt[bt.ISO_TIME == init]
        for _, fx in fixes.iterrows():
            tr = track(run, fx.USA_LAT, fx.USA_LON)
            if tr.empty:
                continue
            tr["init"], tr["sid"], tr["name"], tr["basin"] = init, fx.SID, fx.NAME, fx.BASIN
            tr["valid"] = init + pd.to_timedelta(tr.step, unit="h")
            obs = bt[bt.SID == fx.SID][["ISO_TIME", "USA_LAT", "USA_LON", "USA_WIND", "USA_PRES"]] \
                .rename(columns={"ISO_TIME": "valid", "USA_LAT": "bt_lat", "USA_LON": "bt_lon",
                                 "USA_WIND": "bt_wind_kt", "USA_PRES": "bt_pres_hpa"})
            tr = tr.merge(obs, on="valid", how="left")
            tr["track_err_km"] = haversine_km(tr.lat, tr.lon, tr.bt_lat, tr.bt_lon)
            rows.append(tr)
        print(f"{init:%Y-%m-%d %HZ}: {len(fixes)} storm(s) tracked", flush=True)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return df

if __name__ == "__main__":
    a = sys.argv[1:]
    files = [f"{ROOT}/data/aifs/{x}.npz" for x in a] if a else sorted(glob.glob(f"{ROOT}/data/aifs/*.npz"))
    df = main(files)
    out = f"{ROOT}/data/tc/aifs_tracks.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows, {df.sid.nunique() if len(df) else 0} storms, {df.init.nunique() if len(df) else 0} inits")
