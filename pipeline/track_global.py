"""Track the storms of the other basins in the wp and sh surface-field boxes (item 3, 2026-11-23).

Same tracker as track.py (imported), run over data/aifs_wp and data/aifs_sh against the all-basin
best track of catalog_global.py. A best-track fix at the init time is tracked in a box only if it
lies inside that box; the sh box stores longitudes in 0..360, so best-track longitudes are mapped
to that convention there. The North Atlantic and East Pacific tracks are not redone: build_global.py
takes them from data/tc/aifs_tracks.csv.

Output: data/tc_global/aifs_tracks_wpsh.csv
Usage:  python pipeline/track_global.py [--limit N]
"""
import glob, os, sys
import numpy as np, pandas as pd
from track import track, haversine_km

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TC = os.path.join(ROOT, "data", "tc_global")

def main(limit=None):
    bt = pd.read_csv(f"{TC}/besttrack.csv", parse_dates=["ISO_TIME"], keep_default_na=False, na_values=[""])
    bt["USA_LON"] = np.where(bt.USA_LON > 180, bt.USA_LON - 360, bt.USA_LON)
    rows = []
    for box in ["wp", "sh"]:
        files = sorted(glob.glob(f"{ROOT}/data/aifs_{box}/*.npz"))[:limit]
        for f in files:
            init = pd.Timestamp(os.path.basename(f)[:8]) + pd.Timedelta(hours=int(os.path.basename(f)[8:10]))
            run = np.load(f)
            lats, lons = run["lats"], run["lons"]
            wrap = lons.max() > 180
            fixes = bt[(bt.ISO_TIME == init) & (bt.box == box)].copy()
            if wrap:
                fixes["USA_LON"] = np.where(fixes.USA_LON < 0, fixes.USA_LON + 360, fixes.USA_LON)
            inside = (fixes.USA_LAT >= lats.min()) & (fixes.USA_LAT <= lats.max()) & \
                     (fixes.USA_LON >= lons.min()) & (fixes.USA_LON <= lons.max())
            fixes = fixes[inside]
            n = 0
            for _, fx in fixes.iterrows():
                tr = track(run, fx.USA_LAT, fx.USA_LON)
                if tr.empty:
                    continue
                if wrap:   # back to -180..180 for the output, as in the NA+EP tracks
                    tr["lon"] = np.where(tr.lon > 180, tr.lon - 360, tr.lon)
                tr["init"], tr["sid"], tr["name"], tr["basin"] = init, fx.SID, fx.NAME, fx.BASIN
                tr["valid"] = init + pd.to_timedelta(tr.step, unit="h")
                obs = bt[bt.SID == fx.SID][["ISO_TIME", "USA_LAT", "USA_LON", "USA_WIND", "USA_PRES"]] \
                    .rename(columns={"ISO_TIME": "valid", "USA_LAT": "bt_lat", "USA_LON": "bt_lon",
                                     "USA_WIND": "bt_wind_kt", "USA_PRES": "bt_pres_hpa"})
                tr = tr.merge(obs, on="valid", how="left")
                dlon = (tr.lon - tr.bt_lon + 180) % 360 - 180
                tr["track_err_km"] = haversine_km(tr.lat, tr.bt_lon + dlon, tr.bt_lat, tr.bt_lon)
                rows.append(tr); n += 1
            print(f"{box} {init:%Y-%m-%d %HZ}: {n} storm(s) tracked", flush=True)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out = f"{TC}/aifs_tracks_wpsh.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows, {df.sid.nunique() if len(df) else 0} storms, {df.init.nunique() if len(df) else 0} inits")

if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[a.index("--limit") + 1]) if "--limit" in a else None)
