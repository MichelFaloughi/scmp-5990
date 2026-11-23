"""SHIPS-style predictors for the all-basin dataset (item 3, 2026-11-23).

Reuses ships_features.py (imported) on the wp and sh pressure-level boxes for the cases of
data/tc_global/reference.csv that are not North Atlantic or East Pacific, and takes the existing
data/tc/ships_features.csv rows for those two basins. One change for the Southern Hemisphere: the
850 hPa relative vorticity is multiplied by the sign of the latitude so that cyclonic is positive in
both hemispheres (vort850, vort850_0, vort850_m). The sh box stores longitudes in 0..360, so case
longitudes are mapped to that convention there.

Output: data/tc_global/ships_features.csv
Usage:  python pipeline/ships_features_global.py [--limit N]
"""
import os, sys
import numpy as np, pandas as pd
from ships_features import Fields, init_position

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TC = os.path.join(ROOT, "data", "tc_global")
BOXDIR = {"wp": "aifs_pl_wp", "sh": "aifs_pl_sh"}

def main(limit=None):
    ref = pd.read_csv(f"{TC}/reference.csv", keep_default_na=False, na_values=[""])
    bt = pd.read_csv(f"{TC}/besttrack.csv", keep_default_na=False, na_values=[""])
    bt["ISO_TIME"] = pd.to_datetime(bt.ISO_TIME).astype(str)
    box_of = bt[["SID", "box"]].drop_duplicates().set_index("SID").box
    old = pd.read_csv(f"{ROOT}/data/tc/ships_features.csv", keep_default_na=False, na_values=[""])
    new = ref[~ref.basin.isin(["NA", "EP"])].copy()
    new["box"] = new.sid.map(box_of)
    out = []
    inits = sorted(new.init.unique())[:limit]
    for n, init in enumerate(inits):
        cases = new[new.init == init]
        for box, cb in cases.groupby("box"):
            path = f"{ROOT}/data/{BOXDIR[box]}/{pd.Timestamp(init):%Y%m%d%H}.npz"
            if not os.path.exists(path):
                print(f"[{n + 1}/{len(inits)}] {init} {box}: missing npz, skipped", flush=True); continue
            F = Fields(path)
            wrap = F.lons.max() > 180
            for sid, g in cb.groupby("sid"):
                g = g.sort_values("step")
                pos0 = init_position(bt, sid, init)
                if pos0 is None:
                    a, b = g.iloc[0], g.iloc[1] if len(g) > 1 else g.iloc[0]
                    pos0 = (2 * a.lat - b.lat, 2 * a.lon - b.lon)
                lon0 = pos0[1] + 360 if wrap and pos0[1] < 0 else pos0[1]
                v0 = float(g.aifs0_vmax.iloc[0])
                p0 = F.predictors(0, pos0[0], lon0, v0)
                along = [p0] if p0 else []
                for _, r in g.iterrows():
                    lon = r.lon + 360 if wrap and r.lon < 0 else r.lon
                    p = F.predictors(int(r.step), float(r.lat), float(lon), float(r.vmax_kt))
                    if p is None:
                        continue
                    along.append(p)
                    row = {"init": init, "sid": sid, "step": int(r.step)}
                    row.update(p)
                    if p0:
                        row.update({f"{k}_0": v for k, v in p0.items()})
                    m = pd.DataFrame(along).mean()
                    row.update({f"{k}_m": m[k] for k in p})
                    row["_lat"] = float(r.lat)
                    out.append(row)
        print(f"[{n + 1}/{len(inits)}] {init} {len(out)} rows", flush=True)
    res = pd.DataFrame(out)
    if len(res):
        sgn = np.sign(res.pop("_lat")).replace(0, 1)
        for c in ["vort850", "vort850_0", "vort850_m"]:
            res[c] = res[c] * sgn
    allrows = pd.concat([old, res], ignore_index=True)
    allrows.to_csv(f"{TC}/ships_features.csv", index=False)
    print(f"wrote {len(allrows)} rows ({len(old)} NA+EP reused, {len(res)} new) to data/tc_global/ships_features.csv")

if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[a.index("--limit") + 1]) if "--limit" in a else None)
