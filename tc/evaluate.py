"""Score intensity forecasts against best track, by lead time.

Compares raw AIFS (from the tracker), NHC OFCL, and optionally a corrected AIFS column,
using the same set of cases (init, storm, lead) wherever all are available.
Usage: python tc/evaluate.py [--corrected data/tc/aifs_tracks_corrected.csv]
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEADS = [12, 24, 36, 48, 72, 96, 120, 144, 168]

def load(corrected=None):
    tr = pd.read_csv(corrected or f"{ROOT}/data/tc/aifs_tracks.csv", parse_dates=["init", "valid"])
    bt = pd.read_csv(f"{ROOT}/data/tc/besttrack.csv", parse_dates=["ISO_TIME"])
    sid2atcf = bt.dropna(subset=["USA_ATCF_ID"]).drop_duplicates("SID").set_index("SID").USA_ATCF_ID
    tr["atcf_id"] = tr.sid.map(sid2atcf)
    of = pd.read_csv(f"{ROOT}/data/tc/ofcl.csv", parse_dates=["init"])
    of = of[of.tech == "OFCL"].rename(columns={"lead": "step", "vmax_kt": "ofcl_vmax_kt", "pmin_hpa": "ofcl_pmin_hpa",
                                               "lat": "ofcl_lat", "lon": "ofcl_lon"})
    m = tr.merge(of[["atcf_id", "init", "step", "ofcl_vmax_kt", "ofcl_pmin_hpa", "ofcl_lat", "ofcl_lon"]],
                 on=["atcf_id", "init", "step"], how="left")
    return m

def table(m, cols):
    m = m[m.step.isin(LEADS)].dropna(subset=["bt_wind_kt"])
    out = []
    for lead, g in m.groupby("step"):
        g2 = g.dropna(subset=[c for c in cols if c in g])  # homogeneous sample
        row = {"lead_h": lead, "n": len(g2)}
        for c in cols:
            if c in g2:
                row[f"MAE_{c}"] = np.abs(g2[c] - g2.bt_wind_kt).mean()
        out.append(row)
    return pd.DataFrame(out)

if __name__ == "__main__":
    corrected = sys.argv[sys.argv.index("--corrected") + 1] if "--corrected" in sys.argv else None
    m = load(corrected)
    cols = ["vmax_kt", "ofcl_vmax_kt"] + (["corr_vmax_kt"] if "corr_vmax_kt" in m else [])
    print(f"cases with best track: {m.bt_wind_kt.notna().sum()}  with OFCL: {m.ofcl_vmax_kt.notna().sum()}")
    print("\nMax-wind MAE (kt) vs best track, homogeneous sample per lead:")
    print(table(m, cols).round(1).to_string(index=False))
    bias = m[m.step.isin(LEADS)].dropna(subset=["bt_wind_kt"]).assign(b=lambda d: d.vmax_kt - d.bt_wind_kt).groupby("step").b.mean()
    print("\nRaw AIFS wind bias (kt):", bias.round(1).to_dict())
