"""Per-basin breakdown of the all-basin run (item 3, report Table in the "All basins" section).

Reads quantile_{cv,holdout}_{tag}.csv from data/out/global/ and the all-basin reference, and prints
for each basin: cases, storms, RI cases, raw AIFS MAE, point GBM MAE, quantile-median MAE, CRPS of
the quantile model and of the point GBM, 80% interval coverage, and tail-weighted CRPS above 96 kt.

Run from core/:  SCMP_DATA=global python basin_table.py [--tag base|ships]   (cv, holdout and, when
present, the leave-one-basin-out file quantile_lobo_{tag}.csv)
"""
import os, sys
import numpy as np, pandas as pd
from quantile import OUT, crps_from_quantiles, Q
from common import data_file

QCOLS = [f"q{int(q * 100):02d}" for q in Q]
GROUPS = [("NA", ["NA"]), ("EP", ["EP"]), ("WP", ["WP"]), ("NI", ["NI"]), ("SH", ["SI", "SP", "SA"])]

def table(split, tag, ref):
    d = pd.read_csv(f"{OUT}/quantile_{split}_{tag}.csv", keep_default_na=False, na_values=[""])
    d = d.merge(ref, on=["init", "sid", "step"])
    P = d[QCOLS].values; y = d.bt_wind_kt.values
    d["crps_q"] = crps_from_quantiles(P, y)
    d["crps_p"] = np.abs(y - d.point.values)
    d["tw_q"] = crps_from_quantiles(np.maximum(P, 96), np.maximum(y, 96))
    d["tw_p"] = np.abs(np.maximum(y, 96) - np.maximum(d.point.values, 96))
    d["cov"] = (y >= d.q10.values) & (y <= d.q90.values)
    d["mae_raw"] = np.abs(y - d.vmax_kt.values)
    d["mae_med"] = np.abs(y - d.q50.values)
    print(f"\n=== {split}, {tag}: {len(d)} cases, {d.sid.nunique()} storms ===")
    print("basin   cases storms   RI   raw   GBM  median  CRPS_q  CRPS_p  cover  twCRPS_q  twCRPS_p")
    rows = GROUPS + [("all", [b for _, bs in GROUPS for b in bs])]
    for name, bs in rows:
        m = d.basin.isin(bs)
        if m.sum() == 0:
            continue
        s = d[m]
        print(f"{name:6s} {len(s):6d} {s.sid.nunique():6d} {int(s.ri.sum()):4d} {s.mae_raw.mean():5.1f} {s.crps_p.mean():5.1f}  "
              f"{s.mae_med.mean():5.1f}   {s.crps_q.mean():5.2f}   {s.crps_p.mean():5.2f}  {100 * s['cov'].mean():4.0f}%   "
              f"{s.tw_q.mean():5.2f}     {s.tw_p.mean():5.2f}")

def main():
    a = sys.argv[1:]
    tag = a[a.index("--tag") + 1] if "--tag" in a else "ships"
    ref = pd.read_csv(data_file("reference"), keep_default_na=False, na_values=[""])[["init", "sid", "step", "basin", "ri", "vmax_kt"]]
    for split in ["cv", "holdout", "lobo"]:
        if os.path.exists(f"{OUT}/quantile_{split}_{tag}.csv"):
            table(split, tag, ref)

if __name__ == "__main__":
    main()
