"""Summarise the AIFS-ENS member intensities (data/tc/ens_members.csv) into per-case features.

Output: data/tc/ens_features.csv, one row per (init, sid, step) that has all 51 members:
  ens_mean_kt, ens_std_kt, ens_q10_kt, ens_q90_kt, ens_max_kt   member 10 m wind statistics
  ens_cf_kt            control member wind
  ens_mean_minus_det   ensemble mean minus the AIFS-Single wind at that step
  ens_cf_minus_det     control minus AIFS-Single
  ens_frac_above_det   fraction of members above AIFS-Single
  ens_pmin_mean_hpa, ens_pmin_std_hpa   member minimum MSLP statistics

Usage: python pipeline/ens_features.py
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TC = os.path.join(ROOT, "data", "tc")

def main():
    e = pd.read_csv(f"{TC}/ens_members.csv")
    det = pd.read_csv(f"{TC}/aifs_tracks_eta.csv", keep_default_na=False, na_values=[""])[["init", "sid", "step", "vmax_kt"]]
    keys = ["init", "sid", "step"]
    n = e.groupby(keys).member.transform("size"); e = e[n == 51]
    g = e.groupby(keys)
    f = g.vmax_kt.agg(ens_mean_kt="mean", ens_std_kt="std", ens_max_kt="max",
                      ens_q10_kt=lambda s: s.quantile(0.1), ens_q90_kt=lambda s: s.quantile(0.9))
    f["ens_cf_kt"] = e[e.member == 0].set_index(keys).vmax_kt
    f["ens_pmin_mean_hpa"] = g.pmin_hpa.mean(); f["ens_pmin_std_hpa"] = g.pmin_hpa.std()
    f = f.reset_index().merge(det, on=keys, how="inner")
    f["ens_mean_minus_det"] = f.ens_mean_kt - f.vmax_kt
    f["ens_cf_minus_det"] = f.ens_cf_kt - f.vmax_kt
    above = e.merge(det, on=keys, suffixes=("", "_det")); above = above[above.member > 0]
    f = f.merge((above.vmax_kt > above.vmax_kt_det).groupby([above.init, above.sid, above.step]).mean()
                .rename("ens_frac_above_det").reset_index(), on=keys)
    f = f.drop(columns="vmax_kt")
    f.to_csv(f"{TC}/ens_features.csv", index=False)
    print(f"wrote {len(f)} rows, {f.shape[1] - 3} features to data/tc/ens_features.csv")
    print(f.describe().T[["mean", "std", "min", "max"]].round(2))

if __name__ == "__main__":
    main()
