"""Build the all-basin labelled dataset for the core scripts (item 3, 2026-11-23).

Takes the North Atlantic and East Pacific tracks as they are (data/tc/aifs_tracks.csv) and the new
West Pacific, North Indian and Southern Hemisphere tracks (data/tc_global/aifs_tracks_wpsh.csv),
builds the same features as correct.py against the all-basin best track, adds basin flags, and
writes data/tc_global/reference.csv with the columns the core scripts expect (no repo predictions).

Run from the repo root:  python pipeline/build_global.py
Then:  cd core && SCMP_DATA=global python quantile.py   (core/data/*_global.csv link to data/tc_global/)
"""
import os
import numpy as np, pandas as pd
from correct import build_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TC = os.path.join(ROOT, "data", "tc_global")

def main():
    old = pd.read_csv(f"{ROOT}/data/tc/aifs_tracks.csv", parse_dates=["init", "valid"], keep_default_na=False, na_values=[""])
    new = pd.read_csv(f"{TC}/aifs_tracks_wpsh.csv", parse_dates=["init", "valid"], keep_default_na=False, na_values=[""])
    tr = pd.concat([old, new], ignore_index=True)
    bt = pd.read_csv(f"{TC}/besttrack.csv", parse_dates=["ISO_TIME"], keep_default_na=False, na_values=[""])
    df = build_features(tr, bt)
    df["is_wp"] = (df.basin == "WP").astype(int)
    df["is_ni"] = (df.basin == "NI").astype(int)
    df["is_sh"] = df.basin.isin(["SI", "SP", "SA"]).astype(int)
    lab = df.dropna(subset=["bt_wind_kt", "obs0_wind"]).copy()
    lab = lab[lab.step > 0]
    lab["res_v"] = lab.bt_wind_kt - lab.vmax_kt
    lab.to_csv(f"{TC}/reference.csv", index=False)
    print(f"wrote data/tc_global/reference.csv: {len(lab)} cases, {lab.sid.nunique()} storms, {lab.ri.sum()} RI cases")
    print(lab.groupby("basin").agg(cases=("step", "size"), storms=("sid", "nunique"), ri=("ri", "sum"),
                                   raw_mae=("res_v", lambda x: x.abs().mean())).to_string())

if __name__ == "__main__":
    main()
