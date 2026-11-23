"""Effect of the SHIPS-style predictors on the point GBM (report Table 7), from saved predictions.

Pairs the point GBM of quantile.py with base features (quantile_cv_base.csv, from `python quantile.py
--noships`) against the same GBM with base plus SHIPS (quantile_cv_ships.csv), so the table shares
one GBM code path, seed and weighting with the headline and quantile tables.

Run from core/:  python ships_ablation.py
"""
import numpy as np, pandas as pd
from quantile import OUT, storm_boot, fmt
from common import data_file

def main():
    b = pd.read_csv(f"{OUT}/quantile_cv_base.csv", keep_default_na=False, na_values=[""])
    s = pd.read_csv(f"{OUT}/quantile_cv_ships.csv", keep_default_na=False, na_values=[""])
    ref = pd.read_csv(data_file("reference"), keep_default_na=False, na_values=[""])[["init", "sid", "step", "ri"]]
    d = b.merge(s, on=["init", "sid", "step"], suffixes=("_base", "_ships")).merge(ref, on=["init", "sid", "step"])
    y, g = d.bt_wind_kt_base.values, d.sid.values
    eb, es = np.abs(y - d.point_base.values), np.abs(y - d.point_ships.values)
    print(f"point GBM, base vs base + SHIPS: {len(d)} cases, {d.sid.nunique()} storms")
    print("subset      n   storms   base   +SHIPS   diff [CI]")
    for name, m in [("all", np.ones(len(d), bool)), ("RI", d.ri.values == 1),
                    ("lead<=48", d.step.values <= 48), ("lead>96", d.step.values > 96)]:
        diff = es[m] - eb[m]
        print(f"{name:10s} {m.sum():5d} {len(np.unique(g[m])):6d}   {eb[m].mean():6.2f}   {es[m].mean():6.2f}   "
              f"{fmt(diff.mean(), storm_boot(diff, g[m]))}")

if __name__ == "__main__":
    main()
