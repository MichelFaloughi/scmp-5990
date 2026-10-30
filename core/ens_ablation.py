"""Does AIFS-ENS spread help the correction? Ablation on the cases that have ensemble members.

The ENS fetch (pipeline/fetch_aifs_ens.py) covers inits from 2025-07-02 that contain an RI storm,
steps 6 to 120 h, every case active at those inits: 3323 cases, 33 storms. Both models below are
trained and scored on exactly this subset with the same storm-grouped 5-fold CV, with and without
the 11 ENS features of pipeline/ens_features.py added to base + SHIPS. Paired storm-bootstrap CIs.

Also reports the spread-skill relation: Spearman correlation of the member standard deviation with
the absolute error of the point GBM, and of the quantile model's interval width.

Run from core/:  python ens_ablation.py      (writes data/out/ens_ablation_cv.csv)
"""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from quantile import load, predict_cv, point_gbm_cv, crps_from_quantiles, storm_boot, fmt, Q, OUT

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    df, feats = load(ships=True)
    ef = pd.read_csv(os.path.join(HERE, "data", "ens_features.csv"), keep_default_na=False, na_values=[""])
    ens = [c for c in ef.columns if c not in ("init", "sid", "step")]
    df = df.merge(ef, on=["init", "sid", "step"], how="inner").reset_index(drop=True)
    y, g, ri, st = df.bt_wind_kt.values, df.sid.values, df.ri.values == 1, df.step.values
    print(f"ENS subset: {len(df)} cases, {df.sid.nunique()} storms, {ri.sum()} RI cases")
    res = {}
    for tag, f in [("ships", feats), ("ships_ens", feats + ens)]:
        pt = point_gbm_cv(df, f); P = predict_cv(df, f)
        res[tag] = dict(pt=pt, med=P[:, list(Q).index(0.5)], crps=crps_from_quantiles(P, y),
                        cover=((y >= P[:, 2]) & (y <= P[:, 6])).astype(float), width=P[:, 6] - P[:, 2])
    masks = [("all", np.ones(len(df), bool)), ("RI", ri), ("lead<=48", st <= 48), ("lead>48", st > 48), ("obs>=96kt", y >= 96)]
    print("\nsubset      n   MAE point ships / +ens   diff [CI]              MAE median ships / +ens   diff [CI]              CRPS ships / +ens   cover")
    for name, m in masks:
        a, b = res["ships"], res["ships_ens"]
        dp = np.abs(y - b["pt"])[m] - np.abs(y - a["pt"])[m]; dm = np.abs(y - b["med"])[m] - np.abs(y - a["med"])[m]
        print(f"{name:10s} {m.sum():5d}   {np.abs(y - a['pt'])[m].mean():6.2f} / {np.abs(y - b['pt'])[m].mean():6.2f}   "
              f"{fmt(dp.mean(), storm_boot(dp, g[m])):>22s}   {np.abs(y - a['med'])[m].mean():6.2f} / {np.abs(y - b['med'])[m].mean():6.2f}   "
              f"{fmt(dm.mean(), storm_boot(dm, g[m])):>22s}   {a['crps'][m].mean():5.2f} / {b['crps'][m].mean():5.2f}   "
              f"{a['cover'][m].mean():.0%} / {b['cover'][m].mean():.0%}")
    dc = res["ships_ens"]["crps"] - res["ships"]["crps"]
    print(f"\nCRPS diff (+ens minus ships), all: {fmt(dc.mean(), storm_boot(dc, g))}")
    print("\nspread-skill (Spearman):")
    ae = np.abs(y - res["ships"]["pt"])
    print(f"  ENS member std vs |error| of point GBM (ships):      {spearmanr(df.ens_std_kt, ae).statistic:+.3f}")
    print(f"  ENS member std vs raw AIFS |error|:                   {spearmanr(df.ens_std_kt, np.abs(y - df.vmax_kt)).statistic:+.3f}")
    print(f"  ENS member std vs 10-90 width of quantile model:      {spearmanr(df.ens_std_kt, res['ships']['width']).statistic:+.3f}")
    print(f"  10-90 width (ships) vs |error| of its median:         {spearmanr(res['ships']['width'], np.abs(y - res['ships']['med'])).statistic:+.3f}")
    print(f"  ens_mean_minus_det vs residual (obs minus AIFS):      {spearmanr(df.ens_mean_minus_det, df.res_v).statistic:+.3f}")
    print(f"  ens_frac_above_det vs residual:                       {spearmanr(df.ens_frac_above_det, df.res_v).statistic:+.3f}")
    out = df[["init", "sid", "step", "ri", "bt_wind_kt", "vmax_kt", "ens_std_kt", "ens_mean_minus_det"]].copy()
    for tag in res:
        out[f"point_{tag}"] = res[tag]["pt"]; out[f"median_{tag}"] = res[tag]["med"]; out[f"crps_{tag}"] = res[tag]["crps"]
    os.makedirs(OUT, exist_ok=True); out.to_csv(f"{OUT}/ens_ablation_cv.csv", index=False)

if __name__ == "__main__":
    main()
