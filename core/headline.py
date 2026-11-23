"""Headline score table for the report, from the prediction files already on disk.

Advisor comment (2026-11-11): do not headline MAE on RI cases (selecting by observed outcome rewards
any model that forecasts high); lead with CRPS and a tail-weighted CRPS and add hit rate and false
alarms at 96 and 120 kt. For a point forecast the CRPS is its absolute error, so the point models
(raw AIFS, GBM, tuned MLP, eta) and the quantile model are scored on one scale. twCRPS uses the
weight 1{z >= t}, t = 96 kt (Gneiting and Ranjan 2011). Events: forecast median or point >= t.

Run from core/:  python headline.py      (prints; writes data/out/headline_cv.csv)
"""
import os
import numpy as np, pandas as pd
from quantile import Q, QCOLS, OUT, crps_from_quantiles, twcrps, contingency, storm_boot, fmt

def load(name):
    return pd.read_csv(f"{OUT}/{name}.csv", keep_default_na=False, na_values=[""])

def table(tag, d, models, ref="gbm"):
    y, g = d.bt_wind_kt.values, d.sid.values
    rows = []
    print(f"\n=== {tag}: {len(d)} cases, {d.sid.nunique()} storms ===")
    print(f"{'model':28s} {'CRPS':>6s} {'vs ' + ref:>22s} {'twCRPS>=96':>10s} {'vs ' + ref:>22s}  {'96 kt POD/FAR':>14s} {'120 kt POD/FAR':>15s}")
    scores = {}
    for name, P in models:
        c = crps_from_quantiles(P, y) if P.shape[1] > 1 else np.abs(y - P[:, 0])
        scores[name] = (c, twcrps(P, y, 96), P[:, list(Q).index(0.5)] if P.shape[1] > 1 else P[:, 0])
    cr, tr, _ = scores[ref]
    for name, _ in models:
        c, t, f = scores[name]
        ev = [contingency(f >= thr, y >= thr).split("(")[0].replace("POD ", "").replace("  FAR ", "/").split("  bias")[0] for thr in (96, 120)]
        dc, dt = c - cr, t - tr
        print(f"{name:28s} {c.mean():6.2f} {fmt(dc.mean(), storm_boot(dc, g)) if name != ref else '':>22s} {t.mean():10.2f} "
              f"{fmt(dt.mean(), storm_boot(dt, g)) if name != ref else '':>22s}  {ev[0]:>14s} {ev[1]:>15s}")
        rows.append(dict(eval=tag, model=name, crps=c.mean(), twcrps96=t.mean(),
                         **{f"pod{thr}": ((f >= thr) & (y >= thr)).sum() / max((y >= thr).sum(), 1) for thr in (96, 120)},
                         **{f"far{thr}": ((f >= thr) & (y < thr)).sum() / max((f >= thr).sum(), 1) for thr in (96, 120)}))
    return rows

def main():
    fb = load("fair_baseline_cv"); qc = load("quantile_cv_ships")
    qb = load("quantile_cv_base")[["init", "sid", "step", "point"]].rename(columns={"point": "point_base"})
    d = fb.merge(qc, on=["init", "sid", "step"], suffixes=("", "_q")).merge(qb, on=["init", "sid", "step"])
    col = lambda c: d[c].values[:, None]
    # the reference GBM is the one of quantile.py on base features (unweighted, single seed), so that the
    # GBM rows differ only by their features; the fair-baseline file's own GBM (2x RI-weighted) is kept as a row
    models = [("raw AIFS", col("vmax_kt")), ("GBM (point, base features)", col("point_base")),
              ("GBM (2x RI, fair-baseline run)", col("gbm")), ("plain MLP", col("erm600")),
              ("tuned MLP", col("erm_tuned")), ("tuned MLP + quantile mapping", col("erm_tuned_qm")), ("eta", col("eta")),
              ("GBM (point, base + SHIPS)", col("point")), ("quantile GBM (base + SHIPS)", d[QCOLS].values)]
    rows = table("storm-grouped CV", d, models, ref="GBM (point, base features)")
    if os.path.exists(f"{OUT}/quantile_holdout_ships.csv"):
        h = load("quantile_holdout_ships")
        ref = pd.read_csv("data/reference.csv", keep_default_na=False, na_values=[""])[["init", "sid", "step", "vmax_kt", "corr_vmax_kt", "eta_vmax_kt", "erm_vmax_kt"]]
        h = h.merge(ref, on=["init", "sid", "step"], suffixes=("_q", ""))
        col = lambda c: h[c].values[:, None]
        models = [("raw AIFS", col("vmax_kt")), ("GBM (point, base + SHIPS)", col("point")), ("quantile GBM (base + SHIPS)", h[QCOLS].values)]
        rows += table("temporal holdout 2025->2026", h, models, ref="GBM (point, base + SHIPS)")
    pd.DataFrame(rows).to_csv(f"{OUT}/headline_cv.csv", index=False)

if __name__ == "__main__":
    main()
