"""Point models wrapped in the same conformal residual distribution (advisor, 2026-11-23).

The CRPS of a point forecast is its absolute error, so the quantile GBM wins the headline CRPS
comparison partly by construction. The fair comparison gives every point model a predictive
distribution built the same way: storm-grouped 5-fold split conformal on its out-of-fold errors.
For each outer fold the residuals y - p of the training storms (p taken from the saved CV
predictions, which were made out of fold) give one offset per quantile level and lead band
(0-48, 54-96, 102-168 h); the test storms' forecasts are p plus those offsets. This is the recipe
of quantile.py with the model's own quantiles replaced by a constant residual shape. The quantile
GBM's median is also wrapped, so the comparison isolates what its case-dependent quantiles add.

On the temporal holdout the offsets come from the 2025 out-of-fold residuals and are applied to
the 2026 point forecasts (available for raw AIFS and the point GBM only).

Run from core/:  python point_conformal.py        (SCMP_DATA=global for the all-basin run)
Writes data/out[/global]/point_conformal_cv.csv.
"""
import os
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from quantile import Q, QCOLS, OUT, BANDS, band_of, crps_from_quantiles, twcrps, storm_boot, fmt
from common import data_file

def wrap_cv(p, y, g, step, seed=0):
    P = np.full((len(p), len(Q)), np.nan)
    for tr, te in GroupKFold(5).split(p, y, g):
        P[te] = p[te, None] + offsets(y[tr] - p[tr], step[tr])[band_of(step[te])]
    return np.sort(P, axis=1)

def offsets(r, step):
    bands = band_of(step)
    return np.array([[np.quantile(r[bands == k], q) for q in Q] for k in range(len(BANDS))])

def table(tag, d, models, ref):
    y, g = d.bt_wind_kt.values, d.sid.values
    print(f"\n=== {tag}: {len(d)} cases, {d.sid.nunique()} storms ===")
    print(f"{'model':34s} {'MAE':>6s} {'CRPS':>6s} {'vs ' + ref:>24s} {'twCRPS>=96':>10s} {'vs ' + ref:>22s} {'cover':>6s}")
    sc = {}
    for name, P, pt in models:
        sc[name] = (crps_from_quantiles(P, y), twcrps(P, y, 96), ((y >= P[:, list(Q).index(0.10)]) & (y <= P[:, list(Q).index(0.90)])), np.abs(y - pt))
    cr, tr = sc[ref][0], sc[ref][1]
    rows = []
    for name, _, _ in models:
        c, t, cov, ae = sc[name]
        dc, dt = c - cr, t - tr
        print(f"{name:34s} {ae.mean():6.2f} {c.mean():6.2f} {fmt(dc.mean(), storm_boot(dc, g)) if name != ref else '':>24s} "
              f"{t.mean():10.2f} {fmt(dt.mean(), storm_boot(dt, g)) if name != ref else '':>22s} {cov.mean():6.1%}")
        rows.append(dict(eval=tag, model=name, mae=ae.mean(), crps=c.mean(), twcrps96=t.mean(), cover80=cov.mean()))
    return rows

def main():
    feat = "ships" if os.path.exists(f"{OUT}/quantile_cv_ships.csv") else "base"
    rd = lambda n: pd.read_csv(f"{OUT}/{n}.csv", keep_default_na=False, na_values=[""])
    fb = rd("fair_baseline_cv"); qc = rd(f"quantile_cv_{feat}")
    qb = rd("quantile_cv_base")[["init", "sid", "step", "point"]].rename(columns={"point": "point_base"})
    d = fb.merge(qc, on=["init", "sid", "step"], suffixes=("", "_q")).merge(qb, on=["init", "sid", "step"])
    y, g, st = d.bt_wind_kt.values, d.sid.values, d.step.values
    lab = f"base{' + SHIPS' if feat == 'ships' else ''}"
    pts = [("raw AIFS", "vmax_kt"), ("GBM (point, base features)", "point_base"), ("plain MLP", "erm600"),
           ("tuned MLP", "erm_tuned"), ("tuned MLP + quantile mapping", "erm_tuned_qm"), ("eta", "eta")]
    if feat == "ships":
        pts.append((f"GBM (point, {lab})", "point"))
    models = [(f"{n} + conformal", wrap_cv(d[c].values, y, g, st), d[c].values) for n, c in pts]
    models.append((f"quantile GBM median + conformal", wrap_cv(d.q50.values, y, g, st), d.q50.values))
    models.append((f"quantile GBM ({lab})", d[QCOLS].values, d.q50.values))
    rows = table("storm-grouped CV, split conformal on out-of-fold errors", d, models, ref="GBM (point, base features) + conformal")
    if os.path.exists(f"{OUT}/quantile_holdout_{feat}.csv"):
        h = rd(f"quantile_holdout_{feat}")
        ref = pd.read_csv(data_file("reference"), keep_default_na=False, na_values=[""])[["init", "sid", "step", "vmax_kt"]]
        h = h.merge(ref, on=["init", "sid", "step"], suffixes=("_q", ""))
        yr = pd.to_datetime(d.init).dt.year.values
        tr = yr == 2025
        hm = []
        for n, c, tc in [("raw AIFS", "vmax_kt", "vmax_kt"), (f"GBM (point, {lab})", "point" if feat == "ships" else "point_base", "point")]:
            off = offsets(y[tr] - d[c].values[tr], st[tr])
            P = np.sort(h[tc].values[:, None] + off[band_of(h.step.values)], axis=1)
            hm.append((f"{n} + conformal (2025 residuals)", P, h[tc].values))
        off = offsets(y[tr] - d.q50.values[tr], st[tr])
        hm.append(("quantile GBM median + conformal (2025 residuals)", np.sort(h.q50.values[:, None] + off[band_of(h.step.values)], axis=1), h.q50.values))
        hm.append((f"quantile GBM ({lab})", h[QCOLS].values, h.q50.values))
        rows += table("temporal holdout 2025->2026, offsets from 2025 out-of-fold errors", h, hm, ref=f"GBM (point, {lab}) + conformal (2025 residuals)")
    pd.DataFrame(rows).to_csv(f"{OUT}/point_conformal_cv.csv", index=False)

if __name__ == "__main__":
    main()
