"""P(RI) from a direct model of the 24 h intensity change.

Advisor comment (2026-11-11): reading RI off per-lead intensity marginals, with a modelled base 24 h
earlier, is not a forecast of the 24 h change. Here the target is the observed change itself,
dv24 = best-track wind at the valid time minus best-track wind 24 h earlier (the quantity whose
exceedance of 30 kt defines RI). A quantile GBM on dv24 with the base + SHIPS features plus the
AIFS forecast's own 24 h change, no RI case weights, spread-scaled cross-conformal recalibration
(quantile.py defaults). P(RI) = 1 - F_dv24(30).

Compared with the two earlier routes on the same cases: P(RI) from the intensity quantiles
(quantile.py output, if present) and the direct gradient-boosted classifier.

Run from core/:  python riprob.py [--carq] [--scaled]      (writes data/out/riprob_{cv,holdout}_*.csv)
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import quantile as qm
from quantile import Q, OUT, fit_quantiles, conformal_offsets, apply_offsets, cdf_from_quantiles, brier_report, \
    contingency, crps_from_quantiles, storm_boot, fmt, classifier_ri

HERE = os.path.dirname(os.path.abspath(__file__))
RI_KT = 30.0

def add_dv24(df):
    """Observed 24 h change ending at the valid time, and the AIFS forecast's own 24 h change."""
    from common import data_file
    bt = pd.read_csv(data_file("besttrack"), keep_default_na=False, na_values=[""])
    w = bt.assign(t=pd.to_datetime(bt.ISO_TIME)).set_index(["SID", "t"]).USA_WIND
    valid = pd.to_datetime(df.valid)
    df["dv24"] = df.bt_wind_kt.values - w.reindex(pd.MultiIndex.from_arrays([df.sid, valid - pd.Timedelta(hours=24)])).values
    v = df.set_index(["init", "sid", "step"]).vmax_kt
    prev = v.reindex(pd.MultiIndex.from_arrays([df.init, df.sid, df.step - 24])).values
    df["aifs_dv24"] = df.vmax_kt.values - np.where(df.step.values == 24, df.aifs0_vmax.values, prev)
    return df

def fit_dv24(df, feats, holdout=False, seed=0):
    X = df[feats].values; y = df.dv24.values; g = df.sid.values; st = df.step.values
    w = np.ones(len(df)); P = np.full((len(df), len(Q)), np.nan)
    splits = [(qm.train_mask(df), ~qm.train_mask(df))] if holdout else GroupKFold(5).split(X, y, g)
    for tr, te in splits:
        tr, te = np.asarray(tr), np.asarray(te)
        Pte = fit_quantiles(X[tr], y[tr], w[tr], X[te], seed)
        P[te] = apply_offsets(Pte, conformal_offsets(X[tr], y[tr], w[tr], g[tr], st[tr], seed), st[te])
    return P

def score(tag, d, P, p_cls, p_int):
    y, ri, g = d.dv24.values, d.ri.values, d.sid.values
    print(f"\n=== {tag}: {len(d)} cases, step >= 24, {d.sid.nunique()} storms, {ri.sum()} RI cases ===")
    med = P[:, list(Q).index(0.5)]
    print(f"24 h change: MAE median {np.abs(y - med).mean():.2f}, MAE raw AIFS change {np.abs(y - d.aifs_dv24.values).mean():.2f}, "
          f"MAE persistence (0) {np.abs(y).mean():.2f}; CRPS {crps_from_quantiles(P, y).mean():.2f}; "
          f"80% cover {((y >= P[:, 2]) & (y <= P[:, 6])).mean():.0%}; "
          f"q0.95-level exceeds 30 kt for {(P[:, 7] >= RI_KT).mean():.1%} of cases")
    p = 1.0 - cdf_from_quantiles(P, np.full(len(d), RI_KT))
    out = {"p_ri_dv24": p}
    for name, pp in [("P(RI) from dv24 quantiles", p), ("P(RI) from intensity quantiles (quantile.py)", p_int),
                     ("P(RI) direct classifier", p_cls)]:
        if pp is None or not np.isfinite(pp).any():
            continue
        ok = np.isfinite(pp)
        brier_report(name, pp, ri)
        auc = roc_auc_score(ri[ok], pp[ok])
        print(f"   AUC {auc:.3f};  " + ";  ".join(f"p>={t}: " + contingency(pp[ok] >= t, ri[ok] == 1).split('(')[0] for t in (0.2, 0.3, 0.5)))
    if p_int is not None and np.isfinite(p_int).any():
        ok = np.isfinite(p_int)
        d_b = (p[ok] - ri[ok]) ** 2 - (p_int[ok] - ri[ok]) ** 2
        print(f"   Brier dv24 minus intensity route: {fmt(d_b.mean(), storm_boot(d_b, g[ok]))}")
    return out

def main():
    carq = "--carq" in sys.argv; qm.SCALED = "--scaled" in sys.argv; ships = "--noships" not in sys.argv
    qm.BASIN_HOLDOUT = "--basin-holdout" in sys.argv
    df, feats = qm.load(ships, carq)
    if qm.BASIN_HOLDOUT:
        feats = [f for f in feats if f not in ("is_wp", "is_ni", "is_sh")]
    df = add_dv24(df); feats = feats + ["aifs_dv24"]
    df = df[(df.step >= 24) & df.dv24.notna()].reset_index(drop=True)
    tag = ("ships" if ships else "base") + ("_scaled" if qm.SCALED else "") + ("_carq" if carq else "") \
        + ("_xbasin" if qm.BASIN_HOLDOUT else "")
    os.makedirs(OUT, exist_ok=True)
    hname = "cross-basin holdout NA+EP -> WP+NI+SH" if qm.BASIN_HOLDOUT else "temporal holdout 2025->2026"
    for name, holdout in ([("holdout", True)] if qm.BASIN_HOLDOUT else [("cv", False), ("holdout", True)]):
        P = fit_dv24(df, feats, holdout)
        m = np.isfinite(P[:, 0]); d = df[m].reset_index(drop=True); P = P[m]
        p_cls = classifier_ri(df, feats, holdout=holdout)[m]
        f = f"{OUT}/quantile_{name}_{tag}.csv"
        p_int = None
        if os.path.exists(f):
            qi = pd.read_csv(f, keep_default_na=False, na_values=[""])[["init", "sid", "step", "p_ri"]]
            p_int = d[["init", "sid", "step"]].merge(qi, how="left", on=["init", "sid", "step"]).p_ri.values
        res = score(f"{'storm-grouped CV' if not holdout else hname}, {tag}", d, P, p_cls, p_int)
        o = d[["init", "sid", "step", "ri", "dv24", "aifs_dv24"]].copy()
        for j, q in enumerate(Q):
            o[f"dq{int(q * 100):02d}"] = P[:, j]
        o["p_ri_dv24"] = res["p_ri_dv24"]; o["p_ri_cls"] = p_cls; o["p_ri_int"] = p_int if p_int is not None else np.nan
        o.to_csv(f"{OUT}/riprob_{name}_{tag}.csv", index=False)

if __name__ == "__main__":
    main()
