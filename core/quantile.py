"""Probabilistic intensity correction: quantile regression with gradient-boosted trees.

Predicts quantiles of the residual (observed minus AIFS wind), with the base features plus the
SHIPS-style environmental predictors. Evaluated two ways: storm-grouped 5-fold CV and a temporal
holdout (train on 2025 inits, test on 2026). Every comparison carries a paired storm-bootstrap CI.

Reported: MAE of the median, CRPS (quantile approximation), 80% interval coverage, tail quantiles,
and P(RI) scored with Brier skill and a reliability table.

Quantiles are recalibrated with a cross-conformal step (see conformal_offsets); --raw skips it.
Since 2026-11-18 (advisor comments of 2026-11-11) the default has no RI case weights, and the
P(RI) of record comes from a direct model of the 24 h change (riprob.py). Conformal offsets scaled
by the predicted spread (--scaled) were tried and score worse by CRPS than constant offsets per
lead band; the 2x RI weighting (--riweight) is kept only to reproduce the earlier runs.

Run from core/:  python quantile.py              (both evaluations, writes data/out/quantile_*.csv)
                 python quantile.py --noships    (base features only, for the ablation)
                 python quantile.py --riweight   (the pre-November configuration)
                 python quantile.py --scaled     (offsets in units of the predicted 10 to 90 range)
                 python quantile.py --carq       (CARQ operational initial intensity instead of best track)
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.model_selection import GroupKFold
from common import FEATURES, OUT, data_file

HERE = os.path.dirname(os.path.abspath(__file__))
Q = np.array([0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98])
QCOLS = [f"q{int(q * 100):02d}" for q in Q]
RI_KT = 30.0
RI_WEIGHT = False      # --riweight: 2x sample weight on RI cases (the pre-November setting)
SCALED = False         # --scaled: conformal offsets in units of predicted spread instead of knots
PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0)

def load(ships=True, carq=False):
    df = pd.read_csv(data_file("reference"), keep_default_na=False, na_values=[""])
    feats = list(FEATURES)
    if carq:
        df = swap_carq(df)
    if ships:
        sf = pd.read_csv(data_file("ships_features"), keep_default_na=False, na_values=[""])
        df = df.merge(sf, on=["init", "sid", "step"], how="left")
        feats += [c for c in sf.columns if c not in ("init", "sid", "step")]
    df["year"] = pd.to_datetime(df.init).dt.year
    return df, feats

def swap_carq(df):
    """Replace the best-track initial intensity and its 12 and 24 h tendencies by the CARQ lines of
    the ATCF a-deck (the operational estimate available at forecast time), where a CARQ exists.
    Cases without a CARQ keep best track; a column carq=1 marks the swapped ones."""
    o = pd.read_csv(os.path.join(HERE, "..", "data", "tc", "ofcl.csv"))
    c = o[o.tech == "CARQ"].pivot_table(index=["atcf_id", "init"], columns="lead", values=["vmax_kt", "pmin_hpa"])
    c.columns = [f"{a}_{b}" for a, b in c.columns]; c = c.reset_index()
    c["init"] = c.init.astype(str)
    bt = pd.read_csv(data_file("besttrack"), keep_default_na=False, na_values=[""])
    ids = bt[["SID", "USA_ATCF_ID"]].drop_duplicates().rename(columns={"SID": "sid", "USA_ATCF_ID": "atcf_id"})
    df = df.merge(ids, on="sid", how="left").merge(c, on=["atcf_id", "init"], how="left")
    ok = df["vmax_kt_0"].notna()
    df["carq"] = ok.astype(int)
    dev = np.abs(df.loc[ok, "vmax_kt_0"] - df.loc[ok, "obs0_wind"]).mean()
    df.loc[ok, "obs0_wind"] = df.loc[ok, "vmax_kt_0"]
    df.loc[ok, "obs0_pres"] = df.loc[ok, "pmin_hpa_0"].fillna(df.loc[ok, "obs0_pres"])
    df.loc[ok, "obs_dwind_12h"] = df.loc[ok, "vmax_kt_0"] - df.loc[ok, "vmax_kt_-12"]
    df.loc[ok, "obs_dwind_24h"] = df.loc[ok, "vmax_kt_0"] - df.loc[ok, "vmax_kt_-24"]
    df.loc[ok, "init_bias"] = df.loc[ok, "aifs0_vmax"] - df.loc[ok, "obs0_wind"]
    print(f"CARQ swapped in for {ok.sum()} of {len(df)} cases ({df.loc[ok].sid.nunique()} storms); "
          f"mean |CARQ - best track| at init {dev:.2f} kt")
    return df.drop(columns=[x for x in df.columns if x.startswith("vmax_kt_") or x.startswith("pmin_hpa_")])

def ri_weights(df):
    return np.where(df.ri.values == 1, 2.0, 1.0) if RI_WEIGHT else np.ones(len(df))

def fit_quantiles(Xtr, ytr, wtr, Xte, seed=0):
    """One GBM per quantile level, then sort across levels so quantiles do not cross."""
    P = np.zeros((len(Xte), len(Q)))
    for j, q in enumerate(Q):
        m = HGB(loss="quantile", quantile=q, random_state=seed, **PARAMS)
        m.fit(Xtr, ytr, sample_weight=wtr)
        P[:, j] = m.predict(Xte)
    return np.sort(P, axis=1)

BANDS = [(0, 48), (54, 96), (102, 168)]

def band_of(step):
    return np.digitize(step, [b[0] for b in BANDS[1:]])

def spread(P):
    """The model's own spread, used as the conformal scale: predicted 10 to 90 range, floored."""
    return np.maximum(P[:, list(Q).index(0.90)] - P[:, list(Q).index(0.10)], 2.0) if SCALED else np.ones(len(P))

def conformal_offsets(Xtr, ytr, wtr, gtr, step_tr, seed=0):
    """Per-level, per-lead-band corrections so that, on out-of-fold training storms, the fraction of
    observations below each predicted quantile equals its nominal level. Out-of-fold residuals come
    from an inner storm-grouped 5-fold (the cross-conformal recipe, level by level). Needed because
    quantile GBMs are far too narrow on unseen storms. Offsets by lead band because the spread grows
    with lead. With SCALED the residuals are divided by the model's predicted 10 to 90 range first, so
    the correction is proportional to how uncertain the model already is (wide for intense, fast
    changing storms, narrow for weak steady ones) instead of a constant number of knots."""
    R = np.zeros((len(Xtr), len(Q)))
    for a, b in GroupKFold(5).split(Xtr, ytr, gtr):
        Pb = fit_quantiles(Xtr[a], ytr[a], wtr[a], Xtr[b], seed)
        R[b] = (ytr[b, None] - Pb) / spread(Pb)[:, None]
    bands = band_of(step_tr)
    return np.array([[np.quantile(R[bands == k, j], q) for j, q in enumerate(Q)] for k in range(len(BANDS))])

def apply_offsets(P, off, step):
    return np.sort(P + off[band_of(step)] * spread(P)[:, None], axis=1)

def predict_cv(df, feats, seed=0, conformal=True):
    X = df[feats].values; y = df.res_v.values; w = ri_weights(df); g = df.sid.values
    P = np.zeros((len(df), len(Q)))
    st = df.step.values
    for tr, te in GroupKFold(5).split(X, y, g):
        Pte = fit_quantiles(X[tr], y[tr], w[tr], X[te], seed)
        P[te] = apply_offsets(Pte, conformal_offsets(X[tr], y[tr], w[tr], g[tr], st[tr], seed), st[te]) if conformal else Pte
    return P + df.vmax_kt.values[:, None]

def predict_holdout(df, feats, seed=0, conformal=True):
    tr = (df.year == 2025).values; te = ~tr
    X = df[feats].values; y = df.res_v.values; w = ri_weights(df); g = df.sid.values
    P = np.full((len(df), len(Q)), np.nan)
    st = df.step.values
    Pte = fit_quantiles(X[tr], y[tr], w[tr], X[te], seed)
    P[te] = apply_offsets(Pte, conformal_offsets(X[tr], y[tr], w[tr], g[tr], st[tr], seed), st[te]) if conformal else Pte
    return P + df.vmax_kt.values[:, None]

def point_gbm_cv(df, feats, seed=0):
    """The deterministic reference: MSE GBM on the residual, same CV, same features."""
    X = df[feats].values; y = df.res_v.values; w = ri_weights(df)
    p = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(X, y, df.sid.values):
        p[te] = HGB(random_state=seed, **PARAMS).fit(X[tr], y[tr], sample_weight=w[tr]).predict(X[te])
    return p + df.vmax_kt.values

def point_gbm_holdout(df, feats, seed=0):
    tr = (df.year == 2025).values; te = ~tr
    X = df[feats].values; y = df.res_v.values; w = ri_weights(df)
    p = np.full(len(df), np.nan)
    p[te] = HGB(random_state=seed, **PARAMS).fit(X[tr], y[tr], sample_weight=w[tr]).predict(X[te])
    return p + df.vmax_kt.values

def classifier_ri(df, feats, holdout=False, seed=0):
    """Direct baseline for P(RI): a gradient-boosted classifier on the RI label, same features."""
    from sklearn.ensemble import HistGradientBoostingClassifier as HGC
    X = df[feats].values; y = df.ri.values; p = np.full(len(df), np.nan)
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0)
    splits = [((df.year == 2025).values, (df.year != 2025).values)] if holdout \
        else GroupKFold(5).split(X, y, df.sid.values)
    for tr, te in splits:
        p[te] = HGC(random_state=seed, **kw).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p

# ---- scoring -------------------------------------------------------------------------------

def crps_from_quantiles(P, y):
    """CRPS of the predictive CDF that is piecewise linear through the nine (quantile, level) knots,
    with steps of mass Q[0] at the lowest and 1 - Q[-1] at the highest quantile. Exact integral of
    (F(x) - 1{x >= y})^2 dx. Until 2026-11-23 this was twice the unweighted mean pinball loss over the
    levels, which under-states the CRPS by about a third because the levels crowd the tails."""
    P = np.sort(P, axis=1); n = len(y)
    # knots: the quantiles plus the observation, with F interpolated at the observation
    Fy = np.array([np.interp(y[i], P[i], Q, left=0.0, right=1.0) for i in range(n)])
    X = np.concatenate([P, y[:, None]], axis=1)
    F = np.concatenate([np.tile(Q, (n, 1)), Fy[:, None]], axis=1)
    o = np.argsort(X, axis=1, kind="stable"); X = np.take_along_axis(X, o, 1); F = np.take_along_axis(F, o, 1)
    a, b, Fa, Fb = X[:, :-1], X[:, 1:], F[:, :-1], F[:, 1:]
    h = (a >= y[:, None]).astype(float)               # indicator 1{x >= y} on each segment
    dF = Fb - Fa
    lin = np.where(np.abs(dF) > 1e-12, ((Fb - h) ** 3 - (Fa - h) ** 3) / (3 * np.where(dF == 0, 1, dF)), (Fa - h) ** 2)
    keep = (a >= P[:, :1]) & (b <= P[:, -1:])           # segments between the lowest and highest quantile only
    inner = (keep * (b - a) * lin).sum(axis=1)
    below = np.maximum(P[:, 0] - y, 0)                # F = 0 left of the lowest quantile
    above = np.maximum(y - P[:, -1], 0)               # F = 1 right of the highest quantile
    return inner + below + above

def twcrps(P, y, t):
    """Threshold-weighted CRPS with weight 1{z >= t}: the CRPS of max(Y, t) against max(y, t)
    (Gneiting and Ranjan 2011). For a point forecast (one column) it reduces to |max(f,t) - max(y,t)|."""
    Pt, yt = np.maximum(P, t), np.maximum(y, t)
    if P.shape[1] == 1:
        return np.abs(Pt[:, 0] - yt)
    return crps_from_quantiles(Pt, yt)

def contingency(fc, ob):
    """Hit rate (POD), false-alarm ratio and frequency bias for binary forecast and observed events."""
    fc, ob = np.asarray(fc, bool), np.asarray(ob, bool)
    hits, miss, fa = (fc & ob).sum(), (~fc & ob).sum(), (fc & ~ob).sum()
    pod = hits / max(hits + miss, 1); far = fa / max(hits + fa, 1); bias = (hits + fa) / max(hits + miss, 1)
    return f"POD {pod:.2f}  FAR {far:.2f}  bias {bias:.2f}  (obs {hits + miss}, fc {hits + fa}, hits {hits})"

def cdf_from_quantiles(P, x):
    """P(Y <= x) by linear interpolation of the quantile function, flat tails beyond the outer levels."""
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = np.interp(x[i], P[i], Q, left=0.0, right=1.0)
    return out

def prob_ri(df, P):
    """P(wind at step >= wind 24 h earlier + 30 kt). The earlier wind is the observation at the init
    when step is 24, else the predicted median 24 h earlier for the same forecast."""
    med = pd.Series(P[:, list(Q).index(0.5)], index=df.index)
    key = df.set_index(["init", "sid", "step"]).index
    med_by_key = pd.Series(med.values, index=key)
    prev = pd.MultiIndex.from_arrays([df.init, df.sid, df.step - 24])
    base = med_by_key.reindex(prev).values
    base = np.where(df.step.values == 24, df.obs0_wind.values, base)
    ok = (df.step.values >= 24) & np.isfinite(base)
    p = np.full(len(df), np.nan)
    p[ok] = 1.0 - cdf_from_quantiles(P[ok], base[ok] + RI_KT)
    return p

def prob_ri_24(df, P):
    """Clean version: at step 24 only, P(wind >= observed init wind + 30 kt). No modelled base."""
    p = np.full(len(df), np.nan); ok = df.step.values == 24
    p[ok] = 1.0 - cdf_from_quantiles(P[ok], df.obs0_wind.values[ok] + RI_KT)
    return p

def brier_report(label, p, e):
    ok = np.isfinite(p); pr, e = p[ok], e[ok].astype(float); clim = e.mean()
    bs, bsc = ((pr - e) ** 2).mean(), ((clim - e) ** 2).mean()
    print(f"{label}: {ok.sum()} cases, base rate {clim:.3f}, Brier {bs:.4f}, climatology {bsc:.4f}, BSS {1 - bs / bsc:+.3f}")
    cut = pd.cut(pr, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01], right=False)
    rel = pd.DataFrame({"f": pr, "o": e}).groupby(cut, observed=True).agg(["mean", "count"])
    print("   bin          forecast  observed  n")
    for b, r in rel.iterrows():
        print(f"   {str(b):12s} {r[('f', 'mean')]:.3f}     {r[('o', 'mean')]:.3f}     {int(r[('o', 'count')])}")

def storm_boot(d, groups, n=2000, seed=0):
    """95% CI for the mean of d, resampling storms with replacement."""
    r = np.random.default_rng(seed); u = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in u}
    means = [d[np.concatenate([idx[g] for g in r.choice(u, len(u))])].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])

def fmt(x, ci):
    return f"{x:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"

def report(df, P, point, tag, pcls=None):
    m = np.isfinite(P[:, 0]) & np.isfinite(point)
    d = df[m].reset_index(drop=True); P = P[m]; point = point[m]
    y = d.bt_wind_kt.values; g = d.sid.values
    med = P[:, list(Q).index(0.5)]
    ae_med, ae_pt = np.abs(y - med), np.abs(y - point)
    crps = crps_from_quantiles(P, y)
    lo, hi = P[:, list(Q).index(0.10)], P[:, list(Q).index(0.90)]
    cover = ((y >= lo) & (y <= hi)).astype(float)
    print(f"\n=== {tag}: {len(d)} cases, {len(np.unique(g))} storms ===")
    print("subset      n    storms   MAE median   MAE point   median - point        CRPS   80% cover")
    for name, mask in [("all", np.ones(len(d), bool)), ("RI", d.ri.values == 1),
                       ("lead<=48", d.step.values <= 48), ("lead>96", d.step.values > 96),
                       ("obs>=96kt", y >= 96)]:
        diff = ae_med[mask] - ae_pt[mask]
        print(f"{name:10s} {mask.sum():5d} {len(np.unique(g[mask])):6d}   {ae_med[mask].mean():8.2f}   "
              f"{ae_pt[mask].mean():8.2f}   {fmt(diff.mean(), storm_boot(diff, g[mask])):>22s}   "
              f"{crps[mask].mean():6.2f}   {cover[mask].mean():6.1%}")
    print("CRPS and tail-weighted CRPS (threshold 96 kt), all cases; point forecasts score as MAE")
    for name, Pm in [("quantile model", P), ("point GBM", point[:, None]), ("raw AIFS", d.vmax_kt.values[:, None])]:
        print(f"  {name:15s} CRPS {crps_from_quantiles(Pm, y).mean() if Pm.shape[1] > 1 else np.abs(y - Pm[:, 0]).mean():6.2f}"
              f"   twCRPS>=96 {twcrps(Pm, y, 96).mean():6.2f}   twCRPS>=64 {twcrps(Pm, y, 64).mean():6.2f}")
    print("events at 96 and 120 kt (forecast = median or point >= threshold): hit rate, false-alarm ratio, bias")
    for thr in (96, 120):
        for name, f in [("median", med), ("point GBM", point), ("raw AIFS", d.vmax_kt.values)]:
            print(f"  {thr} kt {name:10s} " + contingency(f >= thr, y >= thr))
    qs = [0.9, 0.95, 0.99]
    print("tail quantiles (kt)   observed  median  q95-level  point")
    for q in qs:
        print(f"  q{q:<5}             {np.quantile(y, q):7.0f} {np.quantile(med, q):7.0f} "
              f"{np.quantile(P[:, list(Q).index(0.95)], q):9.0f} {np.quantile(point, q):7.0f}")
    p = prob_ri(d, P); brier_report("P(RI) from quantiles, step>=24, modelled base", p, d.ri.values)
    p24 = prob_ri_24(d, P); brier_report("P(RI) from quantiles, step 24 only, observed base", p24, d.ri.values)
    if pcls is not None:
        pc = pcls[m]; brier_report("P(RI) direct classifier, step>=24", np.where(d.step.values >= 24, pc, np.nan), d.ri.values)
        brier_report("P(RI) direct classifier, step 24 only", np.where(d.step.values == 24, pc, np.nan), d.ri.values)
    return pd.DataFrame({"init": d.init, "sid": d.sid, "step": d.step, "bt_wind_kt": y, "point": point,
                         **{f"q{int(q * 100):02d}": P[:, j] for j, q in enumerate(Q)}, "p_ri": p, "p_ri_24": p24,
                         "p_ri_cls": pcls[m] if pcls is not None else np.nan})

def rescore(df, tag):
    """Re-run report() on saved predictions (no refit), e.g. after a change to the scoring code."""
    for split, name in [("cv", "storm-grouped CV"), ("holdout", "temporal holdout 2025->2026")]:
        s = pd.read_csv(f"{OUT}/quantile_{split}_{tag}.csv", keep_default_na=False, na_values=[""])
        keys = ["init", "sid", "step"]
        extra = df.drop(columns=[c for c in df.columns if c in s.columns and c not in keys])
        d = s.merge(extra, on=keys, how="left")
        assert d.ri.notna().all(), "saved predictions do not match the current dataset"
        report(d, d[QCOLS].values, d.point.values, f"{name}, {tag} (rescored)", d.p_ri_cls.values)

if __name__ == "__main__":
    ships = "--noships" not in sys.argv; conf = "--raw" not in sys.argv
    RI_WEIGHT = "--riweight" in sys.argv; SCALED = "--scaled" in sys.argv; carq = "--carq" in sys.argv
    df, feats = load(ships, carq)
    tag = ("ships" if ships else "base") + ("" if conf else "_raw") + ("_riw" if RI_WEIGHT else "") \
        + ("_scaled" if SCALED else "") + ("_carq" if carq else "")
    print(f"config: {tag}")
    os.makedirs(OUT, exist_ok=True)
    if "--rescore" in sys.argv:
        rescore(df, tag); sys.exit()
    P = predict_cv(df, feats, conformal=conf); pt = point_gbm_cv(df, feats); pc = classifier_ri(df, feats)
    report(df, P, pt, f"storm-grouped CV, {tag}", pc).to_csv(f"{OUT}/quantile_cv_{tag}.csv", index=False)
    P = predict_holdout(df, feats, conformal=conf); pt = point_gbm_holdout(df, feats); pc = classifier_ri(df, feats, holdout=True)
    report(df, P, pt, f"temporal holdout 2025->2026, {tag}", pc).to_csv(f"{OUT}/quantile_holdout_{tag}.csv", index=False)
