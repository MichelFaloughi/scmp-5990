"""Probabilistic intensity correction: quantile regression with gradient-boosted trees.

Predicts quantiles of the residual (observed minus AIFS wind), with the base features plus the
SHIPS-style environmental predictors. Evaluated two ways: storm-grouped 5-fold CV and a temporal
holdout (train on 2025 inits, test on 2026). Every comparison carries a paired storm-bootstrap CI.

Reported: MAE of the median, CRPS (quantile approximation), 80% interval coverage, tail quantiles,
and P(RI) scored with Brier skill and a reliability table.

Quantiles are recalibrated with a cross-conformal step (see conformal_offsets); --raw skips it.

Run from core/:  python quantile.py            (both evaluations, writes data/out/quantile_*.csv)
                 python quantile.py --noships  (base features only, for the ablation)
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.model_selection import GroupKFold
from common import FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "out")
Q = np.array([0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98])
RI_KT = 30.0
PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0)

def load(ships=True):
    df = pd.read_csv(os.path.join(HERE, "data", "reference.csv"), keep_default_na=False, na_values=[""])
    feats = list(FEATURES)
    if ships:
        sf = pd.read_csv(os.path.join(HERE, "data", "ships_features.csv"), keep_default_na=False, na_values=[""])
        df = df.merge(sf, on=["init", "sid", "step"], how="left")
        feats += [c for c in sf.columns if c not in ("init", "sid", "step")]
    df["year"] = pd.to_datetime(df.init).dt.year
    return df, feats

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

def conformal_offsets(Xtr, ytr, wtr, gtr, step_tr, seed=0):
    """Per-level, per-lead-band additive corrections so that, on out-of-fold training storms, the
    fraction of observations below each predicted quantile equals its nominal level. Out-of-fold
    residuals come from an inner storm-grouped 5-fold (the cross-conformal recipe, level by level).
    Needed because quantile GBMs are far too narrow on unseen storms. Offsets by lead band because
    the spread grows with lead; one global offset over-widens short leads and the upper tail."""
    R = np.zeros((len(Xtr), len(Q)))
    for a, b in GroupKFold(5).split(Xtr, ytr, gtr):
        R[b] = ytr[b, None] - fit_quantiles(Xtr[a], ytr[a], wtr[a], Xtr[b], seed)
    bands = band_of(step_tr)
    return np.array([[np.quantile(R[bands == k, j], q) for j, q in enumerate(Q)] for k in range(len(BANDS))])

def apply_offsets(P, off, step):
    return np.sort(P + off[band_of(step)], axis=1)

def predict_cv(df, feats, seed=0, conformal=True):
    X = df[feats].values; y = df.res_v.values; w = np.where(df.ri.values == 1, 2.0, 1.0); g = df.sid.values
    P = np.zeros((len(df), len(Q)))
    st = df.step.values
    for tr, te in GroupKFold(5).split(X, y, g):
        Pte = fit_quantiles(X[tr], y[tr], w[tr], X[te], seed)
        P[te] = apply_offsets(Pte, conformal_offsets(X[tr], y[tr], w[tr], g[tr], st[tr], seed), st[te]) if conformal else Pte
    return P + df.vmax_kt.values[:, None]

def predict_holdout(df, feats, seed=0, conformal=True):
    tr = (df.year == 2025).values; te = ~tr
    X = df[feats].values; y = df.res_v.values; w = np.where(df.ri.values == 1, 2.0, 1.0); g = df.sid.values
    P = np.full((len(df), len(Q)), np.nan)
    st = df.step.values
    Pte = fit_quantiles(X[tr], y[tr], w[tr], X[te], seed)
    P[te] = apply_offsets(Pte, conformal_offsets(X[tr], y[tr], w[tr], g[tr], st[tr], seed), st[te]) if conformal else Pte
    return P + df.vmax_kt.values[:, None]

def point_gbm_cv(df, feats, seed=0):
    """The deterministic reference: MSE GBM on the residual, same CV, same features."""
    X = df[feats].values; y = df.res_v.values; w = np.where(df.ri.values == 1, 2.0, 1.0)
    p = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(X, y, df.sid.values):
        p[te] = HGB(random_state=seed, **PARAMS).fit(X[tr], y[tr], sample_weight=w[tr]).predict(X[te])
    return p + df.vmax_kt.values

def point_gbm_holdout(df, feats, seed=0):
    tr = (df.year == 2025).values; te = ~tr
    X = df[feats].values; y = df.res_v.values; w = np.where(df.ri.values == 1, 2.0, 1.0)
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
    """Mean pinball loss over the levels, times 2: a standard approximation of the CRPS."""
    d = y[:, None] - P
    pin = np.maximum(Q * d, (Q - 1) * d)
    return 2 * pin.mean(axis=1)

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

if __name__ == "__main__":
    ships = "--noships" not in sys.argv; conf = "--raw" not in sys.argv
    df, feats = load(ships)
    tag = ("ships" if ships else "base") + ("" if conf else "_raw")
    os.makedirs(OUT, exist_ok=True)
    P = predict_cv(df, feats, conformal=conf); pt = point_gbm_cv(df, feats); pc = classifier_ri(df, feats)
    report(df, P, pt, f"storm-grouped CV, {tag}", pc).to_csv(f"{OUT}/quantile_cv_{tag}.csv", index=False)
    P = predict_holdout(df, feats, conformal=conf); pt = point_gbm_holdout(df, feats); pc = classifier_ri(df, feats, holdout=True)
    report(df, P, pt, f"temporal holdout 2025->2026, {tag}", pc).to_csv(f"{OUT}/quantile_holdout_{tag}.csv", index=False)
