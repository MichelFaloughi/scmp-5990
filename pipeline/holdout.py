"""Temporal holdout: train on 2025 inits, test on 2026 (point 4 of the
2026-09-16 checklist). This is the operational number: random storm folds let a
model train on storms from the season it is tested on; a forecaster cannot.

It is also the version-transfer probe (the bonus question): every 2025 run is
AIFS-Single 1.0 and the 2026 hurricane season is 2.0 (switch 2026-05-12), so
"train 2025, test 2026" asks whether a correction learned on one base-model
version survives the next. The aifs_v2 feature is constant within each side,
so the models cannot use it here.

Usage: python pipeline/holdout.py [--iters 600]
"""
import os, sys, time

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import correct_eta as ce
from correct import build_features, FEATURES, RI_WEIGHT
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ce.ITERS = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 600
GBM_SEEDS, MLP_SEEDS = (0, 1, 2, 3, 4), (0, 1, 2)


def gbm_predict(tr_d, te_d):
    X, y = tr_d[FEATURES].values, tr_d.res_v.values
    w = np.where(tr_d.ri.values == 1, RI_WEIGHT, 1.0)
    p = np.zeros(len(te_d))
    for s in GBM_SEEDS:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=s)
        m.fit(X, y, sample_weight=w)
        p += m.predict(te_d[FEATURES].values) / len(GBM_SEEDS)
    return p


def mlp_predict(tr_d, te_d, use_eta):
    X_tr, X_te = tr_d[FEATURES].values.astype(np.float64), te_d[FEATURES].values.astype(np.float64)
    med = np.nanmedian(X_tr, axis=0)
    X_tr, X_te = np.where(np.isnan(X_tr), med, X_tr), np.where(np.isnan(X_te), med, X_te)
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd < 1e-6] = 1.0   # constant-in-train features (e.g. aifs_v2) would explode the test side
    X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
    p = np.zeros(len(te_d))
    for s in MLP_SEEDS:
        m = ce.fit(X_tr, tr_d.res_v.values, tr_d.vmax_kt.values, tr_d.bt_wind_kt.values, s, use_eta)
        p += ce.predict(m, X_te) / len(MLP_SEEDS)
    return p


if __name__ == "__main__":
    tr = pd.read_csv(f"{ROOT}/data/tc/aifs_tracks.csv", parse_dates=["init", "valid"], keep_default_na=False, na_values=[""])
    bt = pd.read_csv(f"{ROOT}/data/tc/besttrack.csv", parse_dates=["ISO_TIME"], keep_default_na=False, na_values=[""])
    lab = build_features(tr, bt).dropna(subset=["bt_wind_kt", "obs0_wind"]).copy()
    lab = lab[lab.step > 0].reset_index(drop=True)
    lab["res_v"] = lab.bt_wind_kt - lab.vmax_kt

    tr_d = lab[lab.init.dt.year == 2025]
    te_d = lab[lab.init.dt.year == 2026].copy()
    print(f"train 2025: {len(tr_d)} cases, {tr_d.sid.nunique()} storms, RI {tr_d.ri.sum()}")
    print(f"test  2026: {len(te_d)} cases, {te_d.sid.nunique()} storms, RI {te_d.ri.sum()}"
          f"  (v2.0 inits: {(te_d.aifs_v2 == 1).mean() * 100:.0f}%)")

    t0 = time.time()
    te_d["gbm"] = te_d.vmax_kt + gbm_predict(tr_d, te_d)
    te_d["erm"] = te_d.vmax_kt + mlp_predict(tr_d, te_d, use_eta=False)
    te_d["eta"] = te_d.vmax_kt + mlp_predict(tr_d, te_d, use_eta=True)
    print(f"trained + predicted ({time.time() - t0:.0f}s)  iters={ce.ITERS}")
    te_d[["sid", "init", "step", "ri", "bt_wind_kt", "vmax_kt", "gbm", "erm", "eta"]] \
        .to_csv(f"{ROOT}/data/tc/holdout.csv", index=False)
    print("wrote data/tc/holdout.csv")

    cols = {"AIFS raw": "vmax_kt", "GBM": "gbm", "MLP ERM": "erm", "MLP eta": "eta"}
    for title, mask in [("all 2026 cases", np.ones(len(te_d), bool)), ("2026 RI cases", te_d.ri.values == 1)]:
        d = te_d[mask]
        print(f"\n--- MAE kt, {title} (n={len(d)}) ---")
        for name, c in cols.items():
            print(f"  {name:>9}: {(d[c] - d.bt_wind_kt).abs().mean():5.1f}")
    print(f"\n--- tail: q0.99 of forecast vmax on 2026 (obs {te_d.bt_wind_kt.quantile(0.99):.0f} kt) ---")
    for name, c in cols.items():
        print(f"  {name:>9}: {te_d[c].quantile(0.99):5.1f}")
