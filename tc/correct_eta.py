"""Extreme Event Aware (eta-) learning for the AIFS intensity correction.

Implements Chang & Sapsis (2026), arXiv:2510.19161, on the same task as correct.py:
an MLP predicts the residual (best track - AIFS vmax), trained with

    L = MSE(residual) + lambda * W1_tail( dist(corrected vmax), dist(observed vmax) )

where the observable g(phi, x) = AIFS vmax + predicted residual (the corrected wind),
W1_tail is the quantile form of the 1-Wasserstein distance restricted to q in [tau, 1],
the reference nu_0 is the empirical best-track wind distribution of the training fold,
and lambda is set by gradient-norm balancing. Same storm-grouped 5-fold CV as the GBM.

An identical MLP trained with MSE only (ERM) isolates the effect of the regularizer.

Output: data/tc/aifs_tracks_eta.csv (adds eta_vmax_kt, erm_vmax_kt) + comparison tables.
Usage : python tc/correct_eta.py [--iters 3000] [--tau 0.95]
"""
import os, sys, time

import numpy as np, pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from correct import build_features, FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITERS = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 3000
TAU = float(sys.argv[sys.argv.index("--tau") + 1]) if "--tau" in sys.argv else 0.95
N_Q, SEEDS, HID = 25, (0, 1, 2), 64
torch.set_num_threads(4)


def mlp(d_in, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d_in, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(),
                         nn.Linear(HID, 1))


def grad_norm(loss, model):
    g = torch.autograd.grad(loss, [p for p in model.parameters()], retain_graph=True,
                            create_graph=False, allow_unused=True)
    return torch.sqrt(sum((x ** 2).sum() for x in g if x is not None))


def fit(X, y, vmax0, bt_wind, seed, use_eta):
    """X standardized features, y residual target, vmax0 raw AIFS wind (for the observable),
    bt_wind observed winds defining nu_0. Full-batch Adam."""
    model = mlp(X.shape[1], seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    X, y, vmax0 = map(torch.as_tensor, (X, y, vmax0))
    X, y, vmax0 = X.float(), y.float(), vmax0.float()
    q = torch.linspace(TAU, 0.999, N_Q)
    ref_q = torch.quantile(torch.as_tensor(bt_wind).float(), q)   # F^-1_{nu_0}(q_i), fixed
    lam = 0.0
    for it in range(ITERS):
        opt.zero_grad()
        pred = model(X).squeeze(-1)
        l_erm = ((pred - y) ** 2).mean()
        if use_eta:
            corr = vmax0 + pred                                   # observable g(phi(x))
            l_w1 = (torch.quantile(corr, q) - ref_q).abs().mean() # eq. 15, tail quantiles
            if it % 100 == 0:                                     # gradient-norm balancing
                lam = float(grad_norm(l_erm, model) / (grad_norm(l_w1, model) + 1e-8))
            loss = l_erm + lam * l_w1
        else:
            loss = l_erm
        loss.backward()
        opt.step()
    return model


def predict(model, X):
    with torch.no_grad():
        return model(torch.as_tensor(X).float()).squeeze(-1).numpy()


def cv_predict(lab, use_eta):
    X_all = lab[FEATURES].values.astype(np.float64)
    y_all = lab.res_v.values
    pred = np.zeros(len(lab))
    for tr_i, te_i in GroupKFold(n_splits=5).split(X_all, y_all, lab.sid.values):
        med = np.nanmedian(X_all[tr_i], axis=0)                   # impute + standardize on train only
        mu, sd = None, None
        Xtr = np.where(np.isnan(X_all[tr_i]), med, X_all[tr_i])
        Xte = np.where(np.isnan(X_all[te_i]), med, X_all[te_i])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        p = np.zeros(len(te_i))
        for s in SEEDS:
            m = fit(Xtr, y_all[tr_i], lab.vmax_kt.values[tr_i], lab.bt_wind_kt.values[tr_i], s, use_eta)
            p += predict(m, Xte) / len(SEEDS)
        pred[te_i] = p
    return pred


def report(lab, cols):
    leads = [24, 48, 72, 96, 120]
    for title, mask in [("all cases", np.ones(len(lab), bool)), ("RI cases", lab.ri.values == 1)]:
        d = lab[mask]
        print(f"\n--- MAE kt, {title} (n={len(d)}) ---")
        print(f"{'lead':>6}" + "".join(f"{n:>12}" for n in cols))
        for l in leads:
            g = d[d.step == l]
            print(f"{l:>6}" + "".join(f"{(g[c] - g.bt_wind_kt).abs().mean():>12.1f}" for c in cols.values()))
        print(f"{'all':>6}" + "".join(f"{(d[c] - d.bt_wind_kt).abs().mean():>12.1f}" for c in cols.values()))
    print("\n--- tail check: quantiles of forecast vmax vs observed (all cases) ---")
    qs = [0.9, 0.95, 0.99, 0.999]
    print(f"{'q':>6}" + "".join(f"{n:>12}" for n in list(cols) + ["observed"]))
    for q in qs:
        vals = [lab[c].quantile(q) for c in cols.values()] + [lab.bt_wind_kt.quantile(q)]
        print(f"{q:>6}" + "".join(f"{v:>12.1f}" for v in vals))


if __name__ == "__main__":
    tr = pd.read_csv(f"{ROOT}/data/tc/aifs_tracks.csv", parse_dates=["init", "valid"], keep_default_na=False, na_values=[""])
    bt = pd.read_csv(f"{ROOT}/data/tc/besttrack.csv", parse_dates=["ISO_TIME"], keep_default_na=False, na_values=[""])
    gbm = pd.read_csv(f"{ROOT}/data/tc/aifs_tracks_corrected.csv", parse_dates=["init", "valid"], keep_default_na=False, na_values=[""])
    df = build_features(tr, bt)
    lab = df.dropna(subset=["bt_wind_kt", "obs0_wind"]).copy()
    lab = lab[lab.step > 0].reset_index(drop=True)
    lab["res_v"] = lab.bt_wind_kt - lab.vmax_kt
    lab = lab.merge(gbm[["sid", "init", "step", "corr_vmax_kt"]], on=["sid", "init", "step"], how="left")
    print(f"cases: {len(lab)}  storms: {lab.sid.nunique()}  tau={TAU}  iters={ITERS}  seeds={len(SEEDS)}")

    t0 = time.time()
    lab["erm_vmax_kt"] = lab.vmax_kt + cv_predict(lab, use_eta=False)
    print(f"ERM MLP done ({time.time() - t0:.0f}s)")
    t0 = time.time()
    lab["eta_vmax_kt"] = lab.vmax_kt + cv_predict(lab, use_eta=True)
    print(f"eta MLP done ({time.time() - t0:.0f}s)")

    lab.to_csv(f"{ROOT}/data/tc/aifs_tracks_eta.csv", index=False)
    print("wrote data/tc/aifs_tracks_eta.csv")
    report(lab, {"AIFS raw": "vmax_kt", "GBM": "corr_vmax_kt", "MLP ERM": "erm_vmax_kt", "MLP eta": "eta_vmax_kt"})
