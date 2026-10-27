"""The fair baseline for eta-learning: a tuned MSE MLP plus post-hoc quantile mapping to nu_0.

The advisor's point (email of 2026-09-29): the plain MSE MLP in eta.ipynb has no early stopping
and overshoots the tail, so part of eta's gain may be regularization. The fair comparison is
  (a) the same MLP with early stopping and weight decay chosen on held-out training storms, and
  (b) that model followed by quantile mapping of its corrected winds onto the training fold's
      observed-wind distribution nu_0 (the classic marginal correction).
If eta still beats (b), it is doing something beyond calibration.

All four models (ERM 600 iters, ERM tuned, ERM tuned + QM, eta) are trained here with the same
folds and seeds as eta.ipynb (64-64 ReLU, full-batch Adam 3e-3, GroupKFold(5) on sid, 3 seeds).
Differences carry paired storm-bootstrap 95% CIs.

Run from core/:  python fair_baseline.py      (writes data/out/fair_baseline_cv.csv)
"""
import os
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from common import FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "out")
SEEDS = (0, 1, 2)
LEVELS = torch.linspace(0.95, 0.999, 25)
MAX_IT, CHECK = 3000, 25
WD_GRID = (0.0, 1e-4, 1e-3)

class MLP(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def grad_norm(loss, model):
    g = torch.autograd.grad(loss, list(model.parameters()), retain_graph=True, allow_unused=True)
    return torch.sqrt(sum((t ** 2).sum() for t in g if t is not None))

def train(X, y, seed, iters, wd=0.0, eta=None, Xval=None, yval=None):
    """Plain MSE on the residual; with eta=(vmax, ref_q) adds the tail penalty of eta.ipynb.
    With a validation set, returns the iteration count with the lowest validation MSE."""
    torch.manual_seed(seed)
    m = MLP(X.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=wd)
    lam, best, best_it = 0.0, np.inf, iters
    for it in range(iters):
        opt.zero_grad()
        out = m(X); mse = ((out - y) ** 2).mean(); loss = mse
        if eta is not None:
            vmax, ref_q = eta
            pen = (torch.quantile(vmax + out, LEVELS) - ref_q).abs().mean()
            if it % 100 == 0:
                lam = float(grad_norm(mse, m) / (grad_norm(pen, m) + 1e-8))
            loss = mse + lam * pen
        loss.backward(); opt.step()
        if Xval is not None and (it + 1) % CHECK == 0:
            with torch.no_grad():
                v = ((m(Xval) - yval) ** 2).mean().item()
            if v < best:
                best, best_it = v, it + 1
    return m, best_it, best

def quantile_map(train_pred, train_obs, test_pred):
    """Map test predictions through F_pred(train) then F_obs(train)^-1: the marginal of the mapped
    predictions on the training fold equals nu_0 by construction."""
    sp, so = np.sort(train_pred), np.sort(train_obs)
    u = np.searchsorted(sp, test_pred, side="right") / len(sp)
    return np.quantile(so, np.clip(u, 0, 1))

def main():
    df = pd.read_csv(os.path.join(HERE, "data", "reference.csv"), keep_default_na=False, na_values=[""])
    X_all, y_all, g = df[FEATURES], df.res_v.values, df.sid.values
    cols = ["erm600", "erm_tuned", "erm_tuned_qm", "eta"]
    pred = {c: np.zeros(len(df)) for c in cols}
    chosen = []
    for k, (tr, te) in enumerate(GroupKFold(5).split(X_all, y_all, g)):
        Xtr, Xte = X_all.iloc[tr], X_all.iloc[te]
        med = Xtr.median(); Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
        mu, sd = Xtr.mean(), Xtr.std(); Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        Xtr_t, Xte_t = torch.tensor(Xtr.values).float(), torch.tensor(Xte.values).float()
        ytr_t = torch.tensor(y_all[tr]).float()
        vmax_tr = torch.tensor(df.vmax_kt.values[tr]).float()
        obs_tr = df.bt_wind_kt.values[tr]
        ref_q = torch.quantile(torch.tensor(obs_tr).float(), LEVELS)
        # inner split by storm for early stopping and weight decay
        a, b = next(GroupKFold(5).split(Xtr, y_all[tr], g[tr]))
        for seed in SEEDS:
            m, _, _ = train(Xtr_t, ytr_t, seed, 600)
            with torch.no_grad(): pred["erm600"][te] += m(Xte_t).numpy() / len(SEEDS)
            best = min((train(Xtr_t[a], ytr_t[a], seed, MAX_IT, wd, Xval=Xtr_t[b], yval=ytr_t[b])[1:], wd)
                       for wd in WD_GRID)
            (it_best, _), wd_best = best
            chosen.append((k, seed, it_best, wd_best))
            m, _, _ = train(Xtr_t, ytr_t, seed, it_best, wd_best)
            with torch.no_grad():
                p_te = m(Xte_t).numpy(); p_tr = m(Xtr_t).numpy()
            pred["erm_tuned"][te] += p_te / len(SEEDS)
            corr_tr, corr_te = df.vmax_kt.values[tr] + p_tr, df.vmax_kt.values[te] + p_te
            pred["erm_tuned_qm"][te] += (quantile_map(corr_tr, obs_tr, corr_te) - df.vmax_kt.values[te]) / len(SEEDS)
            m, _, _ = train(Xtr_t, ytr_t, seed, 600, eta=(vmax_tr, ref_q))
            with torch.no_grad(): pred["eta"][te] += m(Xte_t).numpy() / len(SEEDS)
        print(f"fold {k + 1}/5 done; early stopping picks: "
              + ", ".join(f"{it}it wd={wd:g}" for _, s, it, wd in chosen[-len(SEEDS):]), flush=True)
    for c in cols:
        df[c] = df.vmax_kt + pred[c]
    df["gbm"] = df.corr_vmax_kt
    report(df, cols + ["gbm"])
    os.makedirs(OUT, exist_ok=True)
    df[["init", "sid", "step", "ri", "bt_wind_kt", "vmax_kt"] + cols + ["gbm"]].to_csv(f"{OUT}/fair_baseline_cv.csv", index=False)

def storm_boot(d, groups, n=2000, seed=0):
    r = np.random.default_rng(seed); u = np.unique(groups)
    idx = {x: np.where(groups == x)[0] for x in u}
    means = [d[np.concatenate([idx[x] for x in r.choice(u, len(u))])].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])

def report(df, cols):
    y, g, ri = df.bt_wind_kt.values, df.sid.values, df.ri.values == 1
    print("\nmodel          MAE all   MAE RI   q0.90  q0.95  q0.99   (observed: "
          f"{np.quantile(y, .9):.0f} {np.quantile(y, .95):.0f} {np.quantile(y, .99):.0f})")
    for c in cols:
        e = np.abs(y - df[c].values)
        print(f"{c:13s} {e.mean():8.2f} {e[ri].mean():8.2f}   {np.quantile(df[c], .9):5.0f}  "
              f"{np.quantile(df[c], .95):5.0f}  {np.quantile(df[c], .99):5.0f}")
    print("\npaired differences in MAE (kt), storm bootstrap 95% CI")
    pairs = [("eta", "erm600"), ("eta", "erm_tuned"), ("eta", "erm_tuned_qm"), ("erm_tuned_qm", "erm_tuned"),
             ("eta", "gbm"), ("erm_tuned_qm", "gbm")]
    for a, b in pairs:
        for name, mask in [("all", np.ones(len(df), bool)), ("RI", ri)]:
            d = np.abs(y - df[a].values)[mask] - np.abs(y - df[b].values)[mask]
            ci = storm_boot(d, g[mask])
            print(f"  {a:13s} - {b:13s} {name:4s} {d.mean():+6.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")

if __name__ == "__main__":
    main()
