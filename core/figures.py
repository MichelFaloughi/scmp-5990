"""Figures for the report, from the prediction files written by quantile.py and fair_baseline.py.

Run from core/:  python figures.py      (writes ../report/figures/*.png)
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "out")
FIG = os.path.join(HERE, "..", "report", "figures")
Q = [0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98]
QCOLS = [f"q{int(q * 100):02d}" for q in Q]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

def load(name):
    return pd.read_csv(f"{OUT}/{name}.csv", keep_default_na=False, na_values=[""])

def fig_fair_baseline():
    d = load("fair_baseline_cv")
    bands = [(6, 24), (30, 48), (54, 72), (78, 120), (126, 168)]
    models = [("erm600", "plain MLP, 600 it"), ("erm_tuned", "tuned MLP"), ("erm_tuned_qm", "tuned MLP + quantile mapping"),
              ("eta", "eta"), ("gbm", "GBM")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, mask, title in [(axes[0], np.ones(len(d), bool), "all cases"), (axes[1], d.ri.values == 1, "rapid intensification cases")]:
        for c, lab in models:
            mae = [np.abs(d.bt_wind_kt - d[c])[mask & (d.step >= a) & (d.step <= b)].mean() for a, b in bands]
            ax.plot([np.mean(b) for b in bands], mae, marker="o", label=lab)
        ax.set_xlabel("lead time (h)"); ax.set_ylabel("MAE (kt)"); ax.set_title(title)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fair_baseline_by_lead.png", dpi=200); plt.close(fig)

def fig_tails():
    d = load("fair_baseline_cv")
    qs = np.linspace(0.5, 0.999, 60)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    obs = np.quantile(d.bt_wind_kt, qs)
    for c, lab in [("erm600", "plain MLP"), ("erm_tuned", "tuned MLP"), ("erm_tuned_qm", "tuned MLP + QM"), ("eta", "eta"), ("gbm", "GBM")]:
        ax.plot(obs, np.quantile(d[c], qs), label=lab)
    ax.plot([60, 160], [60, 160], "k--", lw=0.8)
    ax.set_xlabel("observed wind quantile (kt)"); ax.set_ylabel("forecast wind quantile (kt)")
    ax.set_title("Upper tail, quantiles 0.5 to 0.999"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/tail_qq.png", dpi=200); plt.close(fig)

def fig_coverage():
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    for name, lab, ls in [("quantile_cv_ships_raw", "CV, raw quantiles", ":"), ("quantile_cv_ships", "CV, conformal", "-"),
                          ("quantile_holdout_ships", "holdout 2026, conformal", "--")]:
        if not os.path.exists(f"{OUT}/{name}.csv"):
            continue
        d = load(name)
        ax.plot(Q, [(d.bt_wind_kt <= d[c]).mean() for c in QCOLS], marker="o", ls=ls, label=lab)
    ax.plot([0, 1], [0, 1], "k-", lw=0.8)
    ax.set_xlabel("nominal quantile level"); ax.set_ylabel("fraction of observations below")
    ax.set_title("Calibration of the predicted quantiles"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/quantile_calibration.png", dpi=200); plt.close(fig)

def fig_reliability():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]
    for ax, name, title in [(axes[0], "quantile_cv_ships", "storm-grouped CV"), (axes[1], "quantile_holdout_ships", "temporal holdout 2026")]:
        d = load(name); ri = pd.read_csv(f"{HERE}/data/reference.csv", keep_default_na=False, na_values=[""])[["init", "sid", "step", "ri"]]
        d = d.merge(ri, on=["init", "sid", "step"])
        for col, lab in [("p_ri", "quantiles, step >= 24"), ("p_ri_24", "quantiles, step 24"), ("p_ri_cls", "direct classifier, step >= 24")]:
            if col not in d:
                continue
            m = d[col].notna() & (d.step >= 24)
            cut = pd.cut(d.loc[m, col], bins, right=False)
            g = d[m].groupby(cut, observed=True).agg(f=(col, "mean"), o=("ri", "mean"), n=("ri", "size"))
            ax.plot(g.f, g.o, marker="o", label=lab)
            for _, r in g.iterrows():
                ax.annotate(int(r.n), (r.f, r.o), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.plot([0, 0.8], [0, 0.8], "k--", lw=0.8); ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8)
        ax.set_xlabel("forecast P(RI)"); ax.set_ylabel("observed RI frequency"); ax.set_title(title)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/ri_reliability.png", dpi=200); plt.close(fig)

def fig_storm_example(sid_name="LOWELL"):
    d = load("quantile_cv_ships"); ref = pd.read_csv(f"{HERE}/data/reference.csv", keep_default_na=False, na_values=[""])
    d = d.merge(ref[["init", "sid", "step", "name", "valid", "vmax_kt"]], on=["init", "sid", "step"])
    s = d[d.name == sid_name]
    if s.empty:
        return
    inits = sorted(s.init.unique()); pick = inits[len(inits) // 3]
    f = s[s.init == pick].sort_values("step"); t = pd.to_datetime(f.valid)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.fill_between(t, f.q10, f.q90, alpha=0.25, label="10 to 90% interval")
    ax.fill_between(t, f.q25, f.q75, alpha=0.35, label="25 to 75% interval")
    ax.plot(t, f.q50, label="median"); ax.plot(t, f.point, ls="--", label="point GBM")
    ax.plot(t, f.vmax_kt, ls=":", color="gray", label="raw AIFS")
    ax.plot(t, f.bt_wind_kt, "k", marker=".", label="observed")
    ax.set_ylabel("max wind (kt)"); ax.set_title(f"{sid_name.title()}, forecast from {pick[:13]}Z (out-of-fold)")
    ax.legend(fontsize=8, frameon=False, ncol=3); fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(f"{FIG}/storm_example_{sid_name.lower()}.png", dpi=200); plt.close(fig)

if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_fair_baseline(); fig_tails(); fig_coverage(); fig_reliability(); fig_storm_example("LOWELL"); fig_storm_example("ERIN")
    print("figures written to", FIG)
