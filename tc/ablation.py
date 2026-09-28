"""Ablation grid: architecture (GBM / MLP) x RI weighting (1x / 2x) x eta penalty.

Point 1 of the 2026-09-16 checklist: the original eta-vs-GBM comparison changed
architecture, loss, and RI weighting at once. Here every cell is trained fresh
under the same storm-grouped 5-fold CV so each factor varies one at a time:

    GBM w1, GBM w2, MLP ERM w1, MLP ERM w2, MLP eta w1, MLP eta w2

Output: data/tc/ablation.csv (per-case predictions, one column per cell)
Usage : python tc/ablation.py [--iters 600]
"""
import os, sys, time

import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import correct_eta as ce
from correct import build_features, fit_predict_cv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ce.ITERS = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 600

CELLS = [("gbm_w1", "GBM w1"), ("gbm_w2", "GBM w2"),
         ("erm_w1", "MLP ERM w1"), ("erm_w2", "MLP ERM w2"),
         ("eta_w1", "MLP eta w1"), ("eta_w2", "MLP eta w2")]

if __name__ == "__main__":
    tr = pd.read_csv(f"{ROOT}/data/tc/aifs_tracks.csv", parse_dates=["init", "valid"], keep_default_na=False, na_values=[""])
    bt = pd.read_csv(f"{ROOT}/data/tc/besttrack.csv", parse_dates=["ISO_TIME"], keep_default_na=False, na_values=[""])
    lab = build_features(tr, bt).dropna(subset=["bt_wind_kt", "obs0_wind"]).copy()
    lab = lab[lab.step > 0].reset_index(drop=True)
    lab["res_v"] = lab.bt_wind_kt - lab.vmax_kt
    print(f"cases: {len(lab)}  storms: {lab.sid.nunique()}  iters={ce.ITERS}")

    t0 = time.time()
    lab["gbm_w1"] = lab.vmax_kt + fit_predict_cv(lab, "res_v", lab.sid.values, ri_weight=1.0)
    lab["gbm_w2"] = lab.vmax_kt + fit_predict_cv(lab, "res_v", lab.sid.values, ri_weight=2.0)
    print(f"GBM cells done ({time.time() - t0:.0f}s)")
    t0 = time.time()
    lab["erm_w1"] = lab.vmax_kt + ce.cv_predict(lab, use_eta=False)
    lab["erm_w2"] = lab.vmax_kt + ce.cv_predict(lab, use_eta=False, ri_w=2.0)
    lab["eta_w1"] = lab.vmax_kt + ce.cv_predict(lab, use_eta=True)
    lab["eta_w2"] = lab.vmax_kt + ce.cv_predict(lab, use_eta=True, ri_w=2.0)
    print(f"MLP cells done ({time.time() - t0:.0f}s)")

    keep = ["sid", "init", "step", "ri", "bt_wind_kt", "vmax_kt"] + [c for c, _ in CELLS]
    lab[keep].to_csv(f"{ROOT}/data/tc/ablation.csv", index=False)
    print("wrote data/tc/ablation.csv")

    for title, mask in [("all cases", np.ones(len(lab), bool)), ("RI cases", lab.ri.values == 1)]:
        d = lab[mask]
        print(f"\n--- MAE kt, {title} (n={len(d)}) ---")
        for c, name in CELLS:
            print(f"  {name:>11}: {(d[c] - d.bt_wind_kt).abs().mean():5.1f}")
    print("\n--- tail: q0.99 of forecast vmax (obs "
          f"{lab.bt_wind_kt.quantile(0.99):.0f} kt) ---")
    for c, name in CELLS:
        print(f"  {name:>11}: {lab[c].quantile(0.99):5.1f}")
