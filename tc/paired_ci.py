"""Paired storm-bootstrap CIs for the intensity-correction comparison.

Point 2 of the 2026-09-16 checklist: bootstrap the paired MAE difference between
models rather than two separate intervals. All models are scored on the same
cases, so their errors are correlated; resampling storms once per draw and
differencing inside the draw cancels the shared storm-level noise. A 95%
interval on the difference that excludes 0 means the gap is distinguishable.

Input : data/tc/aifs_tracks_eta.csv (corr_vmax_kt, erm_vmax_kt, eta_vmax_kt, ...)
Usage : python tc/paired_ci.py [--n 2000]
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 2000
rng = np.random.default_rng(0)

MODELS = {"AIFS raw": "vmax_kt", "GBM": "corr_vmax_kt", "MLP ERM": "erm_vmax_kt",
          "MLP eta": "eta_vmax_kt", "eta 1980": "eta1980_vmax_kt"}
PAIRS = [("MLP eta", "GBM"), ("MLP eta", "MLP ERM"), ("MLP eta", "eta 1980"),
         ("eta 1980", "GBM"), ("GBM", "AIFS raw")]


def storm_sums(d, mask, sids, models):
    """Per-storm sum of |error| and case count on the masked subset, aligned to sids."""
    m = d[mask]
    cnt = m.groupby("sid").size().reindex(sids, fill_value=0).values.astype(float)
    sums = {n: (m[c] - m.bt_wind_kt).abs().groupby(m.sid).sum()
                .reindex(sids, fill_value=0.0).values for n, c in models.items()}
    return sums, cnt


def boot_draws(sums, cnt):
    """MAE per model on N storm resamples; the same resample serves every model (paired)."""
    S = len(cnt)
    out = {k: np.empty(N) for k in sums}
    for i in range(N):
        c = 0.0
        while c == 0.0:   # a draw of only zero-count storms (small subsets) has no MAE; redraw
            pick = rng.integers(0, S, S)
            c = cnt[pick].sum()
        for k in sums:
            out[k][i] = sums[k][pick].sum() / c
    return out


if __name__ == "__main__":
    d = pd.read_csv(f"{ROOT}/data/tc/aifs_tracks_eta.csv", parse_dates=["init", "valid"],
                    keep_default_na=False, na_values=[""])
    models = {n: c for n, c in MODELS.items() if c in d.columns}
    sids = np.sort(d.sid.unique())
    for title, mask in [("all cases", np.ones(len(d), bool)), ("RI cases", d.ri.values == 1)]:
        sums, cnt = storm_sums(d, mask, sids, models)
        draws = boot_draws(sums, cnt)
        n_cases = int(cnt.sum())
        print(f"\n=== {title} (n={n_cases}, {int((cnt > 0).sum())} storms, {N} draws) ===")
        print("MAE kt with 95% CI:")
        for n_ in models:
            v = sums[n_].sum() / cnt.sum()
            lo, hi = np.percentile(draws[n_], [2.5, 97.5])
            print(f"  {n_:>10}: {v:5.1f} [{lo:5.1f}, {hi:5.1f}]")
        print("paired differences, A - B (negative = A better):")
        for a, b in PAIRS:
            if a not in models or b not in models:
                continue
            v = (sums[a].sum() - sums[b].sum()) / cnt.sum()
            diff = draws[a] - draws[b]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            sig = "distinguishable" if lo > 0 or hi < 0 else "NOT distinguishable"
            print(f"  {a:>10} - {b:<10}: {v:+5.2f} [{lo:+5.2f}, {hi:+5.2f}]  {sig}")
