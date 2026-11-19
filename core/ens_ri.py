"""P(RI) from the AIFS-ENS members: correct each member, count the fraction that intensify.

Advisor comment (2026-11-11): "try correcting each member and using the fraction that intensify as
P(RI)". For every case with ensemble members (RI-storm inits from 2025-07-02, steps to 120 h) and
each of the 51 members, the point GBM correction (base + SHIPS features, trained out-of-fold on
storm-grouped 5-fold over all 52 storms) is evaluated with the member's own wind and pressure in
place of the deterministic ones. Three P(RI) variants, each the fraction of members whose 24 h
change (corrected member at the step minus corrected member 24 h earlier, or minus the observed
init wind at step 24) reaches 30 kt:
  raw     uncorrected member winds
  shift   corrected deterministic wind plus (member minus deterministic): spread around the point correction
  corrected  each member corrected with its own features
Compared with the direct 24 h change model (riprob.py) and the classifier on the same cases.

Run from core/:  python ens_ri.py      (writes data/out/ens_ri_cv.csv)
"""
import os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import quantile as qm
from quantile import PARAMS, OUT, brier_report, contingency, storm_boot, fmt

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["init", "sid", "step"]

def main():
    df, feats = qm.load(True)
    e = pd.read_csv(os.path.join(HERE, "..", "data", "tc", "ens_members.csv"))
    e = e[e.groupby(KEYS).member.transform("size") == 51]
    X = df[feats].values; y = df.res_v.values; g = df.sid.values
    has = df.set_index(KEYS).index.isin(e.set_index(KEYS).index)   # cases that have members
    corr_det = np.full(len(df), np.nan); rows = []
    for tr, te in GroupKFold(5).split(X, y, g):
        m = HGB(random_state=0, **PARAMS).fit(X[tr], y[tr])
        corr_det[te] = df.vmax_kt.values[te] + m.predict(X[te])
        te = te[has[te]]
        if len(te) == 0:
            continue
        cases = df.iloc[te][list(dict.fromkeys(KEYS + feats + ["aifs0_vmax"]))]
        mem = cases.merge(e, on=KEYS, suffixes=("", "_m"))
        Xm = mem[feats].copy()
        Xm["vmax_kt"] = mem.vmax_kt_m; Xm["pmin_hpa"] = mem.pmin_hpa_m; Xm["aifs_dvmax"] = mem.vmax_kt_m - mem.aifs0_vmax
        mem["corr_member"] = mem.vmax_kt_m + m.predict(Xm.values)
        rows.append(mem[KEYS + ["member", "vmax_kt_m", "corr_member"]])
    mem = pd.concat(rows, ignore_index=True)
    df["corr_det"] = corr_det
    mem = mem.merge(df[KEYS + ["vmax_kt", "corr_det", "obs0_wind", "ri"]], on=KEYS)
    mem["raw"] = mem.vmax_kt_m; mem["shift"] = mem.corr_det + (mem.vmax_kt_m - mem.vmax_kt); mem["corrected"] = mem.corr_member
    # 24 h change per member: value at step minus value at step - 24 (same init, sid, member)
    out = df.loc[has, KEYS + ["ri", "obs0_wind"]].copy()
    out = out[out.step >= 24].reset_index(drop=True)
    for var in ("raw", "shift", "corrected"):
        v = mem.set_index(KEYS + ["member"])[var]
        prev = v.reindex(pd.MultiIndex.from_arrays([mem.init, mem.sid, mem.step - 24, mem["member"]])).values
        base = np.where(mem.step.values == 24, mem.obs0_wind.values, prev)
        mem[f"ri_{var}"] = (mem[var].values - base >= 30.0).astype(float)
        mem.loc[~np.isfinite(base), f"ri_{var}"] = np.nan
        out = out.merge(mem.groupby(KEYS)[f"ri_{var}"].mean().rename(f"p_ri_ens_{var}").reset_index(), on=KEYS, how="left")
    rp = pd.read_csv(f"{OUT}/riprob_cv_ships.csv", keep_default_na=False, na_values=[""])[KEYS + ["p_ri_dv24", "p_ri_cls"]]
    out = out.merge(rp, on=KEYS, how="left")
    out = out[out.p_ri_ens_raw.notna()].reset_index(drop=True)
    ri, g = out.ri.values, out.sid.values
    print(f"ENS subset, step >= 24 with a member 24 h earlier: {len(out)} cases, {out.sid.nunique()} storms, {ri.sum()} RI cases")
    cols = [("p_ri_ens_raw", "fraction of raw members"), ("p_ri_ens_shift", "fraction, spread around corrected det"),
            ("p_ri_ens_corrected", "fraction of corrected members"), ("p_ri_dv24", "direct 24 h change model (riprob)"),
            ("p_ri_cls", "direct classifier")]
    for c, lab in cols:
        p = out[c].values; ok = np.isfinite(p)
        brier_report(lab, p, ri)
        print(f"   AUC {roc_auc_score(ri[ok], p[ok]):.3f};  " + ";  ".join(f"p>={t}: " + contingency(p[ok] >= t, ri[ok] == 1).split('(')[0] for t in (0.2, 0.3, 0.5)))
    print("\nBrier differences vs direct 24 h change model (negative = better), storm bootstrap")
    for c, lab in cols[:3]:
        ok = np.isfinite(out[c].values) & np.isfinite(out.p_ri_dv24.values)
        d = (out[c].values[ok] - ri[ok]) ** 2 - (out.p_ri_dv24.values[ok] - ri[ok]) ** 2
        print(f"  {lab:40s} {fmt(d.mean(), storm_boot(d, g[ok]))}")
    out.to_csv(f"{OUT}/ens_ri_cv.csv", index=False)

if __name__ == "__main__":
    main()
