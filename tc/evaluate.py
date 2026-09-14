"""Score intensity forecasts against best track, by lead time, on homogeneous samples.

Columns: raw AIFS (our tracker), AIFS + GBM correction, NHC OFCL, and any extra ATCF techs present
(GDMN = Google DeepMind cyclone model ensemble mean, EAIO = ECMWF's own AIFS tracker).
Confidence intervals: 95% percentile bootstrap resampling *storms* (not cases), 1000 draws.

Usage: python tc/evaluate.py [--corrected data/tc/aifs_tracks_corrected.csv] [--out data/tc/RESULTS_tables.md]
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEADS = [12, 24, 36, 48, 72, 96, 120, 144, 168]
EXTRA_TECHS = ["GDMN", "EAIO"]
rng = np.random.default_rng(0)

def load(corrected=None):
    tr = pd.read_csv(corrected or f"{ROOT}/data/tc/aifs_tracks.csv", parse_dates=["init", "valid"], keep_default_na=False, na_values=[""])
    bt = pd.read_csv(f"{ROOT}/data/tc/besttrack.csv", parse_dates=["ISO_TIME"], keep_default_na=False, na_values=[""])
    sid2atcf = bt[bt.USA_ATCF_ID != ""].drop_duplicates("SID").set_index("SID").USA_ATCF_ID
    tr["atcf_id"] = tr.sid.map(sid2atcf)
    of = pd.read_csv(f"{ROOT}/data/tc/ofcl.csv", parse_dates=["init"], keep_default_na=False, na_values=[""])
    m = tr
    for tech, prefix in [("OFCL", "ofcl")] + [(t, t.lower()) for t in EXTRA_TECHS]:
        o = of[of.tech == tech].rename(columns={"lead": "step", "vmax_kt": f"{prefix}_vmax_kt", "pmin_hpa": f"{prefix}_pmin_hpa"})
        m = m.merge(o[["atcf_id", "init", "step", f"{prefix}_vmax_kt", f"{prefix}_pmin_hpa"]], on=["atcf_id", "init", "step"], how="left")
    return m

def mae(d, c, truth): return float((d[c] - d[truth]).abs().mean())

def boot_ci(d, c, truth, n=1000):
    sids = d.sid.unique(); groups = {s: g for s, g in d.groupby("sid")}
    vals = []
    for _ in range(n):
        pick = rng.choice(sids, size=len(sids), replace=True)
        sample = pd.concat([groups[s] for s in pick])
        vals.append(mae(sample, c, truth))
    return np.percentile(vals, [2.5, 97.5])

def table(d, cols, truth, ci=False):
    rows = []
    for lead, g in d.groupby("step"):
        row = {"lead_h": int(lead), "n": len(g)}
        for name, c in cols.items(): row[name] = mae(g, c, truth)
        rows.append(row)
    allr = {"lead_h": "12-168", "n": len(d)}
    for name, c in cols.items():
        v = mae(d, c, truth); allr[name] = v
        if ci:
            lo, hi = boot_ci(d, c, truth); allr[name] = f"{v:.1f} [{lo:.1f}, {hi:.1f}]"
    t = pd.DataFrame(rows).round(1)
    return pd.concat([t, pd.DataFrame([allr])], ignore_index=True)

def md(df): return df.to_markdown(index=False)

if __name__ == "__main__":
    corrected = sys.argv[sys.argv.index("--corrected") + 1] if "--corrected" in sys.argv else None
    out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    m = load(corrected)
    m = m[m.step.isin(LEADS)]
    base_cols = {"AIFS raw": "vmax_kt", "NHC OFCL": "ofcl_vmax_kt"}
    if "corr_vmax_kt" in m: base_cols = {"AIFS raw": "vmax_kt", "AIFS+GBM": "corr_vmax_kt", "NHC OFCL": "ofcl_vmax_kt"}
    out = []
    def emit(title, df):
        out.append(f"### {title}\n\n{md(df)}\n"); print(f"\n{title}\n{df.to_string(index=False)}")

    # --- main homogeneous sample: raw, corrected, OFCL
    need = [c for c in base_cols.values()] + ["bt_wind_kt"]
    h = m.dropna(subset=need)
    print(f"homogeneous sample: {len(h)} cases, {h.sid.nunique()} storms, {h.groupby('sid').init.nunique().sum()} inits")
    emit(f"Max-wind MAE (kt) vs best track — all cases (n={len(h)}, {h.sid.nunique()} storms). 12-168 row: 95% CI, storm bootstrap", table(h, base_cols, "bt_wind_kt", ci=True))
    ri = h[h.ri == 1]
    emit(f"Rapid-intensification cases only (n={len(ri)}, {ri.sid.nunique()} storms)", table(ri, base_cols, "bt_wind_kt", ci=True))
    for name, sub in [("North Atlantic", h[h.basin == "NA"]), ("East Pacific", h[h.basin == "EP"]),
                      ("2025 season (AIFS 1.0)", h[h.aifs_v2 == 0]), ("2026 season (AIFS 2.0)", h[h.aifs_v2 == 1])]:
        if len(sub): emit(f"{name} (n={len(sub)}, {sub.sid.nunique()} storms)", table(sub, base_cols, "bt_wind_kt", ci=True))
    # --- pressure
    pc = {"AIFS raw": "pmin_hpa", "AIFS+GBM": "corr_pmin_hpa", "NHC OFCL": "ofcl_pmin_hpa"} if "corr_pmin_hpa" in m else {"AIFS raw": "pmin_hpa", "NHC OFCL": "ofcl_pmin_hpa"}
    p = m.dropna(subset=list(pc.values()) + ["bt_pres_hpa"])
    if len(p): emit(f"Min central pressure MAE (hPa) (n={len(p)}, {p.sid.nunique()} storms)", table(p, pc, "bt_pres_hpa", ci=True))
    # --- vs DeepMind (GDMN) where available: separate homogeneous sample
    if "gdmn_vmax_kt" in m and m.gdmn_vmax_kt.notna().any():
        cols = dict(base_cols, **{"DeepMind GDMN": "gdmn_vmax_kt"})
        g = m.dropna(subset=list(cols.values()) + ["bt_wind_kt"])
        emit(f"vs Google DeepMind cyclone model (GDMN), homogeneous (n={len(g)}, {g.sid.nunique()} storms, {g.init.min():%Y-%m-%d}..{g.init.max():%Y-%m-%d})", table(g, cols, "bt_wind_kt", ci=True))
        gri = g[g.ri == 1]
        if len(gri) > 20: emit(f"vs GDMN, RI cases only (n={len(gri)})", table(gri, cols, "bt_wind_kt", ci=True))
    # --- tracker check vs ECMWF's own AIFS tracks (EAIO)
    if "eaio_vmax_kt" in m and m.eaio_vmax_kt.notna().any():
        e = m.dropna(subset=["eaio_vmax_kt", "vmax_kt"])
        d = e.vmax_kt - e.eaio_vmax_kt
        line = f"Our tracker vs ECMWF's AIFS tracker (EAIO), same init/lead (n={len(e)}, {e.sid.nunique()} storms): mean diff {d.mean():+.1f} kt, MAE {d.abs().mean():.1f} kt, 90th pct |diff| {d.abs().quantile(.9):.1f} kt"
        out.append(f"### Tracker validation\n\n{line}\n"); print("\n" + line)
    if out_path:
        open(out_path, "w").write("\n".join(out)); print(f"\nwrote {out_path}")
