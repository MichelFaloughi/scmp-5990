"""Mini AIFS-TC: learn an additive correction to AIFS max-wind (and min-pressure) forecasts.

Follows the GBM half of Allen et al. (2026): tabular features derived from the AIFS forecast plus
the operationally-known initial intensity, gradient-boosted trees predicting the residual
(best track - AIFS), storm-grouped cross-validation so a storm never appears in its own training fold.

Input : data/tc/aifs_tracks.csv (from track.py), data/tc/besttrack.csv
Output: data/tc/aifs_tracks_corrected.csv  (adds corr_vmax_kt, corr_pmin_hpa, persist_vmax_kt)
Usage : python tc/correct.py
"""
import os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RI_WEIGHT = 2.0  # rapid intensification cases (>=30 kt in 24 h observed) weighted 2x, as in the paper

def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

def build_features(tr, bt):
    tr = tr.sort_values(["sid", "init", "step"]).copy()
    # AIFS at step 0 and running extremes along the forecast
    g = tr.groupby(["sid", "init"])
    tr["aifs0_vmax"] = g.vmax_kt.transform("first")
    tr["aifs0_pmin"] = g.pmin_hpa.transform("first")
    tr["aifs_dvmax"] = tr.vmax_kt - tr.aifs0_vmax
    tr["aifs_vmax_runmax"] = g.vmax_kt.cummax()
    tr["aifs_pmin_runmin"] = g.pmin_hpa.cummin()
    tr["aifs_vmax_prev6"] = g.vmax_kt.shift(1).fillna(tr.vmax_kt)
    tr["aifs_dvmax_6h"] = tr.vmax_kt - tr.aifs_vmax_prev6
    lat_p, lon_p = g.lat.shift(1), g.lon.shift(1)
    tr["speed_kmh"] = (haversine_km(tr.lat, tr.lon, lat_p, lon_p) / 6.0).fillna(0)
    # observed state at init (best track), and its recent trend
    b = bt[["SID", "ISO_TIME", "USA_WIND", "USA_PRES"]].rename(columns={"SID": "sid", "ISO_TIME": "init"})
    b = b.sort_values(["sid", "init"])
    b["obs0_wind"] = b.USA_WIND; b["obs0_pres"] = b.USA_PRES
    b["obs_dwind_12h"] = b.groupby("sid").USA_WIND.diff(2)
    b["obs_dwind_24h"] = b.groupby("sid").USA_WIND.diff(4)
    b["age_h"] = (b.init - b.groupby("sid").init.transform("min")).dt.total_seconds() / 3600
    tr = tr.merge(b[["sid", "init", "obs0_wind", "obs0_pres", "obs_dwind_12h", "obs_dwind_24h", "age_h"]],
                  on=["sid", "init"], how="left")
    tr["init_bias"] = tr.aifs0_vmax - tr.obs0_wind          # how much AIFS under-resolves this storm now
    tr["persist_vmax_kt"] = tr.vmax_kt - tr.init_bias       # naive baseline: carry the initial bias forward
    doy = tr.init.dt.dayofyear
    tr["doy_sin"], tr["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    tr["is_ep"] = (tr.basin == "EP").astype(int)
    # AIFS-Single 2.0 replaced 1.0 operationally in May 2026 (HF card: 2.0 released 2026-05-12); the archive is
    # operational output, so the model version changes mid-dataset. Flag it so the trees can split on it.
    tr["aifs_v2"] = (tr.init >= pd.Timestamp("2026-05-12")).astype(int)
    # observed RI label at valid time (for weighting only, never a feature)
    bv = bt[["SID", "ISO_TIME", "USA_WIND"]].rename(columns={"SID": "sid", "ISO_TIME": "valid", "USA_WIND": "w"}).sort_values(["sid", "valid"])
    bv["w_m24"] = bv.groupby("sid").w.shift(4)
    bv["ri"] = ((bv.w - bv.w_m24) >= 30).astype(int)
    tr = tr.merge(bv[["sid", "valid", "ri"]], on=["sid", "valid"], how="left")
    tr["ri"] = tr.ri.fillna(0).astype(int)
    return tr

FEATURES = ["step", "lat", "lon", "is_ep", "aifs_v2", "doy_sin", "doy_cos", "vmax_kt", "pmin_hpa", "aifs0_vmax", "aifs0_pmin",
            "aifs_dvmax", "aifs_vmax_runmax", "aifs_pmin_runmin", "aifs_dvmax_6h", "speed_kmh",
            "obs0_wind", "obs0_pres", "obs_dwind_12h", "obs_dwind_24h", "age_h", "init_bias"]

def fit_predict_cv(df, target, groups, n_splits=5, seeds=(0, 1, 2, 3, 4)):
    pred = np.zeros(len(df)); X = df[FEATURES].values; y = df[target].values
    w = np.where(df.ri.values == 1, RI_WEIGHT, 1.0)
    for tr_idx, te_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        p = np.zeros(len(te_idx))
        for s in seeds:
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=20, l2_regularization=1.0, random_state=s)
            m.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
            p += m.predict(X[te_idx]) / len(seeds)
        pred[te_idx] = p
    return pred

if __name__ == "__main__":
    tr = pd.read_csv(f"{ROOT}/data/tc/aifs_tracks.csv", parse_dates=["init", "valid"])
    bt = pd.read_csv(f"{ROOT}/data/tc/besttrack.csv", parse_dates=["ISO_TIME"])
    df = build_features(tr, bt)
    lab = df.dropna(subset=["bt_wind_kt", "obs0_wind"]).copy()
    lab = lab[lab.step > 0]
    lab["res_v"] = lab.bt_wind_kt - lab.vmax_kt
    lab["res_p"] = lab.bt_pres_hpa - lab.pmin_hpa
    n_storms = lab.sid.nunique()
    print(f"labelled cases: {len(lab)}  storms: {n_storms}  inits: {lab.groupby('sid').init.nunique().sum()}  RI cases: {lab.ri.sum()}")
    if n_storms < 5:
        print("fewer than 5 storms: skipping cross-validation (mechanics check only)")
        df.to_csv(f"{ROOT}/data/tc/aifs_tracks_corrected.csv", index=False); raise SystemExit
    lab["corr_vmax_kt"] = lab.vmax_kt + fit_predict_cv(lab, "res_v", lab.sid.values)
    ok_p = lab.res_p.notna()
    lab["corr_pmin_hpa"] = np.nan
    lab.loc[ok_p, "corr_pmin_hpa"] = lab.loc[ok_p, "pmin_hpa"] + fit_predict_cv(lab[ok_p], "res_p", lab[ok_p].sid.values)
    out = df.merge(lab[["sid", "init", "step", "corr_vmax_kt", "corr_pmin_hpa"]], on=["sid", "init", "step"], how="left")
    out.to_csv(f"{ROOT}/data/tc/aifs_tracks_corrected.csv", index=False)
    print("wrote data/tc/aifs_tracks_corrected.csv")
    for name, col in [("raw AIFS", "vmax_kt"), ("persist-bias baseline", "persist_vmax_kt"), ("GBM corrected", "corr_vmax_kt")]:
        mae = (lab[col] - lab.bt_wind_kt).abs().groupby(lab.step).mean()
        print(f"{name:22s} MAE kt by lead:", {int(k): round(v, 1) for k, v in mae.items() if k in (24, 48, 72, 96, 120)})
