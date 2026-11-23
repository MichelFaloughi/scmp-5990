
import os

# Which dataset the core scripts run on. "" is the North Atlantic + East Pacific set of the report;
# SCMP_DATA=global adds the West Pacific, North Indian and Southern Hemisphere storms (advisor item 3,
# 2026-11-23): data files take the suffix _global and outputs go to data/out/global/.
DATASET = os.environ.get("SCMP_DATA", "")
SUFFIX = f"_{DATASET}" if DATASET else ""
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "out", DATASET) if DATASET else os.path.join(HERE, "..", "data", "out")

def data_file(name):
    """core/data/<name>.csv, or core/data/<name>_global.csv under SCMP_DATA=global."""
    return os.path.join(HERE, "data", f"{name}{SUFFIX}.csv")

FEATURES = ["step", "lat", "lon", "is_ep", "aifs_v2", "doy_sin", "doy_cos", "vmax_kt", "pmin_hpa", "aifs0_vmax", "aifs0_pmin",
            "aifs_dvmax", "aifs_vmax_runmax", "aifs_pmin_runmin", "aifs_dvmax_6h", "speed_kmh",
            "obs0_wind", "obs0_pres", "obs_dwind_12h", "obs_dwind_24h", "age_h", "init_bias"]
if DATASET == "global":
    FEATURES = FEATURES + ["is_wp", "is_ni", "is_sh"]

def evaluate(df, col):
    truth = "bt_wind_kt" 

    abs_err = abs(df[col] - df[truth])

    mae_overall = abs_err.mean()
    mae_ri = abs_err[df["ri"] == 1].mean()
    mae_by_step = abs_err.groupby(df["step"]).mean()

    return mae_overall, mae_ri, mae_by_step

# evaluate(df, "corr_vmax_kt")

def tail_quantiles(df, cols, qs=[0.9, 0.95, 0.99]):
    reduced_df = df[cols]
    return reduced_df.quantile(qs)

# tail_quantiles(df, ["bt_wind_kt", "vmax_kt", "corr_vmax_kt"])