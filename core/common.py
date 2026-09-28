
FEATURES = ["step", "lat", "lon", "is_ep", "aifs_v2", "doy_sin", "doy_cos", "vmax_kt", "pmin_hpa", "aifs0_vmax", "aifs0_pmin",
            "aifs_dvmax", "aifs_vmax_runmax", "aifs_pmin_runmin", "aifs_dvmax_6h", "speed_kmh",
            "obs0_wind", "obs0_pres", "obs_dwind_12h", "obs_dwind_24h", "age_h", "init_bias"]

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