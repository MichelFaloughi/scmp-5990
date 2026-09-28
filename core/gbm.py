from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd
from common import evaluate

df = pd.read_csv("data/reference.csv", keep_default_na=False, na_values=[""])
print(df.to_string(max_rows=8))



# Prepare the data
FEATURES = ["step", "lat", "lon", "is_ep", "aifs_v2", "doy_sin", "doy_cos", "vmax_kt", "pmin_hpa", "aifs0_vmax", "aifs0_pmin",
            "aifs_dvmax", "aifs_vmax_runmax", "aifs_pmin_runmin", "aifs_dvmax_6h", "speed_kmh",
            "obs0_wind", "obs0_pres", "obs_dwind_12h", "obs_dwind_24h", "age_h", "init_bias"]
X = df[FEATURES]
y = df["res_v"]
groups = df["sid"]
w = np.where(df["ri"] == 1, 2, 1) # weight doubles when we have RI
pred = np.zeros(len(df))

# Set up the splitter
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)

# The loops
for train_idx, test_idx in gkf.split(X, y, groups):
    # rows for this fold: X.iloc[train_idx], y.iloc[train_idx], etc.
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    w_train = w[train_idx]
    X_test  = X.iloc[test_idx]

    for seed in range(5):
        # create the model with the hyperparameters and random_state=seed
        gbm = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0, 
            random_state=seed
        )
        
        # fit on the training rows, with sample_weight
        gbm.fit(X_train, y_train, sample_weight=w_train)

        # predict the test rows
        fold_pred = gbm.predict(X_test)

        # add prediction / 5 into pred[test_idx]
        pred[test_idx] += fold_pred/5 

# After the loops
df["my_gbm_kt"] = df["vmax_kt"] + pred
evaluate(df, "my_gbm_kt")
