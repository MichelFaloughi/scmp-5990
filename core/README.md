# scmp-5990 rebuild

Rebuilding the modeling core of this repo by hand, to own it.

## Rules

1. Write each piece from scratch before looking at the repo's version. Compare numbers
   (TARGETS.md), then compare code.
2. Exercises 1 to 4 (`gbm.py`, `eta.ipynb`, `paired_ci.ipynb`) were written by hand with an
   AI assistant acting only as a TA. From 2026-10-27 the later files (`quantile.py`,
   `fair_baseline.py`) were written with the assistant's help, under time pressure.
3. Matching a target within ~0.2 kt = that step is owned. Move on.

## Data

- `data/reference.csv`: the labeled dataset, one row per (storm, init, lead) forecast case.
  All engineered features are already columns (see FEATURES in ../pipeline/correct.py
  for the list). Truth is `bt_wind_kt`, raw AIFS is `vmax_kt`, target residual is
  `res_v = bt_wind_kt - vmax_kt`. Also contains the repo's predictions
  (`corr_vmax_kt`, `erm_vmax_kt`, `eta_vmax_kt`, `eta1980_vmax_kt`) for comparison.
- `data/aifs_tracks.csv`, `data/besttrack.csv`: raw inputs, only needed for the optional
  feature-engineering exercise.
- `data/ships_features.csv`: SHIPS-style environmental predictors computed from AIFS
  pressure-level fields by `../pipeline/ships_features.py`, one row per case, joined on
  (`init`, `sid`, `step`).
- `data/ens_features.csv`: AIFS-ENS member spread features (mean, std, percentiles, control,
  differences from the deterministic wind) from `../pipeline/ens_features.py`, for the 3323 cases
  of RI-storm inits from 2025-07-02 at leads 6 to 120 h. Used only by `ens_ablation.py`.

## Exercises, in order
1. **eval.py** - MAE overall / by lead / RI subset (`ri == 1`), tail quantiles of forecast
   vs observed winds. Score `vmax_kt` and `corr_vmax_kt` from reference.csv. Match targets 1-2.
2. **gbm.py** - HistGradientBoostingRegressor on the residual, storm-grouped 5-fold CV
   (GroupKFold on `sid`), 2x sample weight on RI cases, 5 seeds averaged. Match target 3.
3. **eta.py** - the core. MLP on the residual, plain MSE first (match target 4), then add
   the tail penalty: corrected wind quantiles (q in [0.95, 0.999], 25 points) pulled toward
   the same quantiles of the training fold's observed winds, lambda by gradient-norm
   balancing. Match target 5.
4. **paired_ci.py** - storm bootstrap; then the paired difference version. Match target 6.
5. Optional: features from the raw inputs; a debugger walk through ../pipeline/track.py
   with one storm.

## Papers

- Allen et al. 2026, AIFS-TC: arXiv 2608.09959 (what is being reproduced)
- Chang & Sapsis 2026, eta-learning: arXiv 2510.19161 (the method, esp. eq. 15)
