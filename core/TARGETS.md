# Numbers to match

From the repo's runs (all cases n=9149, 52 storms; RI cases n=585). Settings that matter:
step > 0 cases only, GroupKFold(5) on sid, MLP = 64-64 ReLU, full-batch Adam lr 3e-3,
600 iters, seeds (0,1,2) averaged; nu_0 tail = 25 quantiles, q 0.95 to 0.999.

| # | What | Target |
|---|------|--------|
| 1 | Raw AIFS MAE, all / RI | 28.4 / 73.5 kt |
| 2 | Repo GBM column (corr_vmax_kt) MAE, all / RI | 14.1 / 41.1 kt |
| 3 | Your GBM, same protocol | ~14.1 / ~41.1 kt |
| 4 | Your plain-MSE MLP (ERM) | ~17.7 / ~36.6 kt |
| 5 | Your eta MLP | ~13.1 / ~29.9 kt; q0.99 of forecasts ~143 kt (obs 140) |
| 6 | Paired bootstrap, eta - GBM, all cases | ~-1.0, CI ~[-2.3, +0.5], not distinguishable |
| 6b | Paired bootstrap, eta - GBM, RI cases | ~-11, CI excludes 0 |

MLP numbers wobble ~0.3 kt with seeds/init details; within that = match.
If your eta lands near 18-19 instead of 13, check the iteration count (3000 overfits).
