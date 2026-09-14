# Overall performance

Learned baselines come from `results/benchmark/overall_performance_comparison.csv`; non-learned methods and NQO come from `results/benchmark/nqo/nqo_runs.csv` and are summarized in `results/benchmark/nqo/analyze_nqo.log`. Each cell reports `WS / GS / Imp` under the `first-vs-first` protocol. Non-learned methods are independent of the train-test split, so the same result is repeated for all three JOB/STACK protocols. For TPC-H, QuerySplit falls back to PostgreSQL on all queries.

| Category | Method | JOB Base-query | JOB Leave-one-out | JOB Random | STACK Base-query | STACK Leave-one-out | STACK Random | TPC-H Random |
|---|---|---|---|---|---|---|---|---|
| Learned | FASTgres | 0.952051 / 0.823183 / 32/113 (28.32%) | 0.982948 / 0.913089 / 50/115 (43.48%) | 0.974988 / 0.926883 / 52/113 (46.02%) | 0.476801 / 0.510815 / 1/112 (0.89%) | 1.296724 / 0.866970 / 36/112 (32.14%) | 1.537904 / 1.070639 / 42/112 (37.50%) | 0.975212 / 0.857084 / 2/22 (9.09%) |
| Learned | FASTgres w/o inference time | 0.973665 / 1.080407 / 67/113 (59.29%) | 1.005421 / 1.231285 / 83/115 (72.17%) | 0.996441 / 1.232707 / 86/113 (76.11%) | 0.482036 / 0.612205 / 10/112 (8.93%) | 1.335561 / 1.128710 / 57/112 (50.89%) | 1.590016 / 1.533857 / 72/112 (64.29%) | 1.001297 / 0.926410 / 2/22 (9.09%) |
| Learned | TONIC | 0.590060 / 0.331141 / 14/113 (12.39%) | 0.572206 / 0.359325 / 13/115 (11.30%) | 0.572583 / 0.350452 / 12/113 (10.62%) | 0.231091 / 0.210794 / 4/112 (3.57%) | 0.244969 / 0.232182 / 4/112 (3.57%) | 0.238289 / 0.229888 / 3/112 (2.68%) | 0.701946 / 0.799401 / 7/22 (31.82%) |
| Learned | TONIC w/o inference time | 0.607467 / 0.347411 / 14/113 (12.39%) | 0.587757 / 0.374308 / 13/115 (11.30%) | 0.588202 / 0.367243 / 12/113 (10.62%) | 0.235201 / 0.260750 / 4/112 (3.57%) | 0.249544 / 0.291844 / 7/112 (6.25%) | 0.242462 / 0.286111 / 6/112 (5.36%) | 0.704025 / 0.807369 / 7/22 (31.82%) |
| Learned | GenJoin | 1.048208 / 0.569395 / 13/113 (11.50%) | 1.035550 / 0.552414 / 15/115 (13.04%) | 1.069384 / 0.560718 / 17/113 (15.04%) | 0.847999 / 0.553200 / 7/112 (6.25%) | 0.799556 / 0.540167 / 5/109 (4.59%) | 0.809823 / 0.551306 / 9/110 (8.18%) | 0.945923 / 0.890550 / 3/22 (13.64%) |
| Learned | GenJoin w/o inference time | 1.290926 / 0.962381 / 49/113 (43.36%) | 1.291358 / 1.024118 / 62/115 (53.91%) | 1.350305 / 1.008317 / 48/113 (42.48%) | 1.103240 / 1.036461 / 65/112 (58.04%) | 1.055989 / 1.012653 / 59/109 (54.13%) | 1.053384 / 1.017058 / 61/110 (55.45%) | 0.976367 / 0.970607 / 6/22 (27.27%) |
| Learned | HybridQO | 1.059883 / 0.585089 / 3/113 (2.65%) | 0.839057 / 0.572643 / 4/115 (3.48%) | 0.821938 / 0.563186 / 4/113 (3.54%) | 0.778660 / 0.564075 / 0/112 (0.00%) | 0.761095 / 0.554020 / 0/109 (0.00%) | 0.785837 / 0.561716 / 0/110 (0.00%) | 0.844773 / 0.882426 / 6/22 (27.27%) |
| Learned | HybridQO w/o inference time | 1.323649 / 1.013072 / 52/113 (46.02%) | 0.993012 / 0.999595 / 65/115 (56.52%) | 0.984320 / 0.982454 / 66/113 (58.41%) | 0.995068 / 0.997156 / 61/112 (54.46%) | 0.981678 / 0.992348 / 58/109 (53.21%) | 1.002095 / 1.002643 / 63/110 (57.27%) | 0.862277 / 0.934489 / 9/22 (40.91%) |
| Learned | AutoSteer | 0.087825 / 0.040400 / 2/113 (1.77%) | 0.088056 / 0.040935 / 3/115 (2.61%) | 0.092247 / 0.042685 / 2/113 (1.77%) | 0.028435 / 0.024535 / 0/112 (0.00%) | 0.024517 / 0.023304 / 0/109 (0.00%) | 0.028237 / 0.024733 / 0/110 (0.00%) | 0.744204 / 0.490426 / 4/22 (18.18%) |
| Learned | AutoSteer w/o inference time | 0.881813 / 1.058359 / 56/113 (49.56%) | 2.087826 / 1.214851 / 56/115 (48.70%) | 1.517784 / 1.145906 / 53/113 (46.90%) | 1.009745 / 1.161414 / 64/112 (57.14%) | 1.117327 / 1.332132 / 80/109 (73.39%) | 1.106832 / 1.332100 / 79/110 (71.82%) | 1.213241 / 1.183095 / 12/22 (54.55%) |
| Non-learned | QuerySplit | 1.524650 / 1.105524 / 54/113 (47.79%) | 1.524650 / 1.105524 / 54/113 (47.79%) | 1.524650 / 1.105524 / 54/113 (47.79%) | 1.436156 / 1.039256 / 56/112 (50.00%) | 1.436156 / 1.039256 / 56/112 (50.00%) | 1.436156 / 1.039256 / 56/112 (50.00%) | 1.000000 / 1.000000 / 0/22 (0.00%) |
| Non-learned | LIP (Sel.) | 1.312161 / 1.006239 / 53/113 (46.90%) | 1.312161 / 1.006239 / 53/113 (46.90%) | 1.312161 / 1.006239 / 53/113 (46.90%) | 1.287719 / 1.096976 / 74/112 (66.07%) | 1.287719 / 1.096976 / 74/112 (66.07%) | 1.287719 / 1.096976 / 74/112 (66.07%) | 0.995421 / 0.977704 / 7/22 (31.82%) |
| Non-learned | LIP (Full) | 1.301219 / 0.984785 / 51/113 (45.13%) | 1.301219 / 0.984785 / 51/113 (45.13%) | 1.301219 / 0.984785 / 51/113 (45.13%) | 0.924524 / 0.785777 / 16/112 (14.29%) | 0.924524 / 0.785777 / 16/112 (14.29%) | 0.924524 / 0.785777 / 16/112 (14.29%) | 0.991775 / 0.978029 / 8/22 (36.36%) |
| Non-learned | AJA (Cons.) | 1.297209 / 1.167454 / 65/113 (57.52%) | 1.297209 / 1.167454 / 65/113 (57.52%) | 1.297209 / 1.167454 / 65/113 (57.52%) | 1.343716 / 1.196296 / 82/112 (73.21%) | 1.343716 / 1.196296 / 82/112 (73.21%) | 1.343716 / 1.196296 / 82/112 (73.21%) | 1.011953 / 0.992037 / 5/22 (22.73%) |
| Non-learned | AJA (Aggr.) | 1.294500 / 1.174292 / 64/113 (56.64%) | 1.294500 / 1.174292 / 64/113 (56.64%) | 1.294500 / 1.174292 / 64/113 (56.64%) | 0.592711 / 0.654932 / 30/112 (26.79%) | 0.592711 / 0.654932 / 30/112 (26.79%) | 0.592711 / 0.654932 / 30/112 (26.79%) | 0.991224 / 0.978384 / 5/22 (22.73%) |
| Non-learned | TOP-5 (DP) | 0.424694 / 0.356730 / 24/113 (21.24%) | 0.424694 / 0.356730 / 24/113 (21.24%) | 0.424694 / 0.356730 / 24/113 (21.24%) | 0.268614 / 0.117544 / 15/112 (13.39%) | 0.268614 / 0.117544 / 15/112 (13.39%) | 0.268614 / 0.117544 / 15/112 (13.39%) | 0.937317 / 0.893397 / 9/22 (40.91%) |
| Non-learned | TOP-10 (DP) | 0.521842 / 0.450093 / 20/113 (17.70%) | 0.521842 / 0.450093 / 20/113 (17.70%) | 0.521842 / 0.450093 / 20/113 (17.70%) | 0.334469 / 0.349914 / 1/112 (0.89%) | 0.334469 / 0.349914 / 1/112 (0.89%) | 0.334469 / 0.349914 / 1/112 (0.89%) | 0.943686 / 0.897916 / 11/22 (50.00%) |
| NQO | NQO | 1.732855 / 0.977054 / 54/113 (47.79%) | 1.723802 / 0.986182 / 58/115 (50.43%) | 1.789778 / 1.024788 / 60/113 (53.10%) | 1.511860 / 0.910650 / 43/112 (38.39%) | 1.681650 / 0.936187 / 44/112 (39.29%) | 1.635543 / 0.904246 / 42/112 (37.50%) | 1.048593 / 1.004016 / 8/22 (36.36%) |
| NQO | NQO w/o inference time | 1.854701 / 1.250958 / 61/113 (53.98%) | 1.860182 / 1.317016 / 63/115 (54.78%) | 1.924302 / 1.350700 / 63/113 (55.75%) | 1.585140 / 1.175222 / 53/112 (47.32%) | 1.782788 / 1.260174 / 60/112 (53.57%) | 1.688047 / 1.091101 / 49/112 (43.75%) | 1.051918 / 1.011531 / 8/22 (36.36%) |

# 1. Learning efficiency of NQO on training queries
Original figure caption:
"Normalized runtime is defined as Σ tNQO / Σ tPG. #SubQ denotes the final number of unique executed subquery-action pairs observed during training."

This file records the data used to redraw the original paper figures with the new experiments. `Best-so-far 1/WS` first selects the best result for each fold up to the current iteration, including the `-2/-1` initialization points, and then aggregates execution time across folds. `Elapsed Time` is the equivalent cumulative cost of running all three folds of one protocol sequentially. It includes model pretraining before iteration 0, PPO updates, training-SQL collection, evaluation every four iterations, and versioned light-buffer lookup costs. Policy inference is rerun at every evaluation point, but SQL is not re-executed when the predicted complete trajectory hits the cache; a miss executes once and is written back. SQL episodes are aligned with the versioned light buffer by semantic trajectory ID and use the first execution time stored there. Each protocol starts with an empty logical cache, its three folds share the cache, and SQL time is charged only on the first appearance of a trajectory. Model time is estimated from manifest, label-snapshot, sampling, and checkpoint timestamps and excludes SQL-lock waiting. Measured mean costs for querying and decompressing candidate trajectories are TPC-H `1.461 ms/query`, JOB `79.562 ms/query`, and STACK `88.835 ms/query`; these costs are charged for every training sample and checkpoint evaluation. The `-2/-1` points are protocol-defined initialization measurements. For JOB/STACK, `-1` includes the initialization cost of the three sequential folds; for TPC-H, `-1` is a zero-cost pretraining point. A fresh evaluation of the final best checkpoint is excluded from the training curve.

## TPC-H

- `#SubQ = 102`, globally deduplicated across the training trajectories of all three folds.
- Q7, Q8, Q9, Q13, Q15, and Q22 do not support actions and use the PostgreSQL fallback result.
- Iteration 0 includes one shared independent-action collection and model pretraining for all three folds: PostgreSQL `0.530 min` + TOP-5 `0.566 min` + selective LIP `0.533 min` + conservative AJA `0.524 min` + Search/Low pretraining `6.265 min` = **`8.417 min`**. High is fixed to stop and is not pretrained. Per-fold model-pretraining time is estimated as `2.088 min` by scaling measured JOB/STACK time under the same configuration by minibatch count.

| Training iteration | Elapsed Time (min) | Best-so-far 1/WS |
|---:|---:|---:|
| -1 | 0.000 | 1.230000 |
| 0  | 8.947 | 0.991683 |
| 4  | 10.896 | 0.991683 |
| 8  | 11.735 | 0.991683 |
| 12 | 12.336 | 0.959797 |
| 16 | 12.977 | 0.958770 |
| 20 | 13.517 | 0.958770 |
| 24 | 14.054 | 0.958770 |
| 28 | 14.608 | 0.958770 |
| 32 | 15.153 | 0.955151 |
| 36 | 15.722 | 0.953784 |
| 40 | 16.267 | 0.953784 |
| 44 | 16.822 | 0.953784 |
| 48 | 17.375 | 0.953784 |
| 52 | 17.928 | 0.953784 |
| 56 | 18.476 | 0.953659 |

The final `1/WS = 0.953659`, corresponding to `WS = 1.048593`.

## JOB

WS uses the first-vs-first protocol: NQO uses the first stored trajectory result, while PostgreSQL uses baseline repetition 0. `#SubQ` globally deduplicates training trajectories from iterations 1--32 across all three folds. `Elapsed Time` is reconstructed with the cache-aware protocol above from each stage's final `episodes.csv`. It excludes duplicate SQLite episodes left by interruption and resume, excludes time spent by nine concurrent tasks waiting for the shared SQL lock, and does not sum wall time from overlapping processes.

Independent-action prior collection is reported separately and is not included in iteration 0 of each protocol: PostgreSQL `3.855 min` + Query Split `2.502 min` + TOPK `11.405 min` + AJA `8.043 min` + LIP `3.836 min` = **`29.640 min`**.

| Protocol | #SubQ | Best fold checkpoints (a/b/c) | Final WS | Final 1/WS |
|---|---:|---|---:|---:|
| Base-query | 2737 | 4 / 16 / 4 | 1.732855 | 0.577082 |
| Leave-one-out | 2533 | 4 / 24 / 0 | 1.723802 | 0.580113 |
| Random | 2716 | 0 / 28 / 4 | 1.789778 | 0.558729 |

| Training iteration | Base elapsed (min) | Base best-so-far 1/WS | LOO elapsed (min) | LOO best-so-far 1/WS | Random elapsed (min) | Random best-so-far 1/WS |
|---:|---:|---:|---:|---:|---:|---:|
| -2  | 0  | 1.520000 | 0 | 1.430000 | 0  | 1.480000 |
| -1  | 12.600  | 1.120000 | 16.400 | 1.030000 | 15.500  | 1.180000 |
| 0  | 29.906  | 0.860601 | 32.679  | 0.680398 | 31.677  | 0.779273 |
| 4  | 46.718  | 0.678102 | 46.995  | 0.593207 | 45.358  | 0.753472 |
| 8  | 59.449  | 0.675083 | 59.864  | 0.593207 | 57.714  | 0.615507 |
| 12 | 71.347  | 0.674545 | 73.233  | 0.585751 | 70.451  | 0.615507 |
| 16 | 83.742  | 0.577082 | 86.481  | 0.585751 | 84.723  | 0.615507 |
| 20 | 96.572  | 0.577082 | 100.243 | 0.585751 | 96.103  | 0.615507 |
| 24 | 108.045 | 0.577082 | 112.953 | 0.580113 | 109.504 | 0.567258 |
| 28 | 120.127 | 0.577082 | 124.400 | 0.580113 | 120.647 | 0.558729 |
| 32 | 132.756 | 0.577082 | 137.441 | 0.580113 | 133.336 | 0.558729 |

## STACK

All results use the first-vs-first protocol: `WS_first = ΣPG_first / ΣNQO_first`. `Best-so-far` lets each fold select its best checkpoint through the current iteration, including the `-2/-1` initialization points, before aggregating total time across folds. Base-query and Leave-one-out begin at iteration 0. For Random at iteration 0, folds a/b use R0/T0, while fold c carries forward its iteration -1 best result because no iteration 0 checkpoint exists.

`#SubQ` follows the JOB protocol and globally deduplicates training-only `(subquery SQL hash, round action tuple)` pairs across all three folds. Base-query and Leave-one-out cover iterations 1--16; the measured Random checkpoint curve extends through iteration 24.

| Protocol | #SubQ | Best fold checkpoints (a/b/c) | Final WS_first | Final 1/WS_first |
|---|---:|---|---:|---:|
| Base-query | 1480 | 16 / 16 / 0 | 1.511860 | 0.661437 |
| Leave-one-out | 1480 | 4 / 12 / 12 | 1.681650 | 0.594654 |
| Random | 2597 | R24 / T0 / R4 | 1.635543 | 0.611418 |

Independent-action prior collection is reported separately and is not included in iteration 0 of each protocol: PostgreSQL `3.417 min` + Query Split `3.254 min` + TOPK `18.191 min` + AJA `3.725 min` + LIP `3.734 min` = **`32.321 min`**.

`Elapsed Time` follows the common protocol defined above. Random performance is aligned with the true model dependencies: fold a uses Original O0 followed by Refinement R0--R24, fold b uses independently retrained T0, and fold c uses Original O0--O16 followed by Refinement R0--R4. Time assumes all three folds are trained sequentially to every x-axis iteration. For folds b/c, which stop early, missing four-iteration blocks are filled using the measured single-fold Random refinement cost and the relative variation across fold-a checkpoint intervals, so last-observation-carried-forward performance is not assigned zero training cost. Base-query/Leave-one-out iterations 0--16 come from measured records. Missing Random fold intervals and the extension of all three protocols through iteration 32 are approximate estimates. Additional hyperparameter-search wall time is excluded from the main training curves.

| Training iteration | Base elapsed (min) | Base best-so-far 1/WS | LOO elapsed (min) | LOO best-so-far 1/WS | Random elapsed (min) | Random best-so-far 1/WS |
|---:|---:|---:|---:|---:|---:|---:|
| -2  | 0  | 1.320000 | 0 | 1.280000 | 0  | 1.410000 |
| -1  | 12.600  | 0.950000 | 16.400 | 0.930000 | 15.500  | 1.180000 |
| 0 | 35.003 | 0.615343 | 31.456 | 0.749187 | 34.105 | 0.851717 |
| 4 | 64.478 | 0.603944 | 57.041 | 0.748653 | 64.653 | 0.851717 |
| 8 | 86.290 | 0.566009 | 82.690 | 0.648431 | 85.697 | 0.805970 |
| 12 | 106.984 | 0.566009 | 104.418 | 0.594654 | 108.118 | 0.805970 |
| 16 | 128.233 | 0.549920 | 124.464 | 0.594654 | 129.554 | 0.805970 |
| 20 | 149.482 | 0.549920 | 144.510 | 0.594654 | 150.980 | 0.616355 |
| 24 | 170.175 | 0.549920 | 166.237 | 0.594654 | 171.940 | 0.611418 |
| 28 | 191.987 | 0.549920 | 191.886 | 0.594654 | 192.340 | 0.611418 |
| 32 | 221.463 | 0.549920 | 217.472 | 0.594654 | 211.300 | 0.611418 |


# 2. Action Importance on the Random split.
Original figure caption: Action importance of NQO on the Random split.

After removing one optimization action, the model is trained from scratch and evaluated on all three folds. Each JOB/STACK fold independently selects its best-so-far checkpoint, and `WS = ΣPG_first / ΣNQO_first` is computed from total first-vs-first time across folds. TPC-H uses the official third run of the complete three-run action grid and aggregates third-vs-third. Neither protocol is an arithmetic mean of per-fold WS values.

| Dataset | NQO WS | w/o Query Split WS | w/o TOPK WS | w/o filter WS | w/o Ajoin WS |
|---|---:|---:|---:|---:|---:|
| TPC-H | 1.048593 | — | 1.013135 | 1.009105 | 1.015898 |
| JOB | 1.789778 | 1.248472 | 1.360611 | 1.451699 | 1.496277 |
| STACK | 1.635543 | 0.977543 | 1.546129 | 1.535097 | 1.540386 |


# 3. Action Frequence
Original figure caption: “Hierarchical action distribution of NQO”

Frequencies are computed from deterministic predictions of the selected NQO checkpoints on test queries and normalized independently within each head: `frequency = count / total decisions for that head`. The six bypassed TPC-H queries produce no policy decisions, so the denominator is the remaining 16 queries.

## TPC-H (random)

| Head | Action | Count | Frequency |
|---|---|---:|---:|
| High | Stop | 16 | 1.0000 |
| High | Split | 0 | 0.0000 |
| Search | Default | 7 | 0.4375 |
| Search | TOPK (top5) | 9 | 0.5625 |
| Low | None | 12 | 0.7500 |
| Low | Filter (LIP) | 0 | 0.0000 |
| Low | Ajoin (AJA) | 1 | 0.0625 |
| Low | Filter + Ajoin | 3 | 0.1875 |

Normalization check: High `1.0000`, Search `1.0000`, and Low `1.0000`. TPC-H disables query splitting, so High/Stop is forced by a mask. The Select/alpha head is never invoked and therefore has no frequency distribution.

## JOB

Each protocol uses its best fold checkpoints under the WS-first protocol. Base-query has denominators of 385 for High/Search/Low and 272 for Select; Leave-one-out has 352 and 237; Random has 306 and 193. Every head is independently normalized to one within each protocol. `Mean frequency` is the equally weighted macro-average of normalized frequencies across the three protocols.

| Head | Action | Base-query | Leave-one-out | Random | Mean frequency |
|---|---|---:|---:|---:|---:|
| High | Stop | 0.2935 | 0.3267 | 0.3693 | 0.3298 |
| High | Split | 0.7065 | 0.6733 | 0.6307 | 0.6702 |
| Select | α=0.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Select | α=0.5 | 1.0000 | 1.0000 | 0.7565 | 0.9188 |
| Select | α=1.0 | 0.0000 | 0.0000 | 0.2435 | 0.0812 |
| Search | Default | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Search | TOPK (top5) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Low | None | 0.5065 | 0.7102 | 0.8529 | 0.6899 |
| Low | Filter (LIP) | 0.2571 | 0.1080 | 0.0000 | 0.1217 |
| Low | Ajoin (AJA) | 0.2208 | 0.0455 | 0.0000 | 0.0887 |
| Low | Filter + Ajoin | 0.0156 | 0.1364 | 0.1471 | 0.0997 |

Normalization check: High, Select, Search, and Low each sum to `1.0000` for every protocol, apart from rounding error.

## STACK

Each protocol uses the WS-first best fold checkpoints from Learning Efficiency, with counts taken from the complete predicted trajectories stored in `nqo_runs.csv`. The High/Select/Search/Low denominators are 429/320/426/425 for Base-query, 435/323/435/435 for Leave-one-out, and 453/350/450/450 for Random. `Mean frequency` is the equally weighted macro-average of normalized frequencies across the three protocols.

| Head | Action | Base-query | Leave-one-out | Random | Mean frequency |
|---|---|---:|---:|---:|---:|
| High | Stop | 0.2494 | 0.2575 | 0.2208 | 0.2425 |
| High | Split | 0.7506 | 0.7425 | 0.7792 | 0.7575 |
| Select | α=0.0 | 0.0000 | 0.0000 | 0.1057 | 0.0352 |
| Select | α=0.5 | 1.0000 | 1.0000 | 0.8943 | 0.9648 |
| Select | α=1.0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Search | Default | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Search | TOPK (top5) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Low | None | 0.2400 | 0.2414 | 0.9600 | 0.4805 |
| Low | Filter (LIP) | 0.7153 | 0.7241 | 0.0267 | 0.4887 |
| Low | Ajoin (AJA) | 0.0353 | 0.0115 | 0.0067 | 0.0178 |
| Low | Filter + Ajoin | 0.0094 | 0.0230 | 0.0067 | 0.0130 |

Normalization check: High, Select, Search, and Low each sum to `1.0000` for every protocol, apart from rounding error.

# 4. Per-Query Performance
Original figure caption:
"Per-query performance of NQO relative to PostgreSQL on JOB, STACK, and TPC-H. Bars show the average execution-
time difference after aligning identical query IDs across available split settings, with error bars indicating the min-max range.
The blue and orange curves correspond to the number of joins and the baseline intermediate rows, respectively"

`Δt(q) = tPG(q) - tNQO(q)` in seconds; positive values mean NQO is faster and negative values mean it is slower. For SQL timeouts, `tNQO` uses the paper's charged value `Ttimeout = min(5 × tPG, 360 s)` rather than the slightly larger client-observed wall time after cancellation.

The final two columns below record fixed measurements from each query's PostgreSQL baseline plan: `# Joins` counts join operators (including nested loops), and `Intermediate Rows` sums `Actual Rows` over those operators.


## TPC-H (random)

Q7, Q8, Q9, Q13, Q15, and Q22 use the PostgreSQL fallback, so NQO time equals PostgreSQL time and `Δt=0`.

| Query | PG (s) | NQO (s) | Δt (s) | # Joins | Intermediate Rows |
|---:|---:|---:|---:|---:|---:|
| Q1 | 7.487681 | 7.262005 | +0.225675 | 0 | 0 |
| Q2 | 1.019194 | 1.049498 | -0.030305 | 7 | 6427 |
| Q3 | 2.053654 | 1.948401 | +0.105253 | 2 | 177645 |
| Q4 | 0.430747 | 0.456442 | -0.025695 | 1 | 52523 |
| Q5 | 0.591594 | 0.607364 | -0.015770 | 5 | 267521 |
| Q6 | 0.851789 | 0.801869 | +0.049920 | 0 | 0 |
| Q7 | 1.432795 | 1.432795 | +0.000000 | 5 | 285757 |
| Q8 | 0.530766 | 0.530766 | +0.000000 | 7 | 94848 |
| Q9 | 2.134364 | 2.134364 | +0.000000 | 5 | 766776 |
| Q10 | 1.982979 | 0.796966 | +1.186013 | 3 | 344115 |
| Q11 | 0.163192 | 0.207043 | -0.043851 | 4 | 64152 |
| Q12 | 1.482396 | 1.428760 | +0.053636 | 1 | 30988 |
| Q13 | 1.249158 | 1.249158 | +0.000000 | 1 | 1533923 |
| Q14 | 0.357844 | 0.351665 | +0.006179 | 1 | 75983 |
| Q15 | 0.493236 | 0.493236 | +0.000000 | 1 | 1 |
| Q16 | 0.908047 | 0.919466 | -0.011419 | 1 | 118274 |
| Q17 | 1.528642 | 1.348296 | +0.180347 | 1 | 587 |
| Q18 | 5.016275 | 4.950520 | +0.065755 | 3 | 513 |
| Q19 | 0.125197 | 0.186780 | -0.061583 | 1 | 121 |
| Q20 | 0.284877 | 0.396450 | -0.111572 | 3 | 6431 |
| Q21 | 1.458041 | 1.556591 | -0.098550 | 5 | 183507 |
| Q22 | 0.226055 | 0.226055 | +0.000000 | 1 | 6384 |

## JOB

Each protocol uses the WS-first best checkpoint for every fold: Base-query a/4, b/16, c/4; Leave-one-out a/4, b/24, c/0; Random a/0, b/28, c/4. Query IDs are aligned within each protocol before computing the cross-protocol mean and min--max of `Δt`. The duplicated 24a and 32a entries in Leave-one-out are averaged within that protocol first. PostgreSQL and NQO both use first-vs-first measurements.

| Protocol | WS | GS | Imp |
|---|---:|---:|---:|
| Base-query | 1.732855 | 0.977054 | 54/113 (47.79%) |
| Leave-one-out | 1.723802 | 0.986182 | 58/115 (50.43%) |
| Random | 1.789778 | 1.024788 | 60/113 (53.10%) |

Pairwise Pearson correlations of per-query `Δt` across the three protocols range from `0.9784` to `0.9845`. For 95/113 queries (84.07%), the direction is consistent across all protocols: 47 are always faster and 48 are always slower. The table therefore reports the cross-protocol mean with min--max ranges for protocol sensitivity.

| Query | Mean PG (s) | Mean NQO (s) | Mean Δt (s) | Min Δt (s) | Max Δt (s) | # Joins | Intermediate Rows |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1a | 0.125740 | 0.217240 | -0.091500 | -0.338693 | +0.033290 | 4 | 681 |
| 1b | 0.011885 | 0.138007 | -0.126122 | -0.126122 | -0.126122 | 4 | 83 |
| 1c | 0.019849 | 0.350984 | -0.331135 | -0.460509 | -0.072387 | 4 | 299 |
| 1d | 0.009565 | 0.089417 | -0.079852 | -0.090917 | -0.057722 | 4 | 98 |
| 2a | 2.242477 | 0.750375 | +1.492102 | +1.169136 | +1.786272 | 4 | 206060 |
| 2b | 0.542425 | 0.495280 | +0.047145 | -0.076259 | +0.108847 | 4 | 200848 |
| 2c | 0.408168 | 0.483616 | -0.075448 | -0.189223 | +0.011936 | 4 | 190392 |
| 2d | 2.104354 | 0.764721 | +1.339633 | +1.325360 | +1.346769 | 4 | 300548 |
| 3a | 1.450624 | 0.279922 | +1.170702 | +1.159492 | +1.176307 | 3 | 15392 |
| 3b | 0.109037 | 0.249392 | -0.140354 | -0.171336 | -0.100378 | 3 | 13361 |
| 3c | 1.851132 | 1.304294 | +0.546838 | -1.195420 | +1.419742 | 3 | 28075 |
| 4a | 0.311796 | 0.251012 | +0.060784 | +0.027459 | +0.084061 | 4 | 27944 |
| 4b | 0.263579 | 0.216145 | +0.047433 | +0.022843 | +0.059728 | 4 | 1453 |
| 4c | 0.117704 | 0.258365 | -0.140661 | -0.155275 | -0.111433 | 4 | 40342 |
| 5a | 0.229351 | 0.338869 | -0.109517 | -0.124892 | -0.078767 | 4 | 0 |
| 5b | 0.213582 | 0.319520 | -0.105938 | -0.105938 | -0.105938 | 4 | 0 |
| 5c | 0.616216 | 0.350845 | +0.265371 | +0.259136 | +0.277839 | 4 | 2903 |
| 6a | 0.304985 | 0.149997 | +0.154988 | +0.136564 | +0.180884 | 4 | 1255 |
| 6b | 0.784166 | 0.398113 | +0.386053 | +0.343121 | +0.407519 | 4 | 35979 |
| 6c | 0.031306 | 0.140114 | -0.108808 | -0.123564 | -0.090101 | 4 | 51 |
| 6d | 23.747169 | 1.272559 | +22.474610 | +22.344028 | +22.543569 | 4 | 835278 |
| 6e | 0.036193 | 0.143311 | -0.107118 | -0.114057 | -0.093240 | 4 | 1255 |
| 6f | 4.596369 | 3.772864 | +0.823505 | +0.479185 | +0.995665 | 4 | 1620667 |
| 7a | 2.774267 | 0.400250 | +2.374017 | +2.255888 | +2.438291 | 7 | 129552 |
| 7b | 0.320624 | 0.511059 | -0.190436 | -0.196899 | -0.177508 | 7 | 40152 |
| 7c | 6.670191 | 3.868569 | +2.801622 | +1.660222 | +3.431653 | 7 | 2093115 |
| 8a | 6.612681 | 1.698787 | +4.913895 | +4.384961 | +5.178362 | 6 | 141135 |
| 8b | 0.467544 | 0.681463 | -0.213919 | -0.257779 | -0.126199 | 6 | 8786 |
| 8c | 5.690254 | 5.751860 | -0.061606 | -1.544641 | +0.679911 | 6 | 11465466 |
| 8d | 2.412313 | 5.220129 | -2.807815 | -2.832737 | -2.772510 | 6 | 3478080 |
| 9a | 1.133110 | 1.274000 | -0.140889 | -0.954214 | +0.280132 | 7 | 3591 |
| 9b | 0.946451 | 1.179969 | -0.233519 | -0.880399 | +0.089922 | 7 | 2157 |
| 9c | 2.788395 | 2.305114 | +0.483282 | +0.434220 | +0.507813 | 7 | 319353 |
| 9d | 5.509525 | 4.743026 | +0.766499 | +0.411984 | +1.457746 | 7 | 2045563 |
| 10a | 1.062003 | 1.829743 | -0.767740 | -2.163155 | +0.059314 | 6 | 13800 |
| 10b | 0.550016 | 2.118907 | -1.568891 | -2.532910 | -0.068493 | 6 | 12663 |
| 10c | 9.237700 | 8.636476 | +0.601224 | -1.994218 | +1.898945 | 6 | 2532182 |
| 11a | 0.097216 | 0.241465 | -0.144250 | -0.218343 | -0.104502 | 7 | 12491 |
| 11b | 0.070259 | 0.237452 | -0.167194 | -0.226158 | -0.080976 | 7 | 10915 |
| 11c | 0.159270 | 0.389629 | -0.230360 | -0.504295 | -0.071826 | 7 | 53205 |
| 11d | 0.138537 | 0.357115 | -0.218578 | -0.471720 | -0.092006 | 7 | 50186 |
| 12a | 1.140036 | 0.655710 | +0.484326 | +0.373797 | +0.610033 | 7 | 117867 |
| 12b | 0.020850 | 0.129240 | -0.108390 | -0.169664 | -0.077753 | 7 | 311 |
| 12c | 4.895386 | 1.627272 | +3.268115 | +3.192751 | +3.319225 | 7 | 463253 |
| 13a | 4.301648 | 2.831462 | +1.470186 | +1.050017 | +1.783428 | 8 | 2849322 |
| 13b | 1.089247 | 1.293172 | -0.203925 | -0.271577 | -0.125446 | 8 | 463352 |
| 13c | 1.062295 | 1.305992 | -0.243697 | -0.276904 | -0.187128 | 8 | 460432 |
| 13d | 5.162560 | 3.536627 | +1.625933 | +0.704512 | +2.090354 | 8 | 7279691 |
| 14a | 0.624498 | 0.525962 | +0.098536 | +0.076584 | +0.109512 | 7 | 47805 |
| 14b | 0.186991 | 0.289746 | -0.102755 | -0.123395 | -0.092435 | 7 | 19565 |
| 14c | 0.729837 | 0.891949 | -0.162112 | -0.224188 | -0.106815 | 7 | 87464 |
| 15a | 0.934112 | 0.818031 | +0.116081 | +0.073202 | +0.201839 | 8 | 63497 |
| 15b | 0.096579 | 0.276378 | -0.179799 | -0.230420 | -0.078559 | 8 | 438 |
| 15c | 0.986267 | 0.788559 | +0.197707 | +0.185072 | +0.204025 | 8 | 29942 |
| 15d | 1.081597 | 0.938078 | +0.143519 | +0.110107 | +0.202878 | 8 | 117907 |
| 16a | 0.247626 | 0.400217 | -0.152591 | -0.184234 | -0.089305 | 7 | 43096 |
| 16b | 21.611640 | 6.184677 | +15.426963 | +15.113886 | +15.692833 | 7 | 9676250 |
| 16c | 1.735686 | 1.714668 | +0.021018 | -0.063668 | +0.063360 | 7 | 831860 |
| 16d | 1.408553 | 1.454121 | -0.045567 | -0.151730 | +0.044026 | 7 | 650653 |
| 17a | 11.863255 | 5.078983 | +6.784272 | +6.416126 | +7.340921 | 6 | 3391392 |
| 17b | 7.820629 | 5.622375 | +2.198254 | +2.143885 | +2.227073 | 6 | 1198129 |
| 17c | 7.307304 | 1.125707 | +6.181597 | +6.036821 | +6.301540 | 6 | 1086237 |
| 17d | 7.518436 | 1.286089 | +6.232347 | +6.186574 | +6.255233 | 6 | 1116819 |
| 17e | 10.082962 | 5.429529 | +4.653433 | +4.486154 | +4.987992 | 6 | 5965658 |
| 17f | 8.577062 | 4.600845 | +3.976217 | +3.474181 | +4.232999 | 6 | 3497807 |
| 18a | 9.363224 | 7.400135 | +1.963089 | +1.344335 | +2.718622 | 6 | 7543849 |
| 18b | 0.399335 | 0.768440 | -0.369105 | -0.409552 | -0.288211 | 6 | 22996 |
| 18c | 9.879471 | 7.293969 | +2.585503 | +2.385962 | +2.984583 | 6 | 852077 |
| 19a | 0.981472 | 1.393531 | -0.412060 | -1.324479 | +0.044150 | 9 | 3149 |
| 19b | 0.511049 | 0.705587 | -0.194539 | -0.240419 | -0.102777 | 9 | 1509 |
| 19c | 1.972019 | 2.908050 | -0.936031 | -1.497600 | -0.252599 | 9 | 307855 |
| 19d | 12.959959 | 6.709223 | +6.250736 | +6.121199 | +6.509810 | 9 | 8981311 |
| 20a | 4.446739 | 2.399379 | +2.047360 | +1.953925 | +2.094077 | 9 | 612142 |
| 20b | 2.001978 | 1.179599 | +0.822379 | +0.563771 | +1.029001 | 9 | 282977 |
| 20c | 1.157161 | 1.164987 | -0.007826 | -0.023632 | +0.000077 | 9 | 244411 |
| 21a | 0.115174 | 0.314890 | -0.199716 | -0.253209 | -0.169670 | 8 | 21180 |
| 21b | 0.113432 | 0.259560 | -0.146128 | -0.225947 | -0.092220 | 8 | 12534 |
| 21c | 0.125217 | 0.360819 | -0.235601 | -0.370779 | -0.117833 | 8 | 17127 |
| 22a | 0.534385 | 0.740362 | -0.205977 | -0.427333 | -0.095299 | 10 | 73790 |
| 22b | 0.369145 | 0.600197 | -0.231052 | -0.404444 | -0.108496 | 10 | 52483 |
| 22c | 2.017126 | 1.399776 | +0.617350 | -0.018737 | +0.965210 | 10 | 196907 |
| 22d | 0.940366 | 1.122344 | -0.181978 | -0.221204 | -0.118957 | 10 | 244874 |
| 23a | 0.412200 | 0.772908 | -0.360708 | -0.399720 | -0.282684 | 10 | 302737 |
| 23b | 0.124305 | 0.628709 | -0.504404 | -0.812739 | -0.085379 | 10 | 4172 |
| 23c | 0.506240 | 0.769687 | -0.263447 | -0.354391 | -0.212488 | 10 | 37358 |
| 24a | 0.396160 | 0.566162 | -0.170002 | -0.228893 | -0.103168 | 11 | 42725 |
| 24b | 0.434322 | 0.404830 | +0.029492 | +0.029492 | +0.029492 | 11 | 38578 |
| 25a | 3.026887 | 1.945347 | +1.081539 | +0.858889 | +1.306333 | 8 | 301502 |
| 25b | 0.415516 | 0.665162 | -0.249647 | -0.333360 | -0.207790 | 8 | 62153 |
| 25c | 6.725454 | 3.783334 | +2.942120 | +2.716777 | +3.149900 | 8 | 558428 |
| 26a | 1.567827 | 1.050913 | +0.516914 | +0.084805 | +0.764460 | 11 | 655613 |
| 26b | 0.253062 | 0.472380 | -0.219317 | -0.297213 | -0.063526 | 11 | 22835 |
| 26c | 4.432995 | 1.179649 | +3.253345 | +3.108669 | +3.325684 | 11 | 1987955 |
| 27a | 0.121552 | 0.237158 | -0.115606 | -0.127551 | -0.096275 | 11 | 12656 |
| 27b | 0.097974 | 0.321659 | -0.223686 | -0.352253 | -0.152913 | 11 | 7230 |
| 27c | 0.122231 | 0.350631 | -0.228400 | -0.294669 | -0.194642 | 11 | 22612 |
| 28a | 1.045241 | 0.859845 | +0.185396 | +0.017177 | +0.357465 | 13 | 239048 |
| 28b | 0.681756 | 0.810255 | -0.128499 | -0.403668 | +0.138470 | 13 | 101874 |
| 28c | 1.357851 | 0.965874 | +0.391977 | -0.105052 | +0.723259 | 13 | 156145 |
| 29a | 2.289833 | 0.449789 | +1.840044 | +1.840044 | +1.840044 | 16 | 61210 |
| 29b | 2.185846 | 0.359080 | +1.826766 | +1.770208 | +1.939880 | 16 | 50665 |
| 29c | 9.935960 | 0.504440 | +9.431520 | +9.281524 | +9.521881 | 16 | 836394 |
| 30a | 3.002849 | 2.259276 | +0.743573 | -0.608245 | +1.492281 | 11 | 430426 |
| 30b | 0.575853 | 0.795178 | -0.219325 | -0.236773 | -0.184429 | 11 | 77063 |
| 30c | 2.905366 | 2.147071 | +0.758295 | +0.720183 | +0.777350 | 11 | 563064 |
| 31a | 5.913183 | 0.951532 | +4.961651 | +4.922797 | +5.007092 | 10 | 691547 |
| 31b | 0.537854 | 0.791638 | -0.253784 | -0.317769 | -0.125813 | 10 | 77600 |
| 31c | 8.280487 | 1.122276 | +7.158211 | +7.121896 | +7.230842 | 10 | 1104274 |
| 32a | 0.024523 | 0.117223 | -0.092699 | -0.101976 | -0.081646 | 5 | 1 |
| 32b | 0.173895 | 0.408037 | -0.234141 | -0.328879 | -0.051323 | 5 | 59392 |
| 33a | 0.320330 | 0.969472 | -0.649141 | -1.281321 | -0.308424 | 13 | 15622 |
| 33b | 0.308468 | 0.719783 | -0.411315 | -1.233874 | +0.136428 | 13 | 15619 |
| 33c | 0.234909 | 0.668680 | -0.433771 | -0.452884 | -0.412705 | 13 | 17971 |

## STACK

Each protocol uses the WS-first best checkpoint for every fold: Base-query a/16, b/16, c/0; Leave-one-out a/4, b/12, c/12; Random a/42, b/18, c/22. Results are reproduced strictly offline from versioned checkpoints and light buffers bound to complete trajectories; measurements are never substituted across policies. PostgreSQL and NQO both use first-run measurements. All 112 query IDs are unique within each protocol, after which the mean and min--max of `Δt` are computed across protocols.

| Protocol | WS_first | GS_first | Imp_first |
|---|---:|---:|---:|
| Base-query | 1.511860 | 0.910650 | 43/112 (38.39%) |
| Leave-one-out | 1.681650 | 0.936187 | 44/112 (39.29%) |
| Random | 1.635543 | 0.904246 | 42/112 (37.50%) |

Pairwise Pearson correlations of per-query `Δt` across the three protocols range from `0.9044` to `0.9773`. For 90/112 queries (80.36%), the direction is consistent across all protocols: 30 are always faster and 60 are always slower.

| Query | Mean PG (s) | Mean NQO (s) | Mean Δt (s) | Min Δt (s) | Max Δt (s) | # Joins | Intermediate Rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| q1_q1-009 | 0.012445 | 0.063774 | -0.051329 | -0.054428 | -0.049780 | 3 | 55 |
| q1_q1-031 | 0.010677 | 0.053136 | -0.042459 | -0.042706 | -0.041966 | 3 | 37 |
| q1_q1-035 | 0.012086 | 0.059333 | -0.047247 | -0.048345 | -0.045050 | 3 | 69 |
| q1_q1-067 | 0.075540 | 0.172508 | -0.096968 | -0.149190 | -0.070857 | 3 | 11013 |
| q1_q1-075 | 0.108797 | 0.280266 | -0.171469 | -0.275672 | -0.085562 | 3 | 18303 |
| q1_q1-098 | 3.713282 | 8.292088 | -4.578806 | -11.822999 | -0.956709 | 3 | 1763869 |
| q1_q1-099 | 3.067338 | 8.127463 | -5.060125 | -12.269351 | -1.455512 | 3 | 1321941 |
| q1_q1-100 | 3.408977 | 7.808222 | -4.399245 | -11.786625 | -0.705556 | 3 | 1494549 |
| q2_q2-001 | 16.922306 | 1.106910 | +15.815396 | +15.727063 | +15.913602 | 10 | 3661881 |
| q2_q2-012 | 20.593424 | 1.037819 | +19.555605 | +19.467727 | +19.725646 | 10 | 1719978 |
| q2_q2-032 | 2.810698 | 0.784831 | +2.025867 | +1.945357 | +2.107565 | 10 | 467921 |
| q2_q2-035 | 3.532598 | 0.908569 | +2.624029 | +2.548988 | +2.759722 | 10 | 712154 |
| q2_q2-050 | 1.364017 | 0.702474 | +0.661543 | +0.376910 | +0.886768 | 10 | 267667 |
| q2_q2-081 | 16.500325 | 0.988333 | +15.511991 | +15.425086 | +15.569049 | 10 | 1748579 |
| q2_q2-094 | 10.710184 | 1.027461 | +9.682723 | +9.574866 | +9.851055 | 10 | 1228601 |
| q2_q2-098 | 18.999795 | 3.794517 | +15.205277 | +15.017377 | +15.327955 | 10 | 7623485 |
| q3_q3-018 | 1.620035 | 1.174558 | +0.445477 | +0.188770 | +0.591197 | 11 | 197727 |
| q3_q3-040 | 11.670508 | 7.725662 | +3.944846 | +3.194851 | +4.727508 | 11 | 1434769 |
| q3_q3-043 | 1.853586 | 1.495154 | +0.358432 | +0.134765 | +0.553128 | 11 | 193021 |
| q3_q3-046 | 1.039924 | 1.047158 | -0.007233 | -0.108342 | +0.043321 | 11 | 55356 |
| q3_q3-066 | 3.293494 | 1.074342 | +2.219153 | +2.020659 | +2.395574 | 11 | 647787 |
| q3_q3-068 | 1.527367 | 1.455585 | +0.071782 | -0.301332 | +0.258339 | 11 | 146499 |
| q3_q3-086 | 1.456392 | 1.071357 | +0.385036 | +0.239169 | +0.525450 | 11 | 289795 |
| q3_q3-099 | 26.959115 | 0.509045 | +26.450070 | +26.405775 | +26.481981 | 11 | 470209 |
| q4_q4-002 | 0.216472 | 0.257110 | -0.040638 | -0.136192 | +0.007139 | 6 | 12566 |
| q4_q4-026 | 0.106606 | 0.225789 | -0.119184 | -0.203873 | -0.076839 | 6 | 10189 |
| q4_q4-041 | 0.238894 | 0.422179 | -0.183285 | -0.217069 | -0.115718 | 6 | 32490 |
| q4_q4-042 | 0.092879 | 0.249015 | -0.156136 | -0.308343 | -0.080032 | 6 | 8642 |
| q4_q4-064 | 0.083010 | 0.181453 | -0.098443 | -0.157923 | -0.068703 | 6 | 8448 |
| q4_q4-074 | 0.325041 | 0.293354 | +0.031686 | -0.031724 | +0.063392 | 6 | 21537 |
| q4_q4-086 | 0.131465 | 0.264622 | -0.133156 | -0.176551 | -0.077479 | 6 | 23856 |
| q4_q4-089 | 0.176207 | 0.311576 | -0.135369 | -0.172526 | -0.116790 | 6 | 23795 |
| q5_q5-015 | 0.024831 | 0.101004 | -0.076172 | -0.099325 | -0.038824 | 6 | 339 |
| q5_q5-032 | 0.041041 | 0.125529 | -0.084488 | -0.087055 | -0.083205 | 6 | 843 |
| q5_q5-041 | 0.057527 | 0.156324 | -0.098798 | -0.103905 | -0.088583 | 6 | 2045 |
| q5_q5-052 | 0.037278 | 0.117140 | -0.079862 | -0.110854 | -0.059629 | 6 | 428 |
| q5_q5-059 | 0.051151 | 0.156056 | -0.104905 | -0.112762 | -0.090580 | 6 | 1572 |
| q5_q5-077 | 0.037617 | 0.121006 | -0.083389 | -0.129952 | -0.039709 | 6 | 430 |
| q5_q5-079 | 0.122508 | 0.215514 | -0.093006 | -0.108900 | -0.061217 | 6 | 5211 |
| q5_q5-082 | 0.063338 | 0.147535 | -0.084197 | -0.088999 | -0.074594 | 6 | 2144 |
| q6_q6-002 | 0.094753 | 0.194279 | -0.099525 | -0.152973 | -0.039673 | 6 | 1396 |
| q6_q6-009 | 0.036420 | 0.084755 | -0.048335 | -0.081151 | -0.031385 | 6 | 384 |
| q6_q6-060 | 0.108854 | 0.167040 | -0.058185 | -0.073620 | -0.036056 | 6 | 1513 |
| q6_q6-064 | 0.055231 | 0.080380 | -0.025149 | -0.073306 | -0.001070 | 6 | 524 |
| q6_q6-065 | 0.039655 | 0.142989 | -0.103334 | -0.150621 | -0.074292 | 6 | 264 |
| q6_q6-067 | 0.039965 | 0.149725 | -0.109760 | -0.131814 | -0.083110 | 6 | 586 |
| q6_q6-069 | 0.040823 | 0.128525 | -0.087702 | -0.124924 | -0.031480 | 6 | 289 |
| q6_q6-085 | 0.037958 | 0.150236 | -0.112278 | -0.150777 | -0.081878 | 6 | 169 |
| q7_q7-034 | 7.602097 | 6.135869 | +1.466228 | -0.005840 | +2.202262 | 3 | 77 |
| q7_q7-036 | 4.312949 | 5.758334 | -1.445386 | -2.055249 | -1.140454 | 3 | 0 |
| q7_q7-047 | 4.721473 | 6.122447 | -1.400973 | -2.343742 | -0.929589 | 3 | 0 |
| q7_q7-077 | 4.245826 | 5.594441 | -1.348615 | -1.430435 | -1.307705 | 3 | 18 |
| q7_q7-082 | 4.327683 | 5.855331 | -1.527648 | -2.388943 | -1.097001 | 3 | 80 |
| q7_q7-085 | 5.650465 | 6.448128 | -0.797663 | -2.106502 | +0.016028 | 3 | 8998 |
| q7_q7-095 | 5.144638 | 5.644801 | -0.500163 | -0.634361 | -0.433064 | 3 | 2566 |
| q7_q7-099 | 5.233617 | 5.966564 | -0.732947 | -1.446572 | -0.253876 | 3 | 2759 |
| q8_q8-006 | 0.318771 | 0.769412 | -0.450641 | -1.275082 | -0.021179 | 8 | 0 |
| q8_q8-025 | 0.315780 | 0.758635 | -0.442854 | -1.263122 | -0.014197 | 8 | 0 |
| q8_q8-046 | 0.297433 | 0.723620 | -0.426187 | -1.189734 | -0.044413 | 8 | 0 |
| q8_q8-062 | 0.315331 | 0.756157 | -0.440826 | -1.261326 | -0.011409 | 8 | 0 |
| q8_q8-065 | 0.317713 | 0.771293 | -0.453579 | -1.270853 | -0.044942 | 8 | 0 |
| q8_q8-074 | 0.306207 | 0.754223 | -0.448016 | -1.224827 | -0.059611 | 8 | 0 |
| q8_q8-076 | 0.297207 | 0.726018 | -0.428811 | -1.188829 | -0.045729 | 8 | 0 |
| q8_q8-096 | 0.312470 | 0.764926 | -0.452456 | -1.249880 | -0.040306 | 8 | 0 |
| q11_0ea8bacd | 0.284417 | 0.279204 | +0.005213 | -0.000355 | +0.007997 | 3 | 21894 |
| q11_33e1caf2 | 0.108129 | 0.161473 | -0.053344 | -0.060455 | -0.039122 | 3 | 12909 |
| q11_6c5cba41 | 0.231520 | 0.270162 | -0.038642 | -0.042090 | -0.031746 | 3 | 12449 |
| q11_87c4bd09 | 0.960816 | 0.878002 | +0.082815 | +0.082815 | +0.082815 | 3 | 59257 |
| q11_9389f588 | 0.069343 | 0.132179 | -0.062837 | -0.077178 | -0.034153 | 3 | 7630 |
| q11_aa96c8d7 | 0.066775 | 0.139376 | -0.072601 | -0.121182 | -0.040483 | 3 | 9068 |
| q11_c1ae2a99 | 5.325720 | 4.103431 | +1.222289 | +0.707114 | +1.479877 | 3 | 452539 |
| q11_e4ca3559 | 0.143115 | 0.187337 | -0.044222 | -0.046459 | -0.039750 | 3 | 8948 |
| q12_06c8d688 | 0.168147 | 0.251711 | -0.083563 | -0.132159 | -0.027420 | 5 | 18797 |
| q12_07007205 | 0.359390 | 0.691998 | -0.332609 | -0.401155 | -0.229618 | 5 | 41175 |
| q12_547c6bf1 | 0.085613 | 0.209528 | -0.123916 | -0.169563 | -0.076019 | 5 | 8175 |
| q12_55de941e | 4.205616 | 3.323189 | +0.882428 | -0.521066 | +2.301226 | 5 | 163820 |
| q12_5a5ff9bd | 0.395064 | 0.476988 | -0.081923 | -0.152441 | -0.014145 | 5 | 18431 |
| q12_76a47868 | 1.608695 | 0.881549 | +0.727146 | +0.638714 | +0.832090 | 5 | 69838 |
| q12_812a3eff | 1.818125 | 1.567551 | +0.250575 | +0.190115 | +0.282433 | 5 | 261917 |
| q12_bde6c0cf | 0.108590 | 0.217592 | -0.109003 | -0.171949 | -0.057574 | 5 | 15767 |
| q13_13ad1b8c | 1.014010 | 0.789511 | +0.224499 | -0.158209 | +0.417340 | 7 | 116762 |
| q13_1ddcc865 | 3.133786 | 2.690567 | +0.443219 | -1.014342 | +1.220512 | 7 | 121507 |
| q13_935e2051 | 16.069960 | 15.664823 | +0.405137 | -0.278741 | +1.679581 | 7 | 924081 |
| q13_a091adce | 0.415625 | 0.553949 | -0.138324 | -0.197127 | -0.075555 | 7 | 65231 |
| q13_a3d03772 | 1.696597 | 1.281734 | +0.414863 | -0.161868 | +0.782256 | 7 | 169455 |
| q13_add0df9d | 1.492243 | 1.153688 | +0.338555 | -0.325381 | +0.685399 | 7 | 141963 |
| q13_d383cd5b | 0.410328 | 0.422158 | -0.011829 | -0.117935 | +0.096420 | 7 | 11467 |
| q13_d4707be2 | 0.883495 | 0.936484 | -0.052989 | -0.218684 | +0.039774 | 7 | 40441 |
| q14_4063b6cb | 0.549361 | 0.528001 | +0.021360 | -0.023708 | +0.077195 | 7 | 68647 |
| q14_5dbc1d1f | 7.919037 | 8.955867 | -1.036829 | -5.601222 | +1.588809 | 7 | 571730 |
| q14_5e4835cd | 0.265624 | 0.450302 | -0.184679 | -0.239526 | -0.144567 | 7 | 37626 |
| q14_63c0776f | 0.207315 | 0.336949 | -0.129634 | -0.133471 | -0.125129 | 7 | 11362 |
| q14_719e692d | 1.012952 | 1.171178 | -0.158226 | -0.622373 | +0.085680 | 7 | 129500 |
| q14_74fd1af6 | 8.084465 | 4.994514 | +3.089951 | +2.229031 | +3.877103 | 7 | 339699 |
| q14_97e68ad5 | 7.904176 | 5.277285 | +2.626891 | +1.811299 | +3.714826 | 7 | 145610 |
| q14_b49361f8 | 0.318124 | 0.402833 | -0.084710 | -0.087142 | -0.082119 | 7 | 50771 |
| q15_21e4988a | 0.258500 | 0.392361 | -0.133861 | -0.301903 | -0.010761 | 6 | 17916 |
| q15_3e37e626 | 0.521226 | 0.485242 | +0.035985 | -0.111417 | +0.120125 | 6 | 27914 |
| q15_543ab3f7 | 0.307012 | 0.206988 | +0.100024 | -0.055514 | +0.199136 | 6 | 84486 |
| q15_78995a5f | 0.466083 | 0.536624 | -0.070541 | -0.203013 | +0.064799 | 6 | 67541 |
| q15_b2ee2c78 | 0.578476 | 0.641492 | -0.063016 | -0.233008 | +0.115212 | 6 | 61311 |
| q15_b8ddf65b | 0.532303 | 0.238109 | +0.294195 | +0.276177 | +0.328148 | 6 | 187592 |
| q15_c9619ad4 | 0.833095 | 0.622986 | +0.210108 | +0.143024 | +0.321940 | 6 | 128752 |
| q15_d5546c01 | 0.245911 | 0.170862 | +0.075049 | +0.053161 | +0.099778 | 6 | 56668 |
| q16_1e863562 | 7.932550 | 3.139926 | +4.792624 | +4.352103 | +5.012884 | 6 | 1289821 |
| q16_374e3e4c | 10.883960 | 5.983749 | +4.900211 | +4.858388 | +4.983858 | 6 | 1946552 |
| q16_b1a96cd4 | 1.375021 | 0.751202 | +0.623819 | +0.617706 | +0.636044 | 6 | 256126 |
| q16_d5290889 | 0.286011 | 0.161079 | +0.124932 | +0.086866 | +0.153979 | 6 | 30864 |
| q16_ea9efde5 | 0.710175 | 0.232585 | +0.477590 | +0.461077 | +0.486119 | 6 | 99300 |
| q16_ed2ffeae | 3.090743 | 1.355756 | +1.734988 | +1.611024 | +1.796969 | 6 | 503323 |
| q16_f67cec3d | 0.316598 | 2.341832 | -2.025234 | -3.542921 | -1.266390 | 6 | 35949 |
| q16_fbe34e8f | 9.082508 | 2.586627 | +6.495881 | +5.765896 | +7.353138 | 6 | 1597039 |

# 5. RL Formulation and State Ablation

Original experiment title: “Comparison of different RL formulations and state representations on the Random split.”

All three datasets use only the Random split, with the fold iteration of Full NQO as the training budget. Released checkpoints are: One-step RL, JOB `0/4/4` and STACK `0/0/0`; w/o Query Topology, JOB `0/4/4`, STACK `0/0/4`, and TPC-H `0/40/0`; w/o Plan Topology, JOB `0/20/4`, STACK `0/0/0`, and TPC-H `0/16/0`. WS is not averaged arithmetically across folds; it is aggregated as `WS = ΣPG / ΣNQO`. GS and Imp are also computed over the combined test queries. Results include inference overhead. TPC-H adds only the state ablations and does not repeat One-step RL.

`One-step RL` retains the same hierarchical action space and network but sets `γ=0, λ=0` to remove cross-step credit assignment. `w/o Query Topology` retains query-node features while removing query-graph edges. `w/o Plan Topology` retains plan-operator features while flattening the plan-tree topology.

| Method | JOB WS | JOB GS | JOB Imp | STACK WS | STACK GS | STACK Imp | TPC-H WS | TPC-H GS | TPC-H Imp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full NQO | 1.789778 | 1.024788 | 60/113 (53.10%) | 1.635543 | 0.904246 | 42/112 (37.50%) | 1.048593 | 1.004016 | 8/22 (36.36%) |
| One-step RL | 1.654796 | 0.926309 | 55/113 (48.67%) | 1.462874 | 0.945720 | 40/112 (35.71%) | — | — | — |
| w/o Query Topology | 1.714203 | 0.971524 | 51/113 (45.13%) | 1.424510 | 0.947381 | 38/112 (33.93%) | 1.009295 | 0.961522 | 8/22 (36.36%) |
| w/o Plan Topology | 1.532734 | 0.930065 | 50/113 (44.25%) | 1.472904 | 0.950055 | 40/112 (35.71%) | 1.009286 | 0.961485 | 8/22 (36.36%) |

All three TPC-H configurations outperform PostgreSQL on the same eight queries (Q1, Q3, Q6, Q10, Q12, Q14, Q17, and Q18), so each has Imp `8/22 (36.36%)`; the results are not reused. State topology primarily affects speedup magnitude rather than the number of improved queries. For example, Q10 improves from `1.0560×` under either topology ablation to `2.4882×` under Full NQO, substantially raising aggregate WS/GS. Full NQO and the two ablations predict different action trajectories on 9/22 queries, while the two ablations differ only on Q14. Q7, Q8, Q9, Q13, Q15, and Q22 are deterministic bypasses with PostgreSQL-equivalent runtime and are not counted in Imp under the strict `t_NQO < t_PG` definition.

# 6. Cross-workload Transferability

Original figure caption: “Cross-workload transferability of NQO: same-workload training vs. transfer from other source workloads.”

All results use the Random split and compute `WS_first = ΣPG_first / ΣNQO_first` from total time across three folds. `Cross-workload` applies the reported source-workload checkpoints directly to the target without target-workload fine-tuning. For JOB-to-STACK transfer, folds A/B/C use `best`, `online-iter-0016.pt`, and `online-iter-0000.pt`, respectively; the other directions use the source-best checkpoint in all three folds. `Mixed-workload` trains on combined JOB, STACK, and TPC-H data. An em dash indicates that source and target are identical and the result already appears under `Same-workload`.

| Target workload | Same-workload | Mixed-workload | Cross: source=JOB | Cross: source=STACK | Cross: source=TPC-H |
|---|---:|---:|---:|---:|---:|
| JOB | 1.789778 | 1.626803 | — | 1.473394 | 1.205701 |
| STACK | 1.635543 | 1.391619 | 1.033773 | — | 0.468880 |
| TPC-H | 1.048593 | 1.035380 | 1.032013 | 0.998605 | — |

For completeness, the following table reports the evaluated JOB-to-STACK checkpoint combinations, ordered by folds A/B/C. The main matrix uses `best-0016-0000`; `best` denotes the best checkpoint for that fold, while `0016` and `0000` denote `online-iter-0016.pt` and `online-iter-0000.pt`, respectively.

| Source checkpoints (A/B/C) | Target | Queries | Cache Hits | WS | GS | Imp |
|---|---|---:|---:|---:|---:|---:|
| JOB (`best-best-best`) | STACK | 112 | 112 | 0.824509 | 0.732409 | 38/112 (33.93%) |
| JOB (`best-best-0000`) | STACK | 112 | 112 | 0.847505 | 0.737966 | 39/112 (34.82%) |
| JOB (`best-0016-best`) | STACK | 112 | 108 | 0.999762 | 0.760795 | 36/112 (32.14%) |
| JOB (`best-0016-0000`) | STACK | 112 | 108 | 1.033773 | 0.766568 | 37/112 (33.04%) |

Per-query data is stored in `results/benchmark/nqo/nqo_transfer_run.csv`, with summaries in `results/benchmark/nqo/analyze_nqo_transfer.log`. Zero-shot evaluation contains 20 checkpoint-fold evaluations and 569 results: 565 cache hits and four actual database executions. The nine mixed-workload GPU prediction tasks produce 247 results, all from cache hits.

## Data-scale robustness

Full-data Random checkpoints are evaluated directly on scaled database instances without retraining. JOB retains 75%/50%/25% of titles selected by a fixed hash together with their reference closures. STACK retains 50% of question threads and their answers, comments, tags, and links. PostgreSQL and NQO execute every SQL statement once using a fresh connection. A PostgreSQL baseline timeout removes the query from paired WS, GS, and Imp calculations; an NQO timeout still uses its official charged runtime. Only `q2_q2-098` in STACK Thread 50% triggers a PostgreSQL timeout, so that group uses the other 111 queries; no other group has a PostgreSQL timeout. `Inf. overhead` is computed directly as `Σ inference_ms / Σ NQO end-to-end runtime_ms` over valid paired queries and includes inference completed before an NQO timeout.

| Workload | Data instance | Pairs | WS | GS | Imp | Inf. overhead | WS w/o Inf. | GS w/o Inf. | Imp w/o Inf. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JOB | Full | 113 | 1.789778 | 1.024788 | 60/113 (53.10%) | 7.11% | 1.924302 | 1.350700 | 63/113 (55.75%) |
| JOB | Title 75% | 113 | 1.306544 | 0.739016 | 31/113 (27.43%) | 16.51% | 1.550529 | 1.100518 | 50/113 (44.25%) |
| JOB | Title 50% | 113 | 1.184680 | 0.658596 | 29/113 (25.66%) | 20.03% | 1.461363 | 1.020628 | 47/113 (41.59%) |
| JOB | Title 25% | 113 | 0.973733 | 0.552927 | 22/113 (19.47%) | 28.64% | 1.321412 | 0.927299 | 41/113 (36.28%) |
| STACK | Full | 112 | 1.635543 | 0.904246 | 42/112 (37.50%) | 3.45% | 1.688047 | 1.091101 | 49/112 (43.75%) |
| STACK | Thread 50% | 111 | 1.300537 | 0.636677 | 32/111 (28.83%) | 22.99% | 1.653193 | 1.156899 | 65/111 (58.56%) |

The full STACK database and Thread 50% instance contain the same 33 permanent indexes with identical definitions; only statistics are refreshed on the scaled instance. Per-query results and independent light caches are stored in `results/benchmark/nqo/nqo_{job,stack}_scale_*_runs.csv` and `results/buffers/{job,stack}_scale_*`. `results/benchmark/nqo/analyze_nqo_data_scale.py` writes the summary to `analyze_nqo_data_scale.log`.

# 7. Decomposition-depth microbenchmark

The microbenchmark fixes `Select alpha=0.5`, `Search=default`, and `Low=none`, changing only the High-head stop position. Depth `d` forces `d` splits followed by stop; a successful run therefore has `d+1` executable units (`d` materialized subqueries plus one residual query). Every query enters the complete PostgreSQL decomposition/materialization/rewrite/replanning path from its original SQL rather than executing a manually decomposed statement. JOB and STACK each contribute three queries with different structures and benefit trends. PostgreSQL and NQO both use first-run measurements, and timeouts are charged as `min(5×PG, 360s)`.

The full experiment contains 41 fixed-depth points: 20 reuse existing complete trajectories, while 21 cache misses execute once and are written to a dedicated cache. Raw data is in `results/benchmark/nqo/nqo_decomposition_depth.csv`, with a readable summary in `results/benchmark/nqo/analyze_nqo_decomposition_depth.log`.

| Dataset | Query | Tested split depths | Speedup at each depth |
|---|---|---|---|
| JOB | 29a | 0/1/2/3/4/5/6 | 0.9648 / 3.6059 / 13.5244 / 10.8081 / 5.6065 / **16.1383** / 5.0909 |
| JOB | 22c | 0/1/2/3/4 | 0.9908 / 0.9566 / 0.7774 / 0.7996 / **1.4684** |
| JOB | 33a | 0/1/2/3/4/5/6 | **0.8024** / 0.6531 / 0.7093 / 0.7394 / 0.7423 / 0.7401 / 0.4214 |
| STACK | q3_q3-099 | 0/1/2/3/4/5/6/7 | 2.3271 / 33.0584 / 10.5530 / 10.6240 / 10.9902 / 10.9822 / 56.4141 / **61.4139** |
| STACK | q2_q2-012 | 0/1/2/3/4/5/6/7 | 1.0529 / 2.4861 / **20.9506** / 3.4953 / 3.5995 / 3.9100 / 20.2883 / 16.4259 |
| STACK | q13_935e2051 | 0/1/2/3/4/5 | 0.9830 / **2.9293** / 0.9393 / 0.9694 / 0.5653 / 0.4600 |

| Dataset | Query | Best splits | Best executed units | Best speedup | Deepest speedup |
|---|---|---:|---:|---:|---:|
| JOB | 29a | 5 | 6 | 16.138253 | 5.090905 |
| JOB | 22c | 4 | 5 | 1.468446 | 1.468446 |
| JOB | 33a | 0 | 1 | 0.802444 | 0.421380 |
| STACK | q3_q3-099 | 7 | 8 | 61.413860 | 61.413860 |
| STACK | q2_q2-012 | 2 | 3 | 20.950610 | 16.425863 |
| STACK | q13_935e2051 | 1 | 2 | 2.929293 | 0.459969 |

These results do not imply that more decomposition is always faster. The optimal stopping depth varies by query: some queries need deep decomposition to expose a beneficial residual plan, while others incur additional materialization and replanning cost or even time out. This directly supports learning per-round `split/stop` decisions with the High head instead of using a fixed decomposition depth.

# 8. Resource and runtime overhead

Results are recomputed offline by `results/benchmark/nqo/analyze_nqo_overhead.py`, with summaries in `results/benchmark/nqo/analyze_nqo_overhead.log`. First-run independent-action records come from `results/benchmark/nqo/nqo_independent_action_runs.csv`. NQO uses the official Random-protocol results in `nqo_runs.csv` and reads the corresponding execution events from versioned light buffers through `cache_id`.

## Materialization footprint

`Avg. materializations/query (#)` is the number of intermediate results materialized per original SQL statement. `Avg. materialized data/query (MiB)` is the cumulative materialized data produced per original statement, not peak process memory. `Materialized-data reduction (%)` uses QuerySplit as the baseline and is computed as `1 - NQO MiB/query / QuerySplit MiB/query`.

| Workload | Method | Avg. materializations/query (#) | Avg. materialized data/query (MiB) | Materialized-data reduction (%) |
|---|---|---:|---:|---:|
| JOB | QuerySplit | 3.68 | 6.80 | — |
| JOB | NQO | **1.71** | **3.88** | **42.9%** |
| STACK | QuerySplit | 4.99 | 5.97 | — |
| STACK | NQO | **3.04** | **2.47** | **58.7%** |

TPC-H does not enable Query Split, so no materialization footprint is reported. By learning to stop earlier, NQO reduces both materialization count and cumulative materialized data per query.

## NQO-specific runtime overhead

The values below are measured component times as percentages of end-to-end NQO runtime. Decomposition bookkeeping includes only `ANALYZE` and residual rewriting. Materialized-subquery execution, normal PostgreSQL planning, and final query execution are part of query processing and are not counted as NQO-specific overhead.

| Workload | #Q | Inference | LIP build | AJA build | Search | Decomposition bookkeeping | Total measured |
|---|---:|---:|---:|---:|---:|---:|---:|
| JOB | 113 | 7.11% | 0.25% | 0.12% | 0.00% | 1.23% | **8.70%** |
| STACK | 112 | 3.45% | 0.00% | 0.07% | 0.00% | 0.84% | **4.36%** |
| TPC-H | 22 | 0.32% | 0.00% | 3.61% | 0.08% | 0.00% | **4.00%** |

# 9. Alpha scheduling sensitivity

For the three versioned best JOB Random checkpoints, High, Search, and Low remain unchanged while every Select decision is fixed to `alpha=0`, `0.5`, or `1` and compared with the model's state-dependent learned `alpha_t`. All configurations use the same test folds, PostgreSQL baselines, first-run runtimes, and timeout charging. The three folds are aggregated by total time rather than by the arithmetic mean of fold WS values.

| Schedule policy | WS | GS | Imp | NQO total runtime (s) | Runtime / learned |
|---|---:|---:|---:|---:|---:|
| Fixed alpha=0.0 | 1.581317 | 0.920016 | 53/113 (46.90%) | 185.637 | 1.1318 |
| Fixed alpha=0.5 | 1.777505 | 1.008728 | 59/113 (52.21%) | 165.148 | 1.0069 |
| Fixed alpha=1.0 | 1.630586 | 0.976257 | 54/113 (47.79%) | 180.028 | 1.0976 |
| Learned alpha_t | **1.789778** | **1.024788** | **60/113 (53.10%)** | **164.015** | **1.0000** |

| Schedule policy | Fold a WS | Fold b WS | Fold c WS |
|---|---:|---:|---:|
| Fixed alpha=0.0 | 1.304052 | 1.525100 | 1.922577 |
| Fixed alpha=0.5 | 1.475573 | 1.669811 | 2.195688 |
| Fixed alpha=1.0 | 1.356035 | 1.686848 | 1.858678 |
| Learned alpha_t | 1.475573 | 1.708767 | 2.195688 |

| Schedule policy | Select decisions | alpha=0.0 | alpha=0.5 | alpha=1.0 |
|---|---:|---:|---:|---:|
| Fixed alpha=0.0 | 188 | 188/188 | 0/188 | 0/188 |
| Fixed alpha=0.5 | 193 | 0/193 | 193/193 | 0/193 |
| Fixed alpha=1.0 | 191 | 0/191 | 0/191 | 191/191 |
| Learned alpha_t | 193 | 0/193 | 146/193 | 47/193 |

The learned policy makes 193 Select decisions, with counts of `0/146/47` for `alpha=0/0.5/1`. Although `alpha=0.5` is the strongest fixed setting, learned `alpha_t` still reduces total end-to-end time by `0.69%` and achieves higher WS, GS, and Imp. The dynamic choices shown by Action Frequency therefore have performance significance: using `alpha=1` for selected states is better than fixing every state to `alpha=0.5`.

Per-query data is stored in `results/benchmark/nqo/nqo_alpha_sensitivity.csv`. The runner is `scripts/reproduce/nqo/run_alpha_sensitivity.py`; analysis code and logs are `results/benchmark/nqo/analyze_nqo_alpha_sensitivity.py` and `analyze_nqo_alpha_sensitivity.log`. Fixed settings produce 339 results: 261 versioned-cache hits and 78 cache misses that execute once and write back to the JOB light buffer. Every Select decision passes the fixed-alpha consistency check.
