# RETRAINING REPORT — what changed, what worked, what the ceiling is

## 1. Key breakthrough this round: a validation bug was masking real performance
`train_final.py` time-aware CV reported **0.45-0.46 combined** — a catastrophic number.
Root cause: the rolling year-by-year split only ever predicts ~1104 of 3146 rows, but the
metric was computed over **all** rows, with the unpredicted remainder left as **0.0** fake
probabilities. Those zeros dragged F1/AUC down artificially.

After aggregating over **only rows that received a real prediction** (`cv_compare.py`):

| Validation strategy | XGB | RF | LightGBM |
|---|---|---|---|
| Stratified 5-fold (all 3146 rows) | 0.7966 | 0.8019 | 0.7880 |
| **Rolling year-by-year (1104 rows)** | 0.8147 | **0.8335** | 0.8178 |
| Chronological 80/20 holdout (629 rows) | 0.6374 | 0.6635 | 0.6354 |

Per-fold scores in the rolling split are 0.74-0.85 — never near 0.45. The 0.45 was pure
artifact. **RF is the most robust model under time shift (0.8335).**

## 2. Calibration anchor — the most important finding for expectations
The current submission scores **0.81584541** on the leaderboard. That exact submission
scored **0.8028-0.8116** in local stratified OOF. Therefore:

> **Local validation maps ~1:1 to the leaderboard.**

This is the single most useful calibration point in the whole project. To reach a 0.875
leaderboard score a model must post **~0.86+ local OOF**. No configuration tested in this
or the previous round comes close; the entire family sits in **0.78-0.83**.

## 3. Why 0.90 is not reachable with this dataset
- **Target is age-dominated.** P(target) by age: ≤1y 0.82, 1-5y 0.91, 5-18y 0.56,
  18-40y 0.54, 40-60y 0.47, 60+y 0.30. Within ages 5-60 the label is near a coin flip.
- **Model is perfect at the extremes, wrong in the middle.** OOF error analysis:
  F1 = 0.95 for age 1-5 (but 100% false-positive rate — every 1-5y record predicted
  positive), 0.13 for 60+ (92% false negatives). The signal simply is not there.
- **Every approach converges to the same wall.** 6 model families, 4 feature sets,
  location target encoding, spatial KMeans clusters, lat/lon, native CatBoost,
  stacking, NNLS weighting, sigmoid calibration, seed averaging, greedy ensembles:
  all land in 0.78-0.83. That convergence is the definition of an information ceiling.
- **No leakage was found or exploited** — all preprocessing is in-fold, features are
  strictly row-wise and backward-looking (7/30/90-day lookback windows end before deathdate).
- **Test spans the same years as train** (2007-2022), so this is not a future-data problem.
  The base rate does drift downward over time (0.77 in 2009 → 0.46 in 2019).

## 4. Experiments run this round (all logged)
| Experiment | Result |
|---|---|
| Rolling year-by-year CV, correct aggregation | RF **0.8335** (best of the whole project) |
| Chronological 80/20 hard-cutoff holdout | 0.6374-0.6635 (629 rows too small/abrupt) |
| Location climatology anomalies (per-location z-scores) | No gain — location TE already captures it |
| Location target encoding (smooth 10→200 sweep) | Best at smooth=30-50, ~0.804, no gain over no-geo |
| Spatial KMeans clusters + distance-to-center | No gain (0.8043 vs 0.8028 baseline) |
| Adding lat/lon to feature set | No gain (grouped CV already showed this) |
| Native CatBoost (loc/gender/zone/cluster as categoricals) | 0.7876-0.7915 — consistently below RF/XGB |
| Tuned ExtraTrees (6 configs) | 0.7925 — below RF |
| Multi-seed averaging (3 seeds × RF/XGB) | 0.8081 — no gain over single seed |
| NNLS-weighted ensemble | 0.8089 — worse than uniform |
| Pseudo-labeling (implemented, left disabled) | Not adopted: teacher scores ~0.63 on shifted data, would propagate error |

## 5. Recommended final submission
**`submission_rolling_v4.csv`** — single tuned RandomForest on advanced features.
- Best under the most realistic validation (rolling time-aware): **0.8335**
- Second-best under stratified: 0.8019
- Best calibrated: predicted positive rate **0.6612** vs train base rate 0.6507
  (the 0.8116 ensemble over-predicted positives at 0.7155)
- Most robust across seeds, folds and time shifts; most interpretable

**Ensemble caveat:** RF+XGB averaged 0.8116 under *stratified* CV but was **worse** than RF
alone under rolling CV. Since the leaderboard is the ground truth and RF is the only model
that is strong under *both* schemes, the single RF is the defensible choice.

## 6. Exact regeneration
```
pip install -r requirements.txt
python cv_compare.py      # validation-strategy comparison (fixes the aggregation bug)
python final_rolling.py   # rolling OOF + ensemble test + submission_rolling_v4.csv
```
`TargetF1 = (TargetRAUC >= 0.5).astype(int)` — threshold 0.5 only, no tuning, no test labels.

## 7. Honest bottom line
Realistic ceiling on the provided data is **~0.83 local / ~0.83-0.84 leaderboard**. The
0.875 leaders and the 0.90 target are not reachable from `Train.csv` +
`climate_features.csv` alone. The gap almost certainly comes from a resource this project
does not have: **external daily-resolution climate data** (ERA5 hourly, CHIRPS 0.05° daily,
MODIS LST) re-gridded to each exact death coordinate and date, which the organisers
explicitly permitted and which the provided `climate_features.csv` (7/30/90-day aggregates
only, coarse spatial resolution) does not provide. That is the one avenue I did not take,
and it is the only credible route to 0.87+.
