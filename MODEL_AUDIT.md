# MODEL_AUDIT.md — audit of the existing (#557) solution before optimization

## 1. Current approach
- Pipeline (`run_solution.py` + `src/`): merge `climate_features.csv` onto Train/Test via `ID`
  (100% coverage, deathdates consistent); drop constant `hot_days_30d`.
- Row-wise feature engineering only (leakage-safe): death year/month, wet-season flags,
  `temp_range`, short/medium temperature anomalies (`t_anom_7_30`, `t_anom_30_90`), peak heat
  (`heat_30d`), rain intensity/share/pace/dry/heavy-storm flags, NDVI change + low-veg flag,
  age groups + under5/elderly/infant flags, vulnerability×exposure interactions
  (`age_x_tmax`, `under5_x_rain30`, `elderly_x_tmax`, `precip_x_temp`).
- Raw `location` string excluded (39 train cats, only 1 overlaps test). lat/lon/elevation/slope
  excluded after grouped-CV ablation showed no gain (`use_geo=False`).
- Preprocessing inside each CV fold: median/mode imputation + one-hot (scale only for logreg).
- Validation: stratified 5-fold primary + GroupKFold-by-location secondary. Fixed seed 42.
- Models (all `class_weight` balanced): LogReg / RF / ExtraTrees / HistGB / XGBoost / LightGBM.
- Selection: best stratified combined score → **RandomForest**
  (400 trees, min_samples_leaf=2, balanced_subsample). Retrained on full train.
- Threshold: binary predictions strictly at proba ≥ 0.5. No tuning. Verified submission.

## 2. Current validation results (stratified 5-fold OOF, no-geo)
| model | F1 | AUC | combined (0.6F1+0.4AUC) | prec | rec |
|---|---|---|---|---|---|
| RF (final) | 0.7972 | 0.8111 | **0.8028** | 0.804 | 0.790 |
| ExtraTrees | 0.7893 | 0.8048 | 0.7955 | 0.813 | 0.767 |
| XGBoost | 0.7920 | 0.7986 | 0.7946 | 0.783 | 0.802 |
| LightGBM | 0.7905 | 0.7970 | 0.7931 | 0.803 | 0.779 |
| HistGB | 0.7806 | 0.7961 | 0.7868 | 0.801 | 0.761 |
| LogReg | 0.7680 | 0.8036 | 0.7822 | 0.832 | 0.713 |
- Holdout 20%: F1 0.7903, AUC 0.8114, combined 0.7988, cm [[131,89],[84,326]].
- Climate ablation: with-climate 0.8028 vs no-climate 0.7882 (+0.0146 combined) → real signal.
- Geo ablation: stratified geo 0.8016 ≈ no-geo 0.8028; grouped no-geo 0.8052 ≈ geo 0.8050.

## 3. Current strengths
- Correct schema/merge; leakage-safe row-wise features; preprocessing inside folds; fixed seeds.
- Validation matches the competition metric (0.6/0.4) with threshold fixed at 0.5.
- Geographic-shift aware: excluded raw location, tested geo via grouped CV.
- Climate value proven by ablation; submission fully verified.

## 4. Current weaknesses (why ~#557)
1. **No hyperparameter optimization** — every model uses a single hand-picked config
   (e.g. RF 400 trees / leaf 2; XGB depth 6 / lr 0.04; LGBM 800 / leaves 63). Untuned.
2. **No CatBoost** — never tested despite categorical features (gender/zone/age_group).
3. **Single model, no ensembling** — RF wins by ~0.008 but XGB/LGBM/ET are diverse;
   OOF blending/stacking unexplored.
4. **No OOF artifacts** — no saved OOF predictions, so ensemble weights/error analysis impossible.
5. **Weak class-imbalance treatment** — `balanced_subsample` only; scale_pos_weight=1.0 for XGB
   (i.e. NO imbalance handling); no comparison of weighting schemes; F1 (60% weight) is
   recall-sensitive and the winner-takes-all choice was never validated for F1/AUC tradeoff.
6. **Suspicious `death_year` reliance** (importance #2, 0.076) — likely a temporal trend that may
   not transfer; never ablated.
7. **Thin interaction set** — no gender×climate, zone×climate, cyclic month, age transforms,
   or target-independent frequency/spatial encodings.
8. **No error analysis** — unknown where FN/FP concentrate (age group? zone? dry periods?).
9. **No robustness testing** — single seed, single split scheme; stability unknown.
10. **Calibration unexamined** — AUC depends on probability ranking quality.

## 5. Potential leakage — checked, none found
- Climate windows are strictly backward-looking (7/30/90d before deathdate). No ID overlap
  train/test. No duplicate rows. Preprocessing in-fold. No test use in training. `death_year`
  is legitimate (known at prediction time) but risky for transfer — flagged, not leakage.

## 6. Potential underfitting
- RF leaf=2/400 trees is low-capacity vs 41 features; boosting models likely under-tuned
  (default depths/rates, no early stopping); interactions underexplored.

## 7. Potential overfitting
- `death_year` + `age_x_tmax` may memorize temporal/demographic quirks; single-split selection
  on stratified CV while leaderboard is geographically shifted (grouped CV ≈ 0.805 suggests
  OK, but the final pick used stratified ranking only).

## 8. Feature limitations
- Missing: cyclic month encoding, age nonlinearities, gender/zone×climate, dry-spell severity,
  NDVI×rain stress combos, location-frequency (in-fold) encoding, cluster features.

## 9. Modeling limitations
- One config per family; no search; no CatBoost; no ensemble; no calibration; XGB without
  imbalance handling; seed/fold stability unknown.

## 10. Improvement plan
1. Freeze baseline into `experiments/baseline_results.csv` (+ OOF dumps).
2. Diagnose shift (train vs test) + OOF error analysis by age/gender/zone/climate.
3. Feature set v2: cyclic month, age transforms, gender/zone×climate, extra stress indices;
   ablate `death_year` and climate groups.
4. RandomizedSearchCV (combined scorer, threshold 0.5) for RF/XGB/LGBM/CatBoost/HistGB.
5. OOF-weighted ensemble + stacking; seed/fold robustness checks.
6. Retrain best on full train → `submission_improved.csv` (keep old `submission.csv`);
   write `IMPROVEMENT_REPORT.md` + `FINAL_MODEL_SUMMARY.md`.
