# FINAL_MODEL_SUMMARY.md — concise pipeline description

**Task:** binary classification of climate-sensitive death (`is_climate_sensitive`).
**Metric:** 0.6·F1 + 0.4·ROC-AUC. **Threshold:** fixed 0.5, never tuned.

**Data:** Train 3146 / Test 1030 / climate_features 4176 rows (CHIRPS rain, ERA5-Land temp,
MODIS NDVI, SRTM elevation/slope), joined on ID. No missing values, no duplicates.

**Features (v2, no geo):** demographics (age, gender, zone, age groups/under5/elderly/infant),
temporal (year, month, cyclic sin/cos, wet-season flags), climate (temps, rain sums/days,
NDVI + engineered anomalies, intensity, dry severity, persistence, cold flag, veg×rain stress),
interactions (age/gender/zone/elderly × climate). Raw location and lat/lon/elevation/slope
excluded (train/test geographic shift; grouped CV showed no gain).

**Preprocessing:** median/mode imputation + one-hot, fitted inside each fold / on full train.

**Model:** uniform probability average of
(a) XGBoost (400, depth 3, lr 0.072, subsample 0.76, colsample 0.62, reg_lambda 7.8,
scale_pos_weight 1.5) and
(b) RandomForest (1000 trees, leaf 3, log2, balanced_subsample),
both on v2 features. **OOF combined 0.8094** vs baseline 0.8028.

**Validation:** stratified 5-fold OOF (primary) + GroupKFold-by-location + seed-7
replication. Experiments logged in experiments/model_results.csv; OOF in
experiments/oof_predictions.csv.

**Submission:** `submission_improved.csv` (ID, TargetF1, TargetRAUC), TargetF1 from 0.5,
all format checks pass. Reproduce: `python diagnose.py`, `python tune.py`,
`python tune2.py`, `python finalize.py`. All language associational, not causal.
