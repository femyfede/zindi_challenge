# Climate-Health Mortality Prediction — reproducible solution
Binary classification: predict `is_climate_sensitive` (1 = climate-sensitive death).
Final score = 0.60*F1 + 0.40*ROC-AUC. Binary predictions use ONLY threshold 0.5.
## Data
- `Train.csv` (3146 rows), `Test.csv` (1030 rows), `climate_features.csv` (4176 rows, CHIRPS/ERA5-Land/MODIS/SRTM, joined on ID).
- Target positive rate 65.1%. No missing values, no duplicate IDs, no train/test ID overlap.
- KEY FINDING: only 1 of 11 test locations appears in train; test has far-OOD sites -> raw `location` excluded; lat/lon/elev/slope decided by grouped CV (see report).
## Pipeline
1. Merge climate on ID (drop duplicate deathdate + constant hot_days_30d).
2. Row-wise feature engineering only (temporal, temp/rain/veg indicators, age groups, vulnerability x exposure interactions). See `src/feature_engineering.py`.
3. Preprocess inside CV folds: median/mode imputation + one-hot (scale only for logreg).
4. Validation: stratified 5-fold (primary) + GroupKFold by location (spatial generalisation). Metric 0.6*F1+0.4*AUC at 0.5.
5. Models: logreg / rf / extratrees / histgb / xgboost / lightgbm (class_weight balanced). See `outputs/reports/model_comparison.csv`.
6. Final: retrain selected model on all train, predict test probabilities, TargetF1=(proba>=0.5).
## Reproduce
```
pip install -r requirements.txt
python run_solution.py
```
Outputs: `submission.csv`, `outputs/reports/`, `outputs/figures/`, `outputs/models/final_pipeline.pkl`.
## Submission
Columns ID,TargetF1,TargetRAUC; verified: row count, ID match, 0/1, proba in [0,1], F1<->proba consistency at 0.5.
## Notes
- No external data downloaded (provided climate file already covers ERA5/CHIRPS/MODIS/SRTM with verified alignment).
- Language is associational, not causal.
