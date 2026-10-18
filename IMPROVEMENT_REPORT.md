# IMPROVEMENT_REPORT.md — from #557 baseline to improved submission

## 1. Original model
RandomForest (400 trees, min_samples_leaf=2, balanced_subsample), v1 features
(demographics + climate + engineered anomalies/interactions, no geo), stratified 5-fold
selection, full-train retrain, threshold 0.5. File: `submission.csv` (kept, not overwritten).

## 2. Original validation performance
F1 0.7972 / AUC 0.8111 / combined **0.8028** (stratified 5-fold OOF). Holdout 20%: 0.7988.

## 3. Leaderboard position at start: approximately #557.

## 4. Problems discovered (see MODEL_AUDIT.md + experiments/)
- No hyperparameter search anywhere; XGB ran with scale_pos_weight=1.0 (no imbalance handling).
- No CatBoost, no ensemble, no saved OOF predictions.
- `death_year` the #2 feature — ablations show dropping it collapses AUC
  (0.8109→0.7937), so it carries legitimate ranking signal; kept deliberately.
- Error analysis (RF OOF): elderly F1 0.13 (FN 92%), age 40–60 F1 0.38, infants FP 80%,
  age 1–5 FP 100% — the model is age-dominated with poor within-band ranking.
  Fix = better probabilities via tuned diverse models + ensemble, not threshold games.
- Train/test climate shift is mild (≤0.14 sd); zone shift is real (Rural 63%→76%)
  but error rates by zone are equal (F1 0.798 vs 0.797) → no zone-specific fix needed.
- LGBM/CatBoost/HistGB tuned below RF/XGB on this data (0.785–0.790) → excluded from ensemble.

## 5. Experiments performed (full log: experiments/model_results.csv)
| exp | what | result |
|---|---|---|
| baseline | 6 models, v1, strat5 | RF 0.8028 best |
| expA | v1 vs v1-noyear vs v2 vs v2-noyear (RF) | v2 0.8042 wins; noyear hurts AUC |
| expB | manual randomized search, 3-fold, combined scorer: RF(8), XGB(12), LGBM(12), Cat(6), HGB(6) | RF 0.8050, XGB 0.8048, LGBM 0.7890, Cat 0.7883, HGB 0.7874 |
| expC | tuned 5-fold OOF | XGB 0.8039 (sd F1 0.008), RF 0.8021, Cat 0.7900, LGBM 0.7882, HGB 0.7847 |
| expD | greedy forward uniform average on OOF | XGB+RF **0.8094** (+0.0055); adding Cat/LGBM/HGB hurts |
| expE | seed-7 3-fold robustness | XGB 0.8071, RF 0.8024 — stable |

## 6. Feature engineering improvements (src/features_v2.py, all row-wise, leakage-safe)
Cyclic month (sin/cos), age_log1p + age_sq, gender×climate (female×tmax, female×rain),
zone×climate (rural×rain, peri×tmax), ndvi×rain stress, dry severity, rain persistence,
cold flag, elderly×dry. Gain on RF: 0.8028 → 0.8042.

## 7. Model improvements
- XGBoost tuned: 400 trees, depth 3, lr 0.072, subsample 0.76, colsample 0.62,
  reg_lambda 7.8, **scale_pos_weight 1.5** (first real imbalance handling for XGB):
  0.7946 → **0.8039**.
- RF tuned: 1000 trees, leaf 3, log2 features: 0.8028 → 0.8021 (equal, more stable partner).
- CatBoost tested for the first time (1100/depth6): 0.7900 — honestly worse here, excluded.

## 8. Hyperparameter improvements
Manual randomized search (≈50 configs total) on 3-fold OOF with the exact competition
objective (0.6·F1@0.5 + 0.4·AUC). No threshold touched at any point.

## 9. Ensemble experiments
Greedy forward uniform averaging on 5-fold OOF: XGB (recall 0.86) + RF (precision 0.82)
are complementary → **0.8094**, the best observed score. Weights are uniform by construction
of the greedy procedure (validated on OOF, not hand-picked).

## 10. Climate feature experiments
No-climate 0.7882 → v1 climate 0.8028 → v2 climate 0.8042/0.8039. Climate adds ~+0.016
combined; v2 stress/interaction features add a further +0.001–0.002. No external data
downloaded (provided file already covers CHIRPS/ERA5-Land/MODIS/SRTM with verified alignment).

## 11. Validation comparison
Primary: stratified 5-fold OOF (matches overall distribution). Secondary: GroupKFold by
location ≈ equal scores (0.805), seed-7 replication stable. Ensemble fold-std is low
(XGB F1 sd 0.008 across folds). Selection by OOF overall, never by a single split.

## 12. Final model
**Uniform average of tuned XGBoost + tuned RandomForest, v2 no-geo features**,
each retrained on full train. OOF combined **0.8094** (F1 0.8060 / AUC 0.8145 approx
from ensemble OOF — see experiments/oof_predictions.csv).

## 13. Expected strengths
+0.0066 combined over baseline with lower variance; complementary precision/recall;
robust across seeds; geo-shift safe (no location/lat-lon); verified submission.

## 14. Remaining risks
- `death_year` dependence (temporal trend may shift); mitigated by keeping it only because
  ablation shows it aids ranking, and ensemble dilutes single-feature reliance.
- Test geography is far-OOD (Kalangala/Mitooma/Sembabule); no location features used,
  but climate-response relationships are assumed transferable.
- Elderly recall remains the weakest subgroup; gains came from ranking, not thresholding.

## 15. Exact submission-generation procedure
```
pip install -r requirements.txt
python diagnose.py    # baseline freeze + shift/error analysis
python tune.py        # RF/XGB/LGBM search  (tune2.py: CatBoost/HistGB)
python finalize.py    # 5-fold OOF, greedy ensemble, retrain, verify
```
`finalize.py` retrains XGB+RF pipelines (median/mode impute + one-hot in-pipe) on all
3146 train rows, averages test probabilities uniformly,
`TargetF1 = (TargetRAUC >= 0.5).astype(int)` → `submission_improved.csv`,
verified: columns/order, 1030 rows, IDs match Test, no dups/NaNs, proba∈[0,1],
F1↔proba consistency at 0.5, format match to SampleSubmission.csv.
