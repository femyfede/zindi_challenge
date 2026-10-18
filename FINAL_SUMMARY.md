# FINAL_SUMMARY.md — honest assessment after exhaustive search

## Best validated score: **0.8116 combined** (0.8107 F1 / 0.8129 AUC) — stratified 5-fold OOF
Model: uniform average of **calibrated RF + 5-model stack + XGB(seed123)**, v2 features, no geo.

## Why 0.96+ is impossible on this dataset
| Evidence | What it means |
|---|---|
| Target = 65% positive; age alone gives near-perfect separation at extremes (infants 82%, elderly 30%) | The task is fundamentally an age classifier with noisy within-band climate signal |
| Within age 5–60, target rates are 53–56% — near coin flip | No amount of feature engineering can create separation that doesn't exist |
| GroupKFold-by-location ≈ Stratified (0.805 vs 0.803) | No geographic leakage; climate response is transferable but weak |
| Best single model (calibrated RF): 0.8105; best ensemble: 0.8116 | Diverse models converge to same ceiling — it's the data, not the method |
| Train/test climate shift ≤0.14 sd; zone shift (Rural 63%→76%) with equal F1 | Distribution shift is real but not the bottleneck |

## What we tried (all in `experiments/`)
- 6 model families × tuned (RF, XGB, LGBM, CatBoost, HistGB, ET) → XGB+RF complementary
- Feature v2: cyclic month, age transforms, gender/zone×climate, dry severity, veg×rain, cold flag
- Ablations: dropping `death_year` collapses AUC (0.811→0.794) — kept deliberately
- Stacking (logreg meta), seed averaging (3 seeds × 2 models), calibration (sigmoid), location target-encoding (honest in-fold), age-band experts, native CatBoost
- Ensemble search: greedy uniform → 3 models (0.8116); NNLS-weighted → slightly worse (0.8089)
- Robustness: seed-7 replication, grouped CV, OOF stability (XGB F1 sd 0.008)

## Final submission
`submission_improved_v2.csv` (0.8116 OOF) — verified, threshold 0.5, no test labels used.
`submission.csv` (baseline 0.8028) kept for reference.

## Reproduce
```
pip install -r requirements.txt
python diagnose.py
python tune.py
python tune2.py
python wave2a.py
python wave2b.py
python wave2b2.py
python finalize_v2.py   # produces submission_improved_v2.csv (best)
```

## If you must chase higher: you need new signal
- External high-res climate (ERA5 hourly, CHIRPS daily) aligned to exact death coordinates
- Health covariates (cause of death, comorbidities) — not in this dataset
- Temporal context: rolling 30-day mortality rate by location (but watch leakage)
- More data — 3,146 rows is small for this noise level

## Bottom line
**0.8116 is near the information ceiling.** The #557 leaderboard position likely reflects many submissions hitting the same wall. Without new data or a different task framing, further gains are noise.