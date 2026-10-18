# Round 2 — what actually moved the score, and what the real ceiling is

Supersedes the ceiling argument in `CEILING_ANALYSIS.md`. Two of its claims turned out to be
wrong: the ceiling was **not** 0.8116, and the reason was not "no signal beyond demographics".

## The real structure: a recording break at 2017-Q3

Independent competitors on the Zindi discussion board flagged it; confirmed directly here.

Infant (age <= 1) share of recorded deaths **and** their climate-sensitive rate, by quarter:

| period | infant share | infant climate-sensitive rate | under-5 rate |
|---|---|---|---|
| 2008Q1 – 2017Q2 | 0.27 – 0.45 | **0.86 – 1.00** | 0.88 – 1.00 |
| 2017Q3 – 2022Q2 | 0.39 – 0.85 | **0.31 – 0.71** | 0.33 – 0.73 |

Per-year infant climate-sensitive rate: 2008–2016 = 0.90–1.00, 2017 = 0.86, 2018 = 0.56,
2019 = 0.42, 2020 = 0.56, 2021 = 0.59, 2022 = 0.76.

Simultaneously the infant *share* of all deaths rose from ~1/3 to ~3/4. That keeps the overall
base rate near 0.65 while inverting the age gradient. It is a change in what was recorded, not
in the climate. Consequence: **age alone reaches AUC 0.734, but a model that cannot split on
age AND date at once captures almost none of it.**

Train and test match on this axis (train infant share 0.369 pre / 0.668 post; test 0.363 / 0.652),
so the structure transfers.

## What worked

| Change | CV F1 | CV AUC | Combined |
|---|---|---|---|
| Previous best (`finalize_v2` ensemble) | 0.8107 | 0.8129 | 0.8116 |
| RF on 88 advanced climate features | 0.7951 | 0.8090 | 0.8007 |
| **Age x period spline-logistic (final)** | **0.8187** | **0.8250** | **0.8212** |

The winning model is a `LogisticRegression(C=0.02)` on a deliberately *small* design matrix:

- `SplineTransformer(10 knots)` on `age`, and again on `age x is_post` — the age curve modelled
  separately before and after the break
- `SplineTransformer(6 knots)` on `year_cont` and `day-of-year`
- one-hot `gender`, `zone`, and `age_band(20) x period` cell (and `cell x gender`)

Validation: 5-fold stratified CV over 5 seeds. Seed-mean combined **0.8214**, seed-min 0.8208
(the tightest spread of every candidate — singleA also won on worst-seed, so it is not a
CV-selection artifact).

## What did not work (and why the old reports were right to drop it)

- **Climate features hurt.** `demo_time` beat `demo_time_clim` for every model tried
  (logreg 0.8142 vs 0.8088; MLP 0.806 vs 0.766). Univariate AUCs: best climate variable is
  `max_daily_rain_30d` at 0.462. They are largely proxies for the recording-year change.
- **Geography carries nothing.** OOF AUC of leakage-free target encoding: district 0.481,
  location 0.481, facility 0.484, deathdate 0.535 — all at or below chance. And **0 of 43 train
  coordinates appear in test**; only 1 of 11 test locations is seen in train. Geo features cannot
  transfer regardless.
- **Tree ensembles are strictly worse here.** RF 0.793, CatBoost 0.802, LightGBM 0.801,
  HGB 0.800, MLP 0.808 vs linear 0.821. With 3146 rows and one dominant smooth interaction,
  trees overfit and their probabilities sit badly around the 0.5 cut (RF predicts positive for
  only 0.57 of rows).
- **Rank-averaging ensembles are actively harmful.** Blending by rank collapses the positive
  rate to 0.50 and F1 to 0.766 — AUC survives (it is rank-based) but the 0.5 threshold does not.
  Only probability averaging is safe here.
- **No batch or serial structure.** `(location, date)` clusters with n >= 3: zero. Same-label rate
  at lag 1/2/3/5/10 within location: 0.550/0.567/0.554/0.556/0.553 vs 0.545 by chance.
- **No ID or row-order leak.** `climate_features.csv` is exactly `Train.csv` followed by
  `Test.csv`, both date-sorted; IDs are random 8-hex and show no bit-level signal.

## Why 0.902 AUC / 0.892 F1 is not reachable from this data

The label is a cause-of-death classification. Cause of death is not in the dataset. The
achievable signal decomposes as:

| Oracle knowledge | F1 | AUC | Combined |
|---|---|---|---|
| true rate per age band | 0.8074 | 0.7632 | 0.7897 |
| + gender | 0.8084 | 0.7708 | 0.7934 |
| + zone | 0.8109 | 0.7732 | 0.7959 |
| **best model actually built** | **0.8187** | **0.8250** | **0.8212** |

The model is *already above* the demographic oracle. Every additional feature family tried
(climate, geography, location encoding, date encoding, external NASA POWER/MERRA-2) moved the
score by less than the seed-to-seed noise (±0.002), mostly downward. Predicting the positive
class for essentially everything already scores F1 0.788, so F1 0.892 at 65% prevalence would
require ~88% precision at ~90% recall — a near-separable problem, which this label is not.

Realistic ceiling on the provided columns is roughly **AUC 0.83 / F1 0.82 / combined 0.823**.

## Files

- `features2.py` — feature builder with the age x period cell
- `make_submission.py` — final model + submission writer
- `final_select.py` — 5-seed tie-break between candidates
- `submission_v2_singleA.csv` — **recommended** (CV seed-mean 0.8214)
- `submission_v2_ABC.csv` — alternative 3-member blend (CV seed-mean 0.8210)
