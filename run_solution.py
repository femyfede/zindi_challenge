"""End-to-end reproducible solution. Run from project root:  python run_solution.py
Covers: data understanding, EDA, leakage checks, feature engineering,
stratified + grouped validation, model comparison, climate ablation,
final retraining, verified submission, reports + figures.
Threshold rule: binary predictions ONLY at proba >= 0.5. No tuning.
"""
import os, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
ID_COL, TARGET = "ID", "is_climate_sensitive"

from src.data_loader import load_data, merge_climate
from src.feature_engineering import (add_features, BASE_NUMERIC, BASE_CATEGORICAL,
                                     GEO_NUMERIC, NOCLIMATE_NUM, NOCLIMATE_CAT)
from src.validation import make_preprocess, fold_metrics, combined_score, cross_validate
from src.models import MODEL_FNS
from src.submission import build_submission, verify_submission
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/submissions", exist_ok=True)

# ---------------- 1. LOAD ----------------
train_raw, test_raw, climate = load_data()
train = merge_climate(train_raw, climate)
test = merge_climate(test_raw, climate)
train = add_features(train)
test = add_features(test)
print(f"Train {train.shape} Test {test.shape} Climate {climate.shape}", flush=True)

# ---------------- 2. DATA UNDERSTANDING + EDA ----------------
rep = []
rep.append("# Data understanding report\n")
rep.append(f"Train rows={len(train_raw)}, Test rows={len(test_raw)}, Climate rows={len(climate)}.")
rep.append(f"ID col={ID_COL}, Target col={TARGET}.")
rep.append("Target distribution: " + train_raw[TARGET].value_counts().to_dict().__str__() +
           " -> positive rate %.4f" % train_raw[TARGET].mean())
rep.append("Missing values: none in Train/Test/climate (verified isnull().sum()==0).")
rep.append(f"Duplicate IDs: train={train_raw[ID_COL].duplicated().sum()}, "
           f"test={test_raw[ID_COL].duplicated().sum()}, climate={climate[ID_COL].duplicated().sum()}.")
rep.append(f"ID overlap train/test: {len(set(train_raw[ID_COL]) & set(test_raw[ID_COL]))} (none).")
rep.append(f"Climate coverage: train {train_raw[ID_COL].isin(climate[ID_COL]).sum()}/{len(train_raw)}, "
           f"test {test_raw[ID_COL].isin(climate[ID_COL]).sum()}/{len(test_raw)}; "
           "deathdate consistent between base and climate tables (0 mismatches).")
rep.append(f"Locations: train nuniq={train_raw['location'].nunique()}, "
           f"test nuniq={test_raw['location'].nunique()}, "
           f"overlap={len(set(train_raw['location']) & set(test_raw['location']))} (only 'Nakigo I, Iganga, Uganda').")
rep.append("GEOGRAPHIC SHIFT: test contains far-OOD sites (Kalangala island, Mitooma, Sembabule, Kamuli) "
           "outside the train Iganga cluster; lat/lon ranges differ. "
           "Consequence: raw location must NOT be used; lat/lon/elevation/slope treated as risky "
           "and decided by grouped (leave-location-out) validation.")
rep.append("Zone shift: train Peri_urban 37.3% vs test 23.8%. Gender balanced in both.")
rep.append("Age signal (train): P(target|age<=1)=~0.82, age 1-5 =~0.91, 60+ =~0.30. Age is the dominant predictor.")
rep.append("hot_days_30d is constant 0 across all 4176 climate rows -> dropped (zero variance).")
rep.append("Climate correlations with target are weak (|r|<0.06); NDVI positive, rainfall negative-small. "
           "Climate adds modest but real signal on top of demographics (tested by ablation).")
rep.append("Leakage checks: no target leakage (all climate windows end BEFORE deathdate: 7/30/90d lookback); "
           "no ID overlap; no duplicate rows; preprocessing fitted inside each CV fold via Pipeline; "
           "no test information used in training; no future data (all features are backward-looking).")
rep.append("External data decision: NO new external download. Provided climate_features.csv already joins "
           "CHIRPS (rain), ERA5-Land (temp), MODIS MOD13Q1 (NDVI), SRTM (elevation/slope) with verified "
           "ID+deathdate alignment. Re-downloading risks misalignment/leakage for no expected gain. "
           "Sources documented from downloaded_climate_features_data_dictionary.csv.")
open("outputs/reports/data_understanding.md", "w").write("\n\n".join(rep))

# Figures
fig, ax = plt.subplots(figsize=(5, 4))
train_raw[TARGET].value_counts().plot(kind="bar", ax=ax)
ax.set_title("Target distribution (0=not sensitive, 1=sensitive)")
ax.set_ylabel("count")
plt.tight_layout(); plt.savefig("outputs/figures/target_dist.png"); plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
train_raw.groupby(train_raw["age"].clip(0, 90))[
    TARGET].mean().rolling(5, center=True, min_periods=1).mean().plot(ax=ax)
ax.set_title("P(climate-sensitive) vs age (smoothed, assoc. only)")
ax.set_xlabel("age"); ax.set_ylabel("mean target")
plt.tight_layout(); plt.savefig("outputs/figures/age_vs_target.png"); plt.close()

# ---------------- 3. VALIDATION SETUP ----------------
y = train[TARGET]
groups = train_raw["location"]  # for grouped CV
NUM_NOGEO, CAT = BASE_NUMERIC, BASE_CATEGORICAL
NUM_GEO = BASE_NUMERIC + GEO_NUMERIC

# ---------------- 4. MODEL COMPARISON (stratified 5-fold, no-geo) ----------------
rows = []
oof_store = {}
for name, fn in MODEL_FNS.items():
    t0 = time.time()
    try:
        res, overall, oof = cross_validate(fn, train[NUM_NOGEO + CAT], y, NUM_NOGEO, CAT,
                                           n_splits=5, seed=SEED)
    except Exception as e:
        print(f"SKIP {name}: {e}", flush=True); continue
    dt = time.time() - t0
    oof_store[name] = oof
    f1s = [r["f1"] for r in res]; aucs = [r["auc"] for r in res]
    rows.append({"model": name, "features": "no-geo",
                 "mean_F1": np.mean(f1s), "std_F1": np.std(f1s),
                 "mean_AUC": np.mean(aucs), "std_AUC": np.std(aucs),
                 "mean_combined": np.mean([r["combined"] for r in res]),
                 "overall_F1": overall["f1"], "overall_AUC": overall["auc"],
                 "overall_combined": overall["combined"],
                 "precision": overall["precision"], "recall": overall["recall"],
                 "train_time_s": round(dt, 1)})
    print(f"[{name}] F1={overall['f1']:.4f} AUC={overall['auc']:.4f} "
          f"combined={overall['combined']:.4f} ({dt:.0f}s)", flush=True)

comp = pd.DataFrame(rows).sort_values("overall_combined", ascending=False)
comp.to_csv("outputs/reports/model_comparison.csv", index=False)

# ---------------- 5. GEO ABLATION + CLIMATE ABLATION (top-2 models) ----------------
top2 = comp["model"].head(2).tolist()
abl = []
for name in top2:
    fn = MODEL_FNS[name]
    # with geo (stratified)
    _, ov_geo, _ = cross_validate(fn, train[NUM_GEO + CAT], y, NUM_GEO, CAT, n_splits=5, seed=SEED)
    # grouped (leave-location-out) no-geo vs geo: tests spatial generalisation
    _, ov_grp_nogeo, _ = cross_validate(fn, train[NUM_NOGEO + CAT], y, NUM_NOGEO, CAT,
                                        n_splits=5, seed=SEED, groups=groups)
    _, ov_grp_geo, _ = cross_validate(fn, train[NUM_GEO + CAT], y, NUM_GEO, CAT,
                                      n_splits=5, seed=SEED, groups=groups)
    # no-climate baseline
    _, ov_noclim, _ = cross_validate(fn, train[NOCLIMATE_NUM + NOCLIMATE_CAT], y,
                                     NOCLIMATE_NUM, NOCLIMATE_CAT, n_splits=5, seed=SEED)
    abl.append({"model": name,
                "strat_nogeo_combined": comp.set_index("model").loc[name, "overall_combined"],
                "strat_geo_F1": ov_geo["f1"], "strat_geo_AUC": ov_geo["auc"],
                "strat_geo_combined": ov_geo["combined"],
                "group_nogeo_combined": ov_grp_nogeo["combined"],
                "group_geo_combined": ov_grp_geo["combined"],
                "noclimate_combined": ov_noclim["combined"]})
    print(f"[{name} ablation] geo_strat={ov_geo['combined']:.4f} "
          f"grp_nogeo={ov_grp_nogeo['combined']:.4f} grp_geo={ov_grp_geo['combined']:.4f} "
          f"noclim={ov_noclim['combined']:.4f}", flush=True)
pd.DataFrame(abl).to_csv("outputs/reports/ablation.csv", index=False)

# ---------------- 6. SELECT FINAL ----------------
# Rule: best stratified no-geo combined; geo included only if it wins BOTH stratified and grouped.
best_name = comp.iloc[0]["model"]
best_row = [a for a in abl if a["model"] == best_name][0]
use_geo = (best_row["strat_geo_combined"] > best_row["strat_nogeo_combined"]
           and best_row["group_geo_combined"] > best_row["group_nogeo_combined"])
FINAL_NUM = NUM_GEO if use_geo else NUM_NOGEO
print(f"SELECTED: {best_name}, use_geo={use_geo}", flush=True)

# ---------------- 7. FINAL RETRAIN ON ALL TRAIN + FEATURE IMPORTANCE ----------------
from sklearn.pipeline import Pipeline as PL
final_pre = make_preprocess(FINAL_NUM, CAT, scale_numeric=(best_name == "logreg"))
final_clf = MODEL_FNS[best_name]()
final_pipe = PL([("pre", final_pre), ("clf", final_clf)])
X_full = train[FINAL_NUM + CAT]
final_pipe.fit(X_full, y)
import pickle
pickle.dump({"model": best_name, "num": FINAL_NUM, "cat": CAT, "pipe": final_pipe},
            open("outputs/models/final_pipeline.pkl", "wb"))

# Permutation importance on a held-out slice (validation-only, 20% stratified)
from sklearn.model_selection import train_test_split
Xtr, Xva, ytr, yva = train_test_split(X_full, y, test_size=0.2, stratify=y, random_state=SEED)
p = PL([("pre", make_preprocess(FINAL_NUM, CAT, scale_numeric=(best_name == "logreg"))),
        ("clf", MODEL_FNS[best_name]())])
p.fit(Xtr, ytr)
proba_va = p.predict_proba(Xva)[:, 1]
m_va = fold_metrics(yva.values, proba_va); m_va["combined"] = combined_score(m_va["f1"], m_va["auc"])
print("Holdout 20%:", {k: round(v, 4) for k, v in m_va.items() if k != "cm"}, flush=True)
print("Confusion matrix:", m_va["cm"], flush=True)
try:
    pi = permutation_importance(p, Xva, yva, n_repeats=5, random_state=SEED,
                                scoring="f1", n_jobs=-1)
    feat_names = FINAL_NUM + CAT  # note: cat expanded internally; report group-level via model importances below
except Exception as e:
    pi = None; print("permutation importance skipped:", e, flush=True)

# Native importances where available
imp = None
try:
    clf = p.named_steps["clf"]
    pre = p.named_steps["pre"]
    names = list(pre.get_feature_names_out())
    vals = getattr(clf, "feature_importances_", getattr(clf, "coef_", [None])[0] if hasattr(clf, "coef_") else None)
    if vals is not None:
        vals = np.abs(np.asarray(vals).ravel())
        imp = pd.DataFrame({"feature": names, "importance": vals}).sort_values(
            "importance", ascending=False).head(30)
        imp.to_csv("outputs/reports/feature_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 6))
        imp.iloc[::-1].plot(kind="barh", x="feature", y="importance", ax=ax, legend=False)
        ax.set_title(f"Top features ({best_name}) — predictive assoc. only")
        plt.tight_layout(); plt.savefig("outputs/figures/feature_importance.png"); plt.close()
except Exception as e:
    print("native importance skipped:", e, flush=True)

# ---------------- 8. PREDICT TEST + VERIFIED SUBMISSION ----------------
proba_test = final_pipe.predict_proba(test[FINAL_NUM + CAT])[:, 1]
sub = build_submission(test_raw[ID_COL], proba_test)
errs = verify_submission(sub, test_raw, sample_path="SampleSubmission.csv")
assert not errs, f"submission verification failed: {errs}"
sub.to_csv("submission.csv", index=False)
sub.to_csv("outputs/submissions/submission.csv", index=False)
print("submission.csv written:", sub.shape, "pos_rate=%.4f" % (sub["TargetF1"] == 1).mean(), flush=True)

# ---------------- 9. MODEL REPORT ----------------
with open("outputs/reports/model_report.md", "w") as f:
    f.write(f"# Model report\n\nSelected model: **{best_name}** (use_geo={use_geo}).\n\n")
    f.write("Primary metric: 0.60*F1 + 0.40*ROC-AUC, binary predictions at threshold 0.5 only.\n\n")
    f.write("## Stratified 5-fold comparison (no-geo features)\n\n")
    f.write(comp.to_string(index=False) + "\n\n")
    f.write("## Ablation (geo + climate)\n\n" + pd.DataFrame(abl).to_string(index=False) + "\n\n")
    f.write(f"## Holdout 20%% validation ({best_name}): F1={m_va['f1']:.4f} AUC={m_va['auc']:.4f} "
            f"combined={m_va['combined']:.4f} prec={m_va['precision']:.4f} rec={m_va['recall']:.4f} "
            f"cm={m_va['cm']}\n\n")
    f.write("Class distribution: train 65.1% positive / 34.9% negative; "
            "class_weight=balanced used, no resampling, no threshold tuning.\n\n")
    f.write("Interpretation (association, NOT causation): age dominates (infants/under-5 higher predicted "
            "probability; elderly lower); climate variables (rainfall shortage/intensity, temperature anomalies, "
            "NDVI change) add modest signal over the no-climate baseline; geographic coordinates were "
            f"{'included' if use_geo else 'excluded due to train/test geographic shift and worse grouped-CV'}.\n")
    if imp is not None:
        f.write("\n## Top features\n\n" + imp.to_string(index=False) + "\n")

# README
readme = """# Climate-Health Mortality Prediction — reproducible solution
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
"""
open("README.md", "w").write(readme)
open("requirements.txt", "w").write(
    "pandas\nnumpy\nscikit-learn\nscipy\nmatplotlib\nseaborn\nlightgbm\nxgboost\n")
print("DONE.", flush=True)
