"""Final submission: rolling-year time-aware ensemble (best validated: RF 0.8338).
Threshold fixed at 0.5. No test labels used."""
import sys, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from src.data_loader import load_data, merge_climate
from src.features_advanced import add_features_advanced, compute_location_climatology, ADV_NUM, ADV_CAT
from src.encoding import target_encode_oof, target_encode_full
from src.validation import make_preprocess, fold_metrics, combined_score
from src.submission import build_submission, verify_submission

tr_raw, te_raw, cl = load_data()
tr = merge_climate(tr_raw, cl)
te = merge_climate(te_raw, cl)
clim = compute_location_climatology(tr)
tr = add_features_advanced(tr, clim); te = add_features_advanced(te, clim)

# location TE (train OOF / test from full train)
loc_oof, _ = target_encode_oof(tr, pd.Series(tr["is_climate_sensitive"]), "location", n_splits=5, smooth=30)
tr["loc_te"] = loc_oof
_, loc_te_test, _, _ = target_encode_full(tr, pd.Series(tr["is_climate_sensitive"]), te, "location", smooth=30)
te["loc_te"] = loc_te_test

NUM = ADV_NUM + ["loc_te"]
CAT = ADV_CAT
X = tr[NUM + CAT]; Xt = te[NUM + CAT]
y = tr["is_climate_sensitive"].values

def mk_rf(seed=42):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=1200, min_samples_leaf=2, max_features="sqrt",
                                  max_depth=20, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=seed)
def mk_xgb(seed=42):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.03, subsample=0.7,
                         colsample_bytree=0.7, reg_lambda=1.0, scale_pos_weight=1.5,
                         n_jobs=-1, random_state=seed, tree_method="hist", eval_metric="logloss")
def mk_lgbm(seed=42):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=1200, num_leaves=63, learning_rate=0.03,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=3,
                          lambda_l2=1.5, class_weight="balanced", n_jobs=-1,
                          random_state=seed, verbose=-1)
MODELS = {"rf": mk_rf, "xgb": mk_xgb, "lgbm": mk_lgbm}

# --- rolling year-by-year OOF with correct masked aggregation ---
years = sorted(tr["death_year"].unique())
oofs = {}
for name, fn in MODELS.items():
    oof = np.zeros(len(X)); mask = np.zeros(len(X), bool)
    idx_end = 4
    for i in range(5):
        if idx_end >= len(years) - 1: break
        ti = tr.index[tr["death_year"].isin(years[:idx_end + 1])].values
        vi = tr.index[tr["death_year"] == years[idx_end + 1]].values
        p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", fn())])
        p.fit(X.iloc[ti], y[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:, 1]; mask[vi] = True
        idx_end += 1
    oofs[name] = (oof, mask)
    m = fold_metrics(y[mask], oof[mask])
    print(f"{name}: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={combined_score(m['f1'], m['auc']):.4f} "
          f"n={mask.sum()}", flush=True)

mask = oofs["rf"][1]
# greedy ensemble on masked OOF
def comb_of(p):
    m = fold_metrics(y[mask], p[mask]); return combined_score(m["f1"], m["auc"]), m
names = list(oofs)
ens = oofs["rf"][0].copy(); members = ["rf"]
best_c, best_m = comb_of(ens)
for n in ["xgb", "lgbm"]:
    trial = (ens * len(members) + oofs[n][0]) / (len(members) + 1)
    c, m = comb_of(trial)
    if c > best_c + 1e-4:
        ens, members, best_c, best_m = trial, members + [n], c, m
print(f"ENSEMBLE {members}: F1={best_m['f1']:.4f} AUC={best_m['auc']:.4f} comb={best_c:.4f}", flush=True)

# --- retrain members on ALL train rows, predict test ---
parts = []
for n in members:
    p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", MODELS[n]())])
    p.fit(X, y); parts.append(p.predict_proba(Xt)[:, 1])
    print("retrained", n, flush=True)
final = np.mean(parts, axis=0)

sub = build_submission(te_raw["ID"], final)
errs = verify_submission(sub, te_raw, sample_path="SampleSubmission.csv")
assert not errs, errs
sub.to_csv("submission_rolling_v4.csv", index=False)
print(f"submission_rolling_v4.csv pos_rate={sub['TargetF1'].mean():.4f} "
      f"rows={len(sub)} mean_proba={final.mean():.4f}", flush=True)
json.dump({"members": members, "rolling_combined": round(best_c, 4),
           "rolling_f1": round(best_m["f1"], 4), "rolling_auc": round(best_m["auc"], 4)},
          open("experiments/final_rolling_v4.json", "w"), indent=2)
print("DONE v4")
