"""Aggressive training: location TE + spatial + v3 features, full model search."""
import os, sys, time, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(42)
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from scipy.stats import randint, uniform, loguniform
from src.data_loader import load_data, merge_climate
from src.features_v3 import add_features_v3, V3_NUM, V3_CAT
from src.encoding import target_encode_oof, target_encode_full, spatial_features
from src.validation import make_preprocess, fold_metrics, combined_score
from src.models import MODEL_FNS

os.makedirs("experiments", exist_ok=True)
tr_raw, te_raw, cl = load_data()
tr = add_features_v3(merge_climate(tr_raw, cl))
te = add_features_v3(merge_climate(te_raw, cl))

# ---- spatial features ----
tr = spatial_features(tr, n_clusters=8)
te = spatial_features(te, n_clusters=8)
SPATIAL = [f"dist_cluster_{c}" for c in range(8)] + ["dist_nearest_cluster"]
CAT_EXTRA = ["geo_cluster"]  # treat cluster as categorical

# ---- location target encoding (OOF for train, full for test) ----
loc_oof, loc_glob = target_encode_oof(tr, tr["is_climate_sensitive"], "location", n_splits=5, smooth=20)
tr["loc_te"] = loc_oof
_, tr_loc_test, _, _ = target_encode_full(tr, tr["is_climate_sensitive"], te, "location", smooth=20)
te["loc_te"] = tr_loc_test

# ---- frequency encoding ----
loc_freq = tr["location"].value_counts(normalize=True).to_dict()
tr["loc_freq"] = tr["location"].map(loc_freq).fillna(0)
te["loc_freq"] = te["location"].map(loc_freq).fillna(0)

# ---- zone frequency ----
zone_freq = tr["zone"].value_counts(normalize=True).to_dict()
tr["zone_freq"] = tr["zone"].map(zone_freq)
te["zone_freq"] = te["zone"].map(zone_freq)

y = tr["is_climate_sensitive"].values
NUM = V3_NUM + ["loc_te", "loc_freq", "zone_freq"] + SPATIAL
CAT = V3_CAT + CAT_EXTRA
X = tr[NUM + CAT]

# quick baseline: XGB with location
from xgboost import XGBClassifier
def mk_xgb_loc():
    return XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=3.0, scale_pos_weight=1.5,
                         n_jobs=-1, random_state=42, tree_method="hist", eval_metric="logloss")

def oof_eval(make_fn, X, y, num, cat, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    for ti, vi in skf.split(X, y):
        p = Pipeline([("pre", make_preprocess(num, cat)), ("clf", make_fn())])
        p.fit(X.iloc[ti], y[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:, 1]
    m = fold_metrics(y, oof); m["combined"] = combined_score(m["f1"], m["auc"])
    return m, oof

print("Baseline XGB + location TE:", flush=True)
m, oof_xgb = oof_eval(mk_xgb_loc, X, y, NUM, CAT)
print(f"  F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={m['combined']:.4f}", flush=True)

# ---- CatBoost native with location ----
from catboost import CatBoostClassifier, Pool
cat_cols = CAT
num_imp = SimpleImputer(strategy="median")
Xn = pd.DataFrame(num_imp.fit_transform(X[NUM]), columns=NUM, index=X.index)
Xcat = pd.concat([Xn, X[CAT].astype(str)], axis=1)
cat_idx = [Xcat.columns.get_loc(c) for c in CAT]
print("CatBoost native:", flush=True)
oof_cb = np.zeros(len(X))
for ti, vi in StratifiedKFold(5, shuffle=True, random_state=42).split(Xcat, y):
    m = CatBoostClassifier(iterations=1200, depth=6, learning_rate=0.04, l2_leaf_reg=3.0,
                           auto_class_weights="Balanced", random_seed=42,
                           verbose=False, allow_writing_files=False)
    m.fit(Xcat.iloc[ti], y[ti], cat_features=cat_idx)
    oof_cb[vi] = m.predict_proba(Xcat.iloc[vi])[:, 1]
m = fold_metrics(y, oof_cb); m["combined"] = combined_score(m["f1"], m["auc"])
print(f"  F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={m['combined']:.4f}", flush=True)

# ---- RandomForest with location ----
from sklearn.ensemble import RandomForestClassifier
def mk_rf_loc():
    return RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, max_features="log2",
                                  max_depth=None, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=42)
print("RF + location:", flush=True)
m, oof_rf = oof_eval(mk_rf_loc, X, y, NUM, CAT)
print(f"  F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={m['combined']:.4f}", flush=True)

# ---- LightGBM ----
from lightgbm import LGBMClassifier
def mk_lgbm_loc():
    return LGBMClassifier(n_estimators=1200, num_leaves=63, learning_rate=0.03,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=3,
                          lambda_l2=2.0, class_weight="balanced",
                          n_jobs=-1, random_state=42, verbose=-1)
print("LGBM + location:", flush=True)
m, oof_lgbm = oof_eval(mk_lgbm_loc, X, y, NUM, CAT)
print(f"  F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={m['combined']:.4f}", flush=True)

# ---- HistGB ----
from sklearn.ensemble import HistGradientBoostingClassifier
def mk_hgb_loc():
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          class_weight="balanced", random_state=42)
print("HistGB + location:", flush=True)
m, oof_hgb = oof_eval(mk_hgb_loc, X, y, NUM, CAT)
print(f"  F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={m['combined']:.4f}", flush=True)

# ---- ExtraTrees ----
from sklearn.ensemble import ExtraTreesClassifier
def mk_et_loc():
    return ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=3, max_features="log2",
                                max_depth=None, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=42)
print("ExtraTrees + location:", flush=True)
m, oof_et = oof_eval(mk_et_loc, X, y, NUM, CAT)
print(f"  F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={m['combined']:.4f}", flush=True)

# ---- save OOFs ----
oofs = {"xgb": oof_xgb, "catboost": oof_cb, "rf": oof_rf,
        "lgbm": oof_lgbm, "hgb": oof_hgb, "et": oof_et}
for k, v in oofs.items():
    print(f"{k}: combined={combined_score(*[fold_metrics(y,v)[k] for k in ['f1','auc']]):.4f}")

# quick ensemble
def comb_of(p):
    m = fold_metrics(y, p); return combined_score(m["f1"], m["auc"]), m

order = sorted(oofs, key=lambda n: comb_of(oofs[n])[0], reverse=True)
print("ranking:", [(n, round(comb_of(oofs[n])[0], 4)) for n in order])
ens, members = oofs[order[0]].copy(), [order[0]]
best_c, best_m = comb_of(ens)
for n in order[1:]:
    trial = (ens * len(members) + oofs[n]) / (len(members) + 1)
    c, m = comb_of(trial)
    if c > best_c + 1e-4:
        ens, members, best_c, best_m = trial, members + [n], c, m
        print(f"  + {n} -> {c:.4f}")
print(f"ENSEMBLE {members}: comb={best_c:.4f} F1={best_m['f1']:.4f} AUC={best_m['auc']:.4f}")

# retrain members on full, predict test
test_P = {}
Xt = te[NUM + CAT]
for s in [42, 7]:
    p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_xgb_loc())]); p.fit(X, y)
    test_P[f"xgb_s{s}"] = p.predict_proba(Xt)[:, 1]
for s in [42, 7]:
    p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_rf_loc())]); p.fit(X, y)
    test_P[f"rf_s{s}"] = p.predict_proba(Xt)[:, 1]
# catboost native
cat_imp = SimpleImputer(strategy="median")
Xn_full = pd.DataFrame(cat_imp.fit_transform(X[NUM]), columns=NUM, index=X.index)
Xcat_full = pd.concat([Xn_full, X[CAT].astype(str)], axis=1)
Xn_test = pd.DataFrame(cat_imp.transform(te[NUM]), columns=NUM, index=te.index)
Xcat_test = pd.concat([Xn_test, te[CAT].astype(str)], axis=1)
cb = CatBoostClassifier(iterations=1200, depth=6, learning_rate=0.04, l2_leaf_reg=3.0,
                        auto_class_weights="Balanced", random_seed=42,
                        verbose=False, allow_writing_files=False)
cb.fit(Xcat_full, y, cat_features=cat_idx)
test_P["catboost"] = cb.predict_proba(Xcat_test)[:, 1]

# lgbm
for s in [42]:
    p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_lgbm_loc())]); p.fit(X, y)
    test_P[f"lgbm_s{s}"] = p.predict_proba(Xt)[:, 1]

# ensemble test
Zt = np.column_stack([test_P[m] for m in members])
final = Zt.mean(axis=1)

# submit
from src.submission import build_submission, verify_submission
sub = build_submission(te_raw["ID"], final)
errs = verify_submission(sub, te_raw, sample_path="SampleSubmission.csv")
assert not errs, errs
sub.to_csv("submission_aggressive_v1.csv", index=False)
print("submission_aggressive_v1.csv:", sub.shape, "pos_rate=%.4f" % (sub["TargetF1"]==1).mean())

# save
pd.DataFrame({**{"ID": tr_raw["ID"], "y": y}, **{f"proba_{k}": v for k,v in oofs.items()}}).to_csv(
    "experiments/oof_aggressive.csv", index=False)
json.dump({"members": members, "combined": round(best_c, 4)},
          open("experiments/ensemble_aggressive_v1.json", "w"))
print("DONE aggressive v1")