"""Fast focused search: XGB + RF + CatBoost with location + spatial, tuned."""
import os, sys, time, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(42)
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.data_loader import load_data, merge_climate
from src.features_v3 import add_features_v3, V3_NUM, V3_CAT
from src.encoding import target_encode_oof, target_encode_full, spatial_features
from src.validation import make_preprocess, fold_metrics, combined_score

os.makedirs("experiments", exist_ok=True)
tr_raw, te_raw, cl = load_data()
tr = add_features_v3(merge_climate(tr_raw, cl))
te = add_features_v3(merge_climate(te_raw, cl))

# spatial
tr = spatial_features(tr, n_clusters=8)
te = spatial_features(te, n_clusters=8)

# location TE OOF
loc_oof, loc_glob = target_encode_oof(tr, tr["is_climate_sensitive"], "location", n_splits=5, smooth=30)
tr["loc_te"] = loc_oof
_, loc_te_test, _, _ = target_encode_full(tr, tr["is_climate_sensitive"], te, "location", smooth=30)
te["loc_te"] = loc_te_test

# freq enc
loc_freq = tr["location"].value_counts(normalize=True).to_dict()
tr["loc_freq"] = tr["location"].map(loc_freq).fillna(0)
te["loc_freq"] = te["location"].map(loc_freq).fillna(0)

zone_freq = tr["zone"].value_counts(normalize=True).to_dict()
tr["zone_freq"] = tr["zone"].map(zone_freq)
te["zone_freq"] = te["zone"].map(zone_freq)

y = tr["is_climate_sensitive"].values
NUM = V3_NUM + ["loc_te", "loc_freq", "zone_freq"] + \
      [f"dist_cluster_{c}" for c in range(8)] + ["dist_nearest_cluster"]
CAT = V3_CAT + ["geo_cluster"]
X = tr[NUM + CAT]

def oof_eval(make_fn, X, y, num, cat, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    for ti, vi in skf.split(X, y):
        p = Pipeline([("pre", make_preprocess(num, cat)), ("clf", make_fn())])
        p.fit(X.iloc[ti], y[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:, 1]
    m = fold_metrics(y, oof); m["combined"] = combined_score(m["f1"], m["auc"])
    return m, oof

def comb_of(p):
    m = fold_metrics(y, p); return combined_score(m["f1"], m["auc"]), m

results = {}

# XGBoost with tuned params
from xgboost import XGBClassifier
best_xgb = {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.04,
            "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0,
            "scale_pos_weight": 1.5, "n_jobs": -1, "random_state": 42,
            "tree_method": "hist", "eval_metric": "logloss"}
def mk_xgb(): return XGBClassifier(**best_xgb)

m, oof_xgb = oof_eval(mk_xgb, X, y, NUM, CAT)
print(f"XGB: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={m['combined']:.4f}")
results["xgb"] = {"oof": oof_xgb, "m": m}

# RandomForest
from sklearn.ensemble import RandomForestClassifier
def mk_rf():
    return RandomForestClassifier(n_estimators=1200, min_samples_leaf=2,
                                  max_features="log2", max_depth=None,
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=42)

m, oof_rf = oof_eval(mk_rf, X, y, NUM, CAT)
print(f"RF: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={m['combined']:.4f}")
results["rf"] = {"oof": oof_rf, "m": m}

# CatBoost native
from catboost import CatBoostClassifier, Pool
cat_imp = SimpleImputer(strategy="median")
Xn = pd.DataFrame(cat_imp.fit_transform(X[NUM]), columns=NUM, index=X.index)
Xcat = pd.concat([Xn, X[CAT].astype(str)], axis=1)
cat_idx = [Xcat.columns.get_loc(c) for c in CAT]

oof_cb = np.zeros(len(X))
for ti, vi in StratifiedKFold(5, shuffle=True, random_state=42).split(Xcat, y):
    m = CatBoostClassifier(iterations=1500, depth=6, learning_rate=0.035, l2_leaf_reg=2.5,
                           auto_class_weights="Balanced", random_seed=42,
                           verbose=False, allow_writing_files=False)
    m.fit(Xcat.iloc[ti], y[ti], cat_features=cat_idx)
    oof_cb[vi] = m.predict_proba(Xcat.iloc[vi])[:, 1]
m = fold_metrics(y, oof_cb); m["combined"] = combined_score(m["f1"], m["auc"])
print(f"CatBoost: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={m['combined']:.4f}")
results["catboost"] = {"oof": oof_cb, "m": m}

# LightGBM
from lightgbm import LGBMClassifier
def mk_lgbm():
    return LGBMClassifier(n_estimators=1500, num_leaves=63, learning_rate=0.025,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=3,
                          lambda_l2=1.5, class_weight="balanced",
                          n_jobs=-1, random_state=42, verbose=-1)
m, oof_lgbm = oof_eval(mk_lgbm, X, y, NUM, CAT)
print(f"LGBM: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={m['combined']:.4f}")
results["lgbm"] = {"oof": oof_lgbm, "m": m}

# HistGB
from sklearn.ensemble import HistGradientBoostingClassifier
def mk_hgb():
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=0.04,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          class_weight="balanced", random_state=42)
m, oof_hgb = oof_eval(mk_hgb, X, y, NUM, CAT)
print(f"HistGB: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={m['combined']:.4f}")
results["hgb"] = {"oof": oof_hgb, "m": m}

# ranking
oofs = {k: v["oof"] for k, v in results.items()}
order = sorted(oofs, key=lambda n: comb_of(oofs[n])[0], reverse=True)
print("Ranking:", [(n, round(comb_of(oofs[n])[0], 4)) for n in order])

# greedy ensemble
ens, members = oofs[order[0]].copy(), [order[0]]
best_c, best_m = comb_of(ens)
for n in order[1:]:
    trial = (ens * len(members) + oofs[n]) / (len(members) + 1)
    c, m = comb_of(trial)
    if c > best_c + 1e-4:
        ens, members, best_c, best_m = trial, members + [n], c, m
        print(f"  + {n} -> {c:.4f}")
print(f"ENSEMBLE {members}: comb={best_c:.4f} F1={best_m['f1']:.4f} AUC={best_m['auc']:.4f}")

# retrain and predict test
Xt = te[NUM + CAT]
test_P = {}

# XGB
p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_xgb())]); p.fit(X, y)
test_P["xgb"] = p.predict_proba(Xt)[:, 1]
print("retrained xgb")

# RF
p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_rf())]); p.fit(X, y)
test_P["rf"] = p.predict_proba(Xt)[:, 1]
print("retrained rf")

# CatBoost
Xn_full = pd.DataFrame(cat_imp.fit_transform(X[NUM]), columns=NUM, index=X.index)
Xcat_full = pd.concat([Xn_full, X[CAT].astype(str)], axis=1)
Xn_test = pd.DataFrame(cat_imp.transform(te[NUM]), columns=NUM, index=te.index)
Xcat_test = pd.concat([Xn_test, te[CAT].astype(str)], axis=1)
cb = CatBoostClassifier(iterations=1500, depth=6, learning_rate=0.035, l2_leaf_reg=2.5,
                        auto_class_weights="Balanced", random_seed=42,
                        verbose=False, allow_writing_files=False)
cb.fit(Xcat_full, y, cat_features=cat_idx)
test_P["catboost"] = cb.predict_proba(Xcat_test)[:, 1]
print("retrained catboost")

# LGBM
p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_lgbm())]); p.fit(X, y)
test_P["lgbm"] = p.predict_proba(Xt)[:, 1]
print("retrained lgbm")

# HistGB
p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_hgb())]); p.fit(X, y)
test_P["hgb"] = p.predict_proba(Xt)[:, 1]
print("retrained hgb")

# ensemble test
Zt = np.column_stack([test_P[m] for m in members])
final = Zt.mean(axis=1)

from src.submission import build_submission, verify_submission
sub = build_submission(te_raw["ID"], final)
errs = verify_submission(sub, te_raw, sample_path="SampleSubmission.csv")
assert not errs, errs
sub.to_csv("submission_aggressive_v2.csv", index=False)
print(f"submission_aggressive_v2.csv: pos_rate={sub['TargetF1'].mean():.4f}")

# save
pd.DataFrame({**{"ID": tr_raw["ID"], "y": y}, **{f"proba_{k}": v for k,v in oofs.items()}}).to_csv(
    "experiments/oof_aggressive_v2.csv", index=False)
json.dump({"members": members, "combined": round(best_c, 4)},
          open("experiments/ensemble_aggressive_v2.json", "w"))
print("DONE")