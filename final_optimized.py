"""Final optimized run: smooth=30 for XGB, 50 for RF, lat/lon, aggressive XGB, stacking."""
import os, sys, time, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(42)
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from src.data_loader import load_data, merge_climate
from src.features_v3 import add_features_v3, V3_NUM, V3_CAT
from src.encoding import target_encode_oof, target_encode_full, spatial_features
from src.validation import make_preprocess, fold_metrics, combined_score
from src.submission import build_submission, verify_submission

os.makedirs("experiments", exist_ok=True)
tr_raw, te_raw, cl = load_data()
tr = add_features_v3(merge_climate(tr_raw, cl))
te = add_features_v3(merge_climate(te_raw, cl))

# spatial
tr = spatial_features(tr, n_clusters=8)
te = spatial_features(te, n_clusters=8)

y = tr["is_climate_sensitive"].values
NUM_BASE = V3_NUM + [f"dist_cluster_{c}" for c in range(8)] + ["dist_nearest_cluster"]
CAT = V3_CAT + ["geo_cluster"]

def build_data(smooth_xgb, smooth_rf):
    """Build data with different smooth for each model."""
    # XGB data
    loc_oof_xgb, _ = target_encode_oof(tr, pd.Series(y), "location", n_splits=5, smooth=smooth_xgb)
    tr["loc_te_xgb"] = loc_oof_xgb
    _, loc_te_test_xgb, _, _ = target_encode_full(tr, pd.Series(y), te, "location", smooth=smooth_xgb)
    te["loc_te_xgb"] = loc_te_test_xgb
    
    # RF data
    loc_oof_rf, _ = target_encode_oof(tr, pd.Series(y), "location", n_splits=5, smooth=smooth_rf)
    tr["loc_te_rf"] = loc_oof_rf
    _, loc_te_test_rf, _, _ = target_encode_full(tr, pd.Series(y), te, "location", smooth=smooth_rf)
    te["loc_te_rf"] = loc_te_test_rf
    
    # freq (same for both)
    loc_freq = tr["location"].value_counts(normalize=True).to_dict()
    tr["loc_freq"] = tr["location"].map(loc_freq).fillna(0)
    te["loc_freq"] = te["location"].map(loc_freq).fillna(0)
    zone_freq = tr["zone"].value_counts(normalize=True).to_dict()
    tr["zone_freq"] = tr["zone"].map(zone_freq)
    te["zone_freq"] = te["zone"].map(zone_freq)
    
    NUM_XGB = NUM_BASE + ["loc_te_xgb", "loc_freq", "zone_freq", "latitude", "longitude"]
    NUM_RF = NUM_BASE + ["loc_te_rf", "loc_freq", "zone_freq", "latitude", "longitude"]
    
    return tr[NUM_XGB + CAT], tr[NUM_RF + CAT], te[NUM_XGB + CAT], te[NUM_RF + CAT]

X_xgb, X_rf, Xt_xgb, Xt_rf = build_data(30, 50)

def oof5(make_fn, X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for ti, vi in skf.split(X, y):
        p = Pipeline([("pre", make_preprocess(X.columns.tolist()[:-len(CAT)], CAT)), ("clf", make_fn())])
        p.fit(X.iloc[ti], y[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:, 1]
    m = fold_metrics(y, oof); m["combined"] = combined_score(m["f1"], m["auc"])
    return m, oof

# Aggressive XGB
from xgboost import XGBClassifier
def mk_xgb(): return XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.035,
                                   subsample=0.75, colsample_bytree=0.75, reg_lambda=1.0,
                                   scale_pos_weight=1.5, n_jobs=-1, random_state=42,
                                   tree_method="hist", eval_metric="logloss")

# RF
from sklearn.ensemble import RandomForestClassifier
def mk_rf(): return RandomForestClassifier(n_estimators=1200, min_samples_leaf=1,
                                           max_features="log2", class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=42)

m_xgb, oof_xgb = oof5(mk_xgb, X_xgb, y)
m_rf, oof_rf = oof5(mk_rf, X_rf, y)
print(f"XGB: F1={m_xgb['f1']:.4f} AUC={m_xgb['auc']:.4f} comb={m_xgb['combined']:.4f}")
print(f"RF:  F1={m_rf['f1']:.4f} AUC={m_rf['auc']:.4f} comb={m_rf['combined']:.4f}")

# Stacking meta-learner on OOF
Z = np.column_stack([oof_xgb, oof_rf])
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof_stack = np.zeros(len(y))
for ti, vi in skf.split(Z, y):
    lr = LogisticRegression(max_iter=2000, C=0.5).fit(Z[ti], y[ti])
    oof_stack[vi] = lr.predict_proba(Z[vi])[:, 1]
m_stack = fold_metrics(y, oof_stack); m_stack["combined"] = combined_score(m_stack["f1"], m_stack["auc"])
print(f"Stack: F1={m_stack['f1']:.4f} AUC={m_stack['auc']:.4f} comb={m_stack['combined']:.4f}")

# Uniform ensemble
oof_ens = (oof_xgb + oof_rf) / 2
m_ens = fold_metrics(y, oof_ens); m_ens["combined"] = combined_score(m_ens["f1"], m_ens["auc"])
print(f"Ens:   F1={m_ens['f1']:.4f} AUC={m_ens['auc']:.4f} comb={m_ens['combined']:.4f}")

# Pick best
if m_stack["combined"] > m_ens["combined"]:
    best_oof, method = oof_stack, "stack"
    best_m = m_stack
else:
    best_oof, method = oof_ens, "uniform"
    best_m = m_ens
print(f"BEST ({method}): F1={best_m['f1']:.4f} AUC={best_m['auc']:.4f} comb={best_m['combined']:.4f}")

# Retrain and predict test
if method == "stack":
    # retrain meta on full OOF
    lr = LogisticRegression(max_iter=2000, C=0.5).fit(Z, y)
    
    # XGB test
    p_xgb = Pipeline([("pre", make_preprocess(X_xgb.columns.tolist()[:-len(CAT)], CAT)), ("clf", mk_xgb())])
    p_xgb.fit(X_xgb, y); p_xgb_proba = p_xgb.predict_proba(Xt_xgb)[:, 1]
    
    # RF test
    p_rf = Pipeline([("pre", make_preprocess(X_rf.columns.tolist()[:-len(CAT)], CAT)), ("clf", mk_rf())])
    p_rf.fit(X_rf, y); p_rf_proba = p_rf.predict_proba(Xt_rf)[:, 1]
    
    Zt = np.column_stack([p_xgb_proba, p_rf_proba])
    final = lr.predict_proba(Zt)[:, 1]
else:
    p_xgb = Pipeline([("pre", make_preprocess(X_xgb.columns.tolist()[:-len(CAT)], CAT)), ("clf", mk_xgb())])
    p_xgb.fit(X_xgb, y); p_xgb_proba = p_xgb.predict_proba(Xt_xgb)[:, 1]
    p_rf = Pipeline([("pre", make_preprocess(X_rf.columns.tolist()[:-len(CAT)], CAT)), ("clf", mk_rf())])
    p_rf.fit(X_rf, y); p_rf_proba = p_rf.predict_proba(Xt_rf)[:, 1]
    final = (p_xgb_proba + p_rf_proba) / 2

sub = build_submission(te_raw["ID"], final)
errs = verify_submission(sub, te_raw, sample_path="SampleSubmission.csv")
assert not errs, errs
sub.to_csv("submission_final_optimized.csv", index=False)
print(f"submission_final_optimized.csv: pos_rate={sub['TargetF1'].mean():.4f}")

# save
pd.DataFrame({"ID": tr_raw["ID"], "y": y, "proba_xgb": oof_xgb, "proba_rf": oof_rf,
              "proba_stack": oof_stack, "proba_ens": oof_ens}).to_csv(
    "experiments/oof_final_optimized.csv", index=False)
json.dump({"method": method, "combined": round(best_m["combined"], 4),
           "f1": round(best_m["f1"], 4), "auc": round(best_m["auc"], 4)},
          open("experiments/final_optimized.json", "w"))
print("DONE final_optimized")