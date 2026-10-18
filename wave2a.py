"""Wave 2A: stacking on saved OOF + seed-averaged OOF + calibration. Fast."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(42)
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.data_loader import load_data, merge_climate
from src.feature_engineering import add_features, BASE_CATEGORICAL, BASE_NUMERIC
from src.features_v2 import add_features_v2, V2_EXTRA
from src.validation import make_preprocess, fold_metrics, combined_score

os.makedirs("experiments", exist_ok=True)
tr_raw, te_raw, cl = load_data()
tr = add_features_v2(add_features(merge_climate(tr_raw, cl)))
y = tr["is_climate_sensitive"].values
NUM = BASE_NUMERIC + V2_EXTRA
CAT = BASE_CATEGORICAL
X = tr[NUM + CAT]

def rep(p, tag):
    m = fold_metrics(y, p); c = combined_score(m["f1"], m["auc"])
    print(f"{tag}: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={c:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f}", flush=True)
    return c, m

oof = pd.read_csv("experiments/oof_predictions.csv")
base_cols = ["proba_rf", "proba_xgboost", "proba_lightgbm", "proba_catboost", "proba_histgb"]
Z = oof[base_cols].values
print("sanity single:", {c: round(rep(oof[c].values, c)[0], 4) for c in base_cols})

# stacking: logistic meta with 5-fold CV over OOF matrix (no leakage: meta sees only OOF)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
meta_oof = np.zeros(len(y))
for ti, vi in skf.split(Z, y):
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Z[ti][:, [0, 1]], y[ti]); meta_oof[vi] = lr.predict_proba(Z[vi][:, [0, 1]])[:, 1]
rep(meta_oof, "stack_logreg[xgb,rf]")
meta_oof5 = np.zeros(len(y))
for ti, vi in skf.split(Z, y):
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Z[ti], y[ti]); meta_oof5[vi] = lr.predict_proba(Z[vi])[:, 1]
rep(meta_oof5, "stack_logreg[all5]")
oof["proba_stack2"] = meta_oof; oof["proba_stack5"] = meta_oof5

# seed-averaged OOF: xgb/rf x seeds 7,123 (5-fold each)
def mk_xgb(s):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.07248, subsample=0.7599,
                         colsample_bytree=0.6187, reg_lambda=7.803, scale_pos_weight=1.5,
                         n_jobs=-1, random_state=s, tree_method="hist", eval_metric="logloss")
def mk_rf(s):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, max_features="log2",
                                  max_depth=None, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=s)
import time
new_oofs = {}
for name, mk in [("xgb", mk_xgb), ("rf", mk_rf)]:
    for s in [7, 123]:
        t0 = time.time(); o = np.zeros(len(X))
        for ti, vi in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
            p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk(s))])
            p.fit(X.iloc[ti], y[ti]); o[vi] = p.predict_proba(X.iloc[vi])[:, 1]
        new_oofs[f"{name}_s{s}"] = o; oof[f"proba_{name}_s{s}"] = np.round(o, 6)
        print(f"{name} seed{s} ({time.time()-t0:.0f}s):", flush=True); rep(o, f"{name}_s{s}")

avg6 = np.mean([oof["proba_xgboost"], oof["proba_rf"], oof["proba_xgb_s7"],
                oof["proba_xgb_s123"], oof["proba_rf_s7"], oof["proba_rf_s123"]], axis=0)
rep(avg6, "avg6_xgb_rf_3seeds")
oof["proba_avg6"] = np.round(avg6, 6)

# calibration: CalibratedClassifierCV(sigmoid, cv=3) outer 5-fold for xgb + rf(seed42)
from sklearn.calibration import CalibratedClassifierCV
for name, mk in [("xgb", lambda: mk_xgb(42)), ("rf", lambda: mk_rf(42))]:
    t0 = time.time(); o = np.zeros(len(X))
    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        base = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk())])
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        cal.fit(X.iloc[ti], y[ti]); o[vi] = cal.predict_proba(X.iloc[vi])[:, 1]
    oof[f"proba_cal_{name}"] = np.round(o, 6)
    print(f"calibrated {name} ({time.time()-t0:.0f}s):", flush=True); rep(o, f"cal_{name}")

oof.to_csv("experiments/oof_predictions.csv", index=False)
print("DONE wave2A")
