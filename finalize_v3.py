"""Final wave: weighted OOF ensemble (NNLS on 5-fold OOF), refit on full train."""
import sys, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(42)
sys.path.insert(0, ".")
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from scipy.optimize import nnls
from src.data_loader import load_data, merge_climate
from src.feature_engineering import add_features, BASE_CATEGORICAL, BASE_NUMERIC
from src.features_v2 import add_features_v2, V2_EXTRA
from src.validation import make_preprocess, fold_metrics, combined_score
from src.submission import build_submission, verify_submission

tr_raw, te_raw, cl = load_data()
tr = add_features_v2(add_features(merge_climate(tr_raw, cl)))
te = add_features_v2(add_features(merge_climate(te_raw, cl)))
y = tr["is_climate_sensitive"].values
NUM = BASE_NUMERIC + V2_EXTRA
CAT = BASE_CATEGORICAL
X = tr[NUM + CAT]
Xt = te[NUM + CAT]
oof = pd.read_csv("experiments/oof_predictions.csv")

# ---- choose pool: strong, diverse, calibrated ----
pool = ["proba_cal_rf", "proba_xgboost", "proba_xgb_s7", "proba_xgb_s123",
        "proba_rf_s7", "proba_rf_s123", "proba_stack5", "proba_avg6"]
pool = [c for c in pool if c in oof.columns]
Z = oof[pool].values

# ---- NNLS for weights minimizing Brier score on OOF (convex, non-neg, sums to 1) ----
# Brier = mean((p - y)^2) = p^T p - 2 p^T y + y^T y. Min wrt weights s.t. w>=0, sum=1
# Equivalent to min ||Z w - y||^2 s.t. w>=0, sum=1
# Solve unconstrained NNLS then project onto simplex, or use cvxpy if available. Simple:
from scipy.optimize import minimize
def obj(w):
    return np.mean((Z @ w - y) ** 2)
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1)] * len(pool)
w0 = np.ones(len(pool)) / len(pool)
res = minimize(obj, w0, bounds=bounds, constraints=cons, method='SLSQP')
w_opt = res.x
print("NNLS weights:", {pool[i]: round(float(w_opt[i]), 4) for i in range(len(pool))}, flush=True)

w_opt = np.maximum(w_opt, 0); w_opt = w_opt / w_opt.sum()
p_opt = Z @ w_opt
m = fold_metrics(y, p_opt)
c = combined_score(m["f1"], m["auc"])
print(f"weighted_opt: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={c:.4f}", flush=True)

# uniform for comparison
p_uni = Z.mean(axis=1)
m = fold_metrics(y, p_uni)
c = combined_score(m["f1"], m["auc"])
print(f"uniform_avg: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={c:.4f}", flush=True)

# pick best
if c > combined_score(*[fold_metrics(y, p_opt)[k] for k in ['f1','auc']]):
    w_final = np.ones(len(pool)) / len(pool); method = "uniform"
else:
    w_final = w_opt; method = "nnls"
p_final = Z @ w_final
m = fold_metrics(y, p_final); c = combined_score(m["f1"], m["auc"])
print(f"FINAL ({method}): F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={c:.4f}", flush=True)

# ---- retrain needed models & build test probas ----
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

test_P = {}
for s in [42, 7, 123]:
    p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_xgb(s))]); p.fit(X, y)
    test_P[f"proba_xgboost" if s==42 else f"proba_xgb_s{s}"] = p.predict_proba(Xt)[:, 1]
for s in [42, 7, 123]:
    key = "proba_rf" if s==42 else f"proba_rf_s{s}"
    if s == 42:
        base = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_rf(s))])
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=3); cal.fit(X, y)
        test_P["proba_cal_rf"] = cal.predict_proba(Xt)[:, 1]
    p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_rf(s))]); p.fit(X, y)
    test_P[key] = p.predict_proba(Xt)[:, 1]

# stack5 / avg6 as averages of available
if "proba_stack5" in pool:
    # approximate with available members
    test_P["proba_stack5"] = np.mean([test_P["proba_rf"], test_P["proba_xgboost"]], axis=0)
if "proba_avg6" in pool:
    test_P["proba_avg6"] = np.mean([test_P["proba_xgboost"], test_P["proba_rf"],
                                    test_P["proba_xgb_s7"], test_P["proba_xgb_s123"],
                                    test_P["proba_rf_s7"], test_P["proba_rf_s123"]], axis=0)

Zt = np.column_stack([test_P[c] for c in pool])
final_proba = Zt @ w_final
sub = build_submission(te_raw["ID"], final_proba)
errs = verify_submission(sub, te_raw, sample_path="SampleSubmission.csv")
assert not errs, errs
sub.to_csv("submission_improved_v3.csv", index=False)
print("submission_improved_v3.csv:", sub.shape, "pos_rate=%.4f" % (sub["TargetF1"]==1).mean(), flush=True)
json.dump({"method": method, "pool": pool, "weights": w_final.tolist(), "combined": round(c,4)},
          open("experiments/final_ensemble_v3.json", "w"))
print("DONE v3")