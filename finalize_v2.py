"""Finalize v2: greedy ensemble over full OOF pool -> retrain -> submission_improved_v2.csv."""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(42)
sys.path.insert(0, ".")
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
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

CANDS = ["proba_cal_rf", "proba_xgboost", "proba_xgb_s123", "proba_xgb_s7",
         "proba_rf", "proba_rf_s7", "proba_rf_s123", "proba_stack5", "proba_avg6"]
CANDS = [c for c in CANDS if c in oof.columns]
def comb_of(p):
    m = fold_metrics(y, p); return combined_score(m["f1"], m["auc"]), m
order = sorted(CANDS, key=lambda n: comb_of(oof[n].values)[0], reverse=True)
print("ranking:", [(n, round(comb_of(oof[n].values)[0], 4)) for n in order], flush=True)
ens, members = oof[order[0]].values.copy(), [order[0]]
best_c, best_m = comb_of(ens)
for n in order[1:]:
    trial = (ens * len(members) + oof[n].values) / (len(members) + 1)
    c, m = comb_of(trial)
    tag = ""
    if c > best_c + 1e-4: ens, members, best_c, best_m = trial, members + [n], c, m; tag = "KEEP"
    print(f"  try +{n}: {c:.4f} {tag}", flush=True)
print("FINAL MEMBERS:", members, "comb:", round(best_c, 4),
      "F1:", round(best_m["f1"], 4), "AUC:", round(best_m["auc"], 4), flush=True)

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
# map OOF columns to fitted predictors (stack5/avg6 recomputed as averages of member test probas)
test_P = {}
for s in [42, 7, 123]:
    p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_xgb(s))]); p.fit(X, y)
    test_P[f"proba_xgboost" if s == 42 else f"proba_xgb_s{s}"] = p.predict_proba(Xt)[:, 1]
    print("retrained xgb", s, flush=True)
for s in [42, 7, 123]:
    key = "proba_rf" if s == 42 else f"proba_rf_s{s}"
    if s == 42:
        base = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_rf(s))])
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=3); cal.fit(X, y)
        test_P["proba_cal_rf"] = cal.predict_proba(Xt)[:, 1]
    p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_rf(s))]); p.fit(X, y)
    test_P[key] = p.predict_proba(Xt)[:, 1]
    print("retrained rf", s, flush=True)

def get_test(col):
    if col in test_P: return test_P[col]
    if col == "proba_stack5":
        Z = np.column_stack([test_P["proba_rf"], test_P["proba_xgboost"], test_P["proba_xgb_s7"] if "proba_xgb_s7" in test_P else test_P["proba_xgboost"], test_P.get("proba_catboost", test_P["proba_xgboost"]), test_P.get("proba_histgb", test_P["proba_rf"])])
        # stack5 was trained on [rf,xgb,lgbm,cat,hgb] OOF; approximate with available members:
        from sklearn.linear_model import LogisticRegression
        Ztr = oof[["proba_rf", "proba_xgboost", "proba_lightgbm", "proba_catboost", "proba_histgb"]].values
        lr = LogisticRegression(max_iter=2000).fit(Ztr, y)
        return lr.predict_proba(Z)[:, 1]
    if col == "proba_avg6":
        return np.mean([test_P["proba_xgboost"], test_P["proba_rf"], test_P["proba_xgb_s7"],
                        test_P["proba_xgb_s123"], test_P["proba_rf_s7"], test_P["proba_rf_s123"]], axis=0)
    raise KeyError(col)

final = np.mean([get_test(c) for c in members], axis=0)
sub = build_submission(te_raw["ID"], final)
errs = verify_submission(sub, te_raw, sample_path="SampleSubmission.csv")
assert not errs, errs
sub.to_csv("submission_improved_v2.csv", index=False)
print("submission_improved_v2.csv:", sub.shape, "pos_rate=%.4f" % (sub["TargetF1"] == 1).mean(), flush=True)

# log to model_results
res = pd.read_csv("experiments/model_results.csv")
nrow = pd.DataFrame([dict(experiment="expF_final_v2", model="avg(" + "+".join(members) + ")",
    features="v2_tuned", cv_F1=round(best_m["f1"], 4), cv_ROC_AUC=round(best_m["auc"], 4),
    combined=round(best_c, 4), precision=round(best_m["precision"], 4),
    recall=round(best_m["recall"], 4), notes="greedy uniform over extended OOF pool")])
pd.concat([res, nrow], ignore_index=True).to_csv("experiments/model_results.csv", index=False)
import json
json.dump({"members": members, "combined": round(best_c, 4)}, open("experiments/final_ensemble_v2.json", "w"))
print("DONE finalize_v2")
