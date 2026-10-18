"""Finalize: 5-fold OOF of tuned models -> greedy ensemble -> seed robustness ->
retrain on full train -> submission_improved.csv -> reports. Threshold always 0.5."""
import os, sys, time, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(42)
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from src.data_loader import load_data, merge_climate
from src.feature_engineering import add_features, BASE_CATEGORICAL, BASE_NUMERIC
from src.features_v2 import add_features_v2, V2_EXTRA
from src.validation import make_preprocess, fold_metrics, combined_score
from src.submission import build_submission, verify_submission

os.makedirs("experiments", exist_ok=True)
tr_raw, te_raw, cl = load_data()
tr = add_features_v2(add_features(merge_climate(tr_raw, cl)))
te = add_features_v2(add_features(merge_climate(te_raw, cl)))
y = tr["is_climate_sensitive"]
NUM = BASE_NUMERIC + V2_EXTRA
CAT = BASE_CATEGORICAL
X = tr[NUM + CAT]

def mk_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, max_features="log2",
                                  max_depth=None, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=42)
def mk_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.07248, subsample=0.7599,
                         colsample_bytree=0.6187, reg_lambda=7.803, scale_pos_weight=1.5,
                         n_jobs=-1, random_state=42, tree_method="hist", eval_metric="logloss")
def mk_lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=1000, num_leaves=92, learning_rate=0.05775,
                          feature_fraction=0.923, bagging_fraction=0.9584, bagging_freq=3,
                          lambda_l2=2.885, class_weight="balanced",
                          n_jobs=-1, random_state=42, verbose=-1)
def mk_cat():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=1100, depth=6, learning_rate=0.04684, l2_leaf_reg=4.999,
                              auto_class_weights="Balanced", random_seed=42,
                              verbose=False, allow_writing_files=False)
def mk_hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06217, max_leaf_nodes=16,
                                          l2_regularization=3.795, class_weight="balanced",
                                          random_state=42)
MODELS = {"rf": mk_rf, "xgboost": mk_xgb, "lightgbm": mk_lgbm, "catboost": mk_cat, "histgb": mk_hgb}

def oof5(make_fn, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X)); folds = []
    for ti, vi in skf.split(X, y):
        p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", make_fn())])
        p.fit(X.iloc[ti], y.iloc[ti]); pr = p.predict_proba(X.iloc[vi])[:, 1]
        oof[vi] = pr
        m = fold_metrics(y.iloc[vi].values, pr); m["combined"] = combined_score(m["f1"], m["auc"])
        folds.append(m)
    ov = fold_metrics(y.values, oof); ov["combined"] = combined_score(ov["f1"], ov["auc"])
    return ov, oof, folds

oofs, rows = {}, []
for name, fn in MODELS.items():
    t0 = time.time()
    ov, oof, folds = oof5(fn)
    oofs[name] = oof
    f1s = [f["f1"] for f in folds]; aus = [f["auc"] for f in folds]
    print(f"5FOLD {name}: F1={ov['f1']:.4f}(sd{np.std(f1s):.4f}) AUC={ov['auc']:.4f}(sd{np.std(aus):.4f}) "
          f"comb={ov['combined']:.4f} prec={ov['precision']:.4f} rec={ov['recall']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    rows.append(dict(experiment="expC_tuned5fold", model=name, features="v2_tuned",
        cv_F1=round(ov["f1"],4), cv_ROC_AUC=round(ov["auc"],4), combined=round(ov["combined"],4),
        precision=round(ov["precision"],4), recall=round(ov["recall"],4),
        notes=f"F1sd={np.std(f1s):.4f} AUCsd={np.std(aus):.4f}"))

def comb_of(p):
    m = fold_metrics(y.values, p); return combined_score(m["f1"], m["auc"]), m
order = sorted(oofs, key=lambda n: comb_of(oofs[n])[0], reverse=True)
print("OOF ranking:", [(n, round(comb_of(oofs[n])[0], 4)) for n in order], flush=True)
ens, members = oofs[order[0]].copy(), [order[0]]
best_c, best_m = comb_of(ens)
for n in order[1:]:
    trial = (ens * len(members) + oofs[n]) / (len(members) + 1)
    c, m = comb_of(trial)
    if c > best_c + 1e-4: ens, members, best_c, best_m = trial, members + [n], c, m
    print(f"  try +{n}: {c:.4f} {'KEEP' if c > best_c else ''}", flush=True)
print("ENSEMBLE:", members, round(best_c, 4), flush=True)
rows.append(dict(experiment="expD_ensemble", model="avg(" + "+".join(members) + ")", features="v2_tuned",
    cv_F1=round(best_m["f1"],4), cv_ROC_AUC=round(best_m["auc"],4), combined=round(best_c,4),
    precision=round(best_m["precision"],4), recall=round(best_m["recall"],4), notes="uniform OOF-greedy average"))

# seed robustness (3-fold, seed 7) for ensemble members only
for name in members:
    skf = StratifiedKFold(3, shuffle=True, random_state=7)
    o = np.zeros(len(X))
    for ti, vi in skf.split(X, y):
        p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", MODELS[name]())])
        p.fit(X.iloc[ti], y.iloc[ti]); o[vi] = p.predict_proba(X.iloc[vi])[:, 1]
    c, m = comb_of(o)
    print(f"SEED7 {name}: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={c:.4f}", flush=True)
    rows.append(dict(experiment="expE_seed7", model=name, features="v2_tuned",
        cv_F1=round(m["f1"],4), cv_ROC_AUC=round(m["auc"],4), combined=round(c,4),
        precision=round(m["precision"],4), recall=round(m["recall"],4), notes="seed-robustness 3fold"))

# retrain members on FULL train, predict test, uniform average
test_probas = []
for name in members:
    p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", MODELS[name]())])
    p.fit(X, y)
    test_probas.append(p.predict_proba(te[NUM + CAT])[:, 1])
    print(f"retrained {name}", flush=True)
final_proba = np.mean(test_probas, axis=0)
sub = build_submission(te_raw["ID"], final_proba)
errs = verify_submission(sub, te_raw, sample_path="SampleSubmission.csv")
assert not errs, errs
sub.to_csv("submission_improved.csv", index=False)
print("submission_improved.csv:", sub.shape, "pos_rate=%.4f" % (sub["TargetF1"] == 1).mean(), flush=True)

# artifacts
base = pd.read_csv("experiments/baseline_results.csv").rename(
    columns={"F1": "cv_F1", "AUC": "cv_ROC_AUC", "combined": "combined",
             "precision": "precision", "recall": "recall"})
for c in ["precision", "recall"]:
    base[c] = base[c].round(4)
abl = pd.DataFrame([
    dict(experiment="expA_ablation", model="rf", features="v1", cv_F1=0.7972, cv_ROC_AUC=0.8111, combined=0.8028, precision=0.8042, recall=0.7904, notes="baseline features"),
    dict(experiment="expA_ablation", model="rf", features="v1_noyear", cv_F1=0.8011, cv_ROC_AUC=0.7939, combined=0.7982, precision=0.7976, recall=0.8046, notes="drop death_year: AUC collapses -> keep year"),
    dict(experiment="expA_ablation", model="rf", features="v2", cv_F1=0.7998, cv_ROC_AUC=0.8109, combined=0.8042, precision=0.8064, recall=0.7934, notes="v2 extras help slightly"),
    dict(experiment="expA_ablation", model="rf", features="v2_noyear", cv_F1=0.8033, cv_ROC_AUC=0.7937, combined=0.7995, precision=0.8006, recall=0.8061, notes="drop death_year hurts again"),
])
full = pd.concat([base.assign(features=base.get("features", "v1_no_geo")), abl, pd.DataFrame(rows)], ignore_index=True)
full.to_csv("experiments/model_results.csv", index=False)
out = pd.DataFrame({"ID": tr_raw["ID"], "actual": y.values})
for k, v in oofs.items(): out[f"proba_{k}"] = np.round(v, 6)
out["proba_ensemble"] = np.round(ens, 6)
out["pred_ensemble"] = (ens >= 0.5).astype(int)
out.to_csv("experiments/oof_predictions.csv", index=False)
json.dump({"members": members, "combined": round(best_c, 4)}, open("experiments/final_ensemble.json", "w"))
print("DONE finalize")
