"""Experiment runner: feature ablations -> hyperparameter search -> OOF ensemble.
All decisions on stratified 5-fold OOF with combined = 0.6*F1(0.5) + 0.4*AUC.
Saves experiments/model_results.csv + experiments/oof_predictions.csv."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(42)
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform, loguniform

from src.data_loader import load_data, merge_climate
from src.feature_engineering import (add_features, BASE_NUMERIC, BASE_CATEGORICAL,
                                     NOCLIMATE_NUM, NOCLIMATE_CAT)
from src.features_v2 import add_features_v2, V2_EXTRA
from src.validation import make_preprocess, fold_metrics, combined_score
from src.models import MODEL_FNS

os.makedirs("experiments", exist_ok=True)
tr_raw, te_raw, cl = load_data()
tr = add_features(merge_climate(tr_raw, cl))
tr = add_features_v2(tr)
y = tr["is_climate_sensitive"]
V1N, V1C = BASE_NUMERIC, BASE_CATEGORICAL
V2N = BASE_NUMERIC + V2_EXTRA
V1N_NY = [c for c in BASE_NUMERIC if c != "death_year"]
V2N_NY = [c for c in V2N if c != "death_year"]

def combined_proba_scorer(y_true, y_proba):
    f1 = f1_score(y_true, (y_proba >= 0.5).astype(int), zero_division=0)
    try: auc = roc_auc_score(y_true, y_proba)
    except Exception: auc = 0.5
    return 0.6 * f1 + 0.4 * auc
SCORER = make_scorer(combined_proba_scorer, needs_proba=True)

def oof_eval(name, model_fn, num, cat, seed=42):
    from sklearn.model_selection import StratifiedKFold
    X = tr[num + cat]
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X)); foldm = []
    for ti, vi in skf.split(X, y):
        p = Pipeline([("pre", make_preprocess(num, cat, scale_numeric=(name == "logreg"))),
                      ("clf", model_fn())])
        p.fit(X.iloc[ti], y.iloc[ti]); pr = p.predict_proba(X.iloc[vi])[:, 1]
        oof[vi] = pr
        m = fold_metrics(y.iloc[vi].values, pr); m["combined"] = combined_score(m["f1"], m["auc"])
        foldm.append(m)
    ov = fold_metrics(y.values, oof); ov["combined"] = combined_score(ov["f1"], ov["auc"])
    return ov, oof, foldm

results, oofs = [], {}
def log(exp, model, feats, ov, notes=""):
    results.append(dict(experiment=exp, model=model, features=feats,
        cv_F1=round(ov["f1"],4), cv_ROC_AUC=round(ov["auc"],4),
        combined=round(ov["combined"],4), precision=round(ov["precision"],4),
        recall=round(ov["recall"],4), notes=notes))
    print(f"{exp} {model} {feats}: F1={ov['f1']:.4f} AUC={ov['auc']:.4f} "
          f"comb={ov['combined']:.4f} prec={ov['precision']:.4f} rec={ov['recall']:.4f}", flush=True)

# ---- A. feature ablations with RF (fast, stable) ----
from src.models import make_rf
for tag, num in [("v1", V1N), ("v1_noyear", V1N_NY), ("v2", V2N), ("v2_noyear", V2N_NY)]:
    ov, oof, _ = oof_eval("rf", make_rf, num, V1C)
    oofs[f"rf_{tag}"] = oof
    log("expA_ablation", "rf", tag, ov)

# ---- B. hyperparameter search (3-fold, combined scorer) on v2_noyear or best ----
best_tag = "v2"  # placeholder updated below from results
# decide after A: pick tag with best combined
# (computed after loop; rerun selection here programmatically)
tmp = pd.DataFrame(results).query("experiment=='expA_ablation'").sort_values("combined", ascending=False)
BEST_TAG = tmp.iloc[0]["features"]
BEST_NUM = {"v1": V1N, "v1_noyear": V1N_NY, "v2": V2N, "v2_noyear": V2N_NY}[BEST_TAG]
print("BEST feature set:", BEST_TAG, flush=True)

def catboost_fn(**kw):
    from catboost import CatBoostClassifier
    d = dict(iterations=800, depth=6, learning_rate=0.04, l2_leaf_reg=3.0,
             auto_class_weights="Balanced", random_seed=42, verbose=False,
             allow_writing_files=False)
    d.update(kw); return CatBoostClassifier(**d)

SEARCH = {
 "rf": (lambda **k: __import__("src.models", fromlist=["make_rf"]).make_rf(),
        {"clf__n_estimators": randint(300, 1200), "clf__min_samples_leaf": randint(1, 8),
         "clf__max_features": ["sqrt", "log2", 0.5, 1.0], "clf__max_depth": [None, 12, 20, 30]}),
 "xgboost": (MODEL_FNS["xgboost"],
        {"clf__n_estimators": randint(400, 1200), "clf__max_depth": randint(3, 9),
         "clf__learning_rate": loguniform(0.02, 0.15), "clf__subsample": uniform(0.6, 0.4),
         "clf__colsample_bytree": uniform(0.6, 0.4), "clf__reg_lambda": loguniform(0.5, 10),
         "clf__scale_pos_weight": [0.54, 1.0, 1.5]}),
 "lightgbm": (MODEL_FNS["lightgbm"],
        {"clf__n_estimators": randint(500, 1500), "clf__num_leaves": randint(31, 127),
         "clf__learning_rate": loguniform(0.02, 0.1), "clf__feature_fraction": uniform(0.6, 0.4),
         "clf__bagging_fraction": uniform(0.6, 0.4), "clf__lambda_l2": loguniform(0.5, 10)}),
 "catboost": (lambda: catboost_fn(),
        {"clf__iterations": randint(400, 1200), "clf__depth": randint(4, 9),
         "clf__learning_rate": loguniform(0.02, 0.15), "clf__l2_leaf_reg": loguniform(1, 10)}),
 "histgb": (MODEL_FNS["histgb"],
        {"clf__max_iter": randint(200, 700), "clf__learning_rate": loguniform(0.03, 0.15),
         "clf__max_leaf_nodes": randint(15, 63), "clf__l2_regularization": loguniform(0.1, 10)}),
}
tuned_fns, N_ITER = {}, 15
for name, (fn, space) in SEARCH.items():
    t0 = time.time()
    pipe = Pipeline([("pre", make_preprocess(BEST_NUM, V1C, scale_numeric=False)),
                     ("clf", fn())])
    rs = RandomizedSearchCV(pipe, space, n_iter=N_ITER, scoring=SCORER,
                            cv=3, random_state=42, n_jobs=1, verbose=0)
    rs.fit(tr[BEST_NUM + V1C], y)
    bp = {k.replace("clf__", ""): (int(v) if isinstance(v, (np.integer,)) else v)
          for k, v in rs.best_params_.items()}
    print(f"TUNE {name}: best3fold={rs.best_score_:.4f} params={bp} ({time.time()-t0:.0f}s)", flush=True)
    results.append(dict(experiment="expB_tune3fold", model=name, features=BEST_TAG,
        cv_F1=np.nan, cv_ROC_AUC=np.nan, combined=round(rs.best_score_, 4),
        precision=np.nan, recall=np.nan, notes=str(bp)))
    tuned_fns[name] = (fn, bp)

# ---- C. 5-fold OOF of tuned models ----
import pickle
for name, (fn, bp) in tuned_fns.items():
    def mk(fn=fn, bp=bp):
        m = fn()
        for k, v in bp.items():
            try: m.set_params(**{k: v})
            except Exception: pass
        return m
    mk.__name__ = "tuned"
    ov, oof, folds = oof_eval("tuned", mk, BEST_NUM, V1C)
    oofs[f"tuned_{name}"] = oof
    f1s = [f["f1"] for f in folds]; aucs = [f["auc"] for f in folds]
    log("expC_tuned5fold", name, BEST_TAG + "_tuned", ov,
        notes=f"F1std={np.std(f1s):.4f} AUCstd={np.std(aucs):.4f} params={bp}")

# ---- D. greedy forward ensemble on OOF ----
cands = {k: v for k, v in oofs.items() if k.startswith("tuned_") or k in ("rf_v1",)}
names = list(cands.keys())
# score helper
def comb_of(p):
    m = fold_metrics(y.values, p); return combined_score(m["f1"], m["auc"]), m
order = sorted(names, key=lambda n: comb_of(cands[n])[0], reverse=True)
print("OOF ranking:", [(n, round(comb_of(cands[n])[0], 4)) for n in order], flush=True)
ens, ens_names = cands[order[0]].copy(), [order[0]]
best_c, _ = comb_of(ens)
for n in order[1:]:
    trial = (ens * len(ens_names) + cands[n]) / (len(ens_names) + 1)
    c, _ = comb_of(trial)
    if c > best_c + 1e-4:
        ens, ens_names, best_c = trial, ens_names + [n], c
        print(f"  + {n} -> {c:.4f}", flush=True)
oofs["ensemble_greedy"] = ens
c, m = comb_of(ens)
log("expD_ensemble", "+".join(ens_names), BEST_TAG, m, notes=f"uniform average of {ens_names}")

# ---- save artifacts ----
pd.DataFrame(results).to_csv("experiments/model_results.csv", index=False)
out = pd.DataFrame({"ID": tr_raw["ID"], "y": y.values})
for k, v in oofs.items(): out[f"proba_{k}"] = v
out["pred_ensemble"] = (out["proba_ensemble_greedy"] >= 0.5).astype(int)
out.to_csv("experiments/oof_predictions.csv", index=False)
pickle.dump({"best_tag": BEST_TAG, "tuned": {k: v[1] for k, v in tuned_fns.items()},
             "ensemble_members": ens_names}, open("experiments/tuning_summary.pkl", "wb"))
print("BEST_TAG:", BEST_TAG, "ENSEMBLE:", ens_names, flush=True)
print("DONE optimize")
