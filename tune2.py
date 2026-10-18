"""Tune remaining models: catboost + histgb (6 iters each, 3-fold OOF combined)."""
import os, sys, time, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(7)
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from src.data_loader import load_data, merge_climate
from src.feature_engineering import add_features, BASE_CATEGORICAL, BASE_NUMERIC
from src.features_v2 import add_features_v2, V2_EXTRA
from src.validation import make_preprocess, fold_metrics, combined_score

os.makedirs("experiments", exist_ok=True)
tr_raw, te_raw, cl = load_data()
tr = add_features_v2(add_features(merge_climate(tr_raw, cl)))
y = tr["is_climate_sensitive"]
NUM = BASE_NUMERIC + V2_EXTRA
CAT = BASE_CATEGORICAL
X = tr[NUM + CAT]
rng = np.random.RandomState(7)

def cv3(make_fn):
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for ti, vi in skf.split(X, y):
        p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", make_fn())])
        p.fit(X.iloc[ti], y.iloc[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:, 1]
    m = fold_metrics(y.values, oof); m["combined"] = combined_score(m["f1"], m["auc"])
    return m

summary = {}
# catboost 6 iters
best, best_kw, best_m = -1, None, None
for i in range(6):
    from catboost import CatBoostClassifier
    kw = dict(iterations=int(rng.choice([500, 800, 1100])), depth=int(rng.randint(4, 9)),
              learning_rate=float(rng.uniform(0.02, 0.12)), l2_leaf_reg=float(rng.uniform(1, 9)),
              auto_class_weights="Balanced", random_seed=42, verbose=False, allow_writing_files=False)
    t0=time.time()
    m = cv3(lambda kw=kw: CatBoostClassifier(**kw)); c = m["combined"]
    print(f"catboost iter{i}: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={c:.4f} ({time.time()-t0:.0f}s) {kw}", flush=True)
    if c > best: best, best_kw, best_m = c, kw, m
summary["catboost"] = {"combined": round(best,4), "F1": round(best_m["f1"],4),
                       "AUC": round(best_m["auc"],4), "params": best_kw}
# histgb 6 iters
best, best_kw, best_m = -1, None, None
for i in range(6):
    from sklearn.ensemble import HistGradientBoostingClassifier
    kw = dict(max_iter=int(rng.choice([250, 400, 600])), learning_rate=float(rng.uniform(0.03, 0.12)),
              max_leaf_nodes=int(rng.randint(15, 63)), l2_regularization=float(rng.uniform(0.1, 8)),
              class_weight="balanced", random_state=42)
    t0=time.time()
    m = cv3(lambda kw=kw: HistGradientBoostingClassifier(**kw)); c = m["combined"]
    print(f"histgb iter{i}: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={c:.4f} ({time.time()-t0:.0f}s) {kw}", flush=True)
    if c > best: best, best_kw, best_m = c, kw, m
summary["histgb"] = {"combined": round(best,4), "F1": round(best_m["f1"],4),
                     "AUC": round(best_m["auc"],4), "params": best_kw}
json.dump(summary, open("experiments/tuning_summary2.json", "w"), indent=1,
          default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else str(o))
print("DONE tune2")
