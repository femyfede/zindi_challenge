"""Manual randomized search (avoids make_scorer quirks): 3-fold stratified OOF,
combined = 0.6*F1(0.5)+0.4*AUC. Feature set v2 (winner of expA: 0.8042)."""
import os, sys, time, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(42)
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from src.data_loader import load_data, merge_climate
from src.feature_engineering import add_features, BASE_CATEGORICAL
from src.features_v2 import add_features_v2, V2_EXTRA
from src.validation import make_preprocess, fold_metrics, combined_score
from src.feature_engineering import BASE_NUMERIC

os.makedirs("experiments", exist_ok=True)
tr_raw, te_raw, cl = load_data()
tr = add_features_v2(add_features(merge_climate(tr_raw, cl)))
y = tr["is_climate_sensitive"]
NUM = BASE_NUMERIC + V2_EXTRA
CAT = BASE_CATEGORICAL
X = tr[NUM + CAT]
rng = np.random.RandomState(42)

def cv3(make_fn):
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for ti, vi in skf.split(X, y):
        p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", make_fn())])
        p.fit(X.iloc[ti], y.iloc[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:, 1]
    m = fold_metrics(y.values, oof); m["combined"] = combined_score(m["f1"], m["auc"])
    return m

def sample_rf():
    from sklearn.ensemble import RandomForestClassifier
    kw = dict(n_estimators=int(rng.choice([400, 600, 800, 1000])),
              min_samples_leaf=int(rng.randint(1, 7)),
              max_features=rng.choice(["sqrt", "log2", 0.5, 1.0]),
              max_depth=rng.choice([None, 14, 20, 30]),
              class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    return lambda: RandomForestClassifier(**kw), kw

def sample_xgb():
    from xgboost import XGBClassifier
    kw = dict(n_estimators=int(rng.choice([400, 700, 1000])), max_depth=int(rng.randint(3, 9)),
              learning_rate=float(rng.uniform(0.02, 0.12)), subsample=float(rng.uniform(0.6, 1.0)),
              colsample_bytree=float(rng.uniform(0.6, 1.0)), reg_lambda=float(rng.uniform(0.5, 8)),
              scale_pos_weight=float(rng.choice([0.54, 1.0, 1.5])),
              n_jobs=-1, random_state=42, tree_method="hist", eval_metric="logloss")
    return lambda: XGBClassifier(**kw), kw

def sample_lgbm():
    from lightgbm import LGBMClassifier
    kw = dict(n_estimators=int(rng.choice([600, 1000, 1500])), num_leaves=int(rng.randint(31, 127)),
              learning_rate=float(rng.uniform(0.02, 0.09)), feature_fraction=float(rng.uniform(0.6, 1.0)),
              bagging_fraction=float(rng.uniform(0.6, 1.0)), bagging_freq=3,
              lambda_l2=float(rng.uniform(0.5, 8)), class_weight="balanced",
              n_jobs=-1, random_state=42, verbose=-1)
    return lambda: LGBMClassifier(**kw), kw

def sample_cat():
    from catboost import CatBoostClassifier
    kw = dict(iterations=int(rng.choice([500, 800, 1100])), depth=int(rng.randint(4, 9)),
              learning_rate=float(rng.uniform(0.02, 0.12)), l2_leaf_reg=float(rng.uniform(1, 9)),
              auto_class_weights="Balanced", random_seed=42, verbose=False, allow_writing_files=False)
    return lambda: CatBoostClassifier(**kw), kw

def sample_hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    kw = dict(max_iter=int(rng.choice([250, 400, 600])), learning_rate=float(rng.uniform(0.03, 0.12)),
              max_leaf_nodes=int(rng.randint(15, 63)), l2_regularization=float(rng.uniform(0.1, 8)),
              class_weight="balanced", random_state=42)
    return lambda: HistGradientBoostingClassifier(**kw), kw

SAMPLERS = {"rf": (sample_rf, 8), "xgboost": (sample_xgb, 12), "lightgbm": (sample_lgbm, 12),
            "catboost": (sample_cat, 10), "histgb": (sample_hgb, 8)}
summary = {}
for name, (samp, n) in SAMPLERS.items():
    t0 = time.time(); best, best_kw = -1, None
    for i in range(n):
        fn, kw = samp()
        try: m = cv3(fn); c = m["combined"]
        except Exception as e: print(f"  {name} iter{i} FAIL {e}", flush=True); continue
        print(f"  {name} iter{i}: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={c:.4f}", flush=True)
        if c > best: best, best_kw, best_m = c, kw, m
    summary[name] = {"combined": round(best, 4), "F1": round(best_m["f1"], 4),
                     "AUC": round(best_m["auc"], 4), "params": best_kw}
    print(f"TUNE {name}: best comb={best:.4f} {best_kw} ({time.time()-t0:.0f}s)", flush=True)

json.dump(summary, open("experiments/tuning_summary.json", "w"), indent=1,
          default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else str(o))
print("DONE tune")
