"""v2 benchmark: feature families x models, 5-fold stratified, exact competition metric."""
import numpy as np, pandas as pd, warnings, time, sys
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False
from bench import load
from features2 import build

NJ = 4
tr, te = load()
y = tr["is_climate_sensitive"].values


def make(clf, num, cat, scale=False):
    ntr = Pipeline([("imp", SimpleImputer(strategy="median"))] +
                   ([("sc", StandardScaler())] if scale else []))
    pre = ColumnTransformer([
        ("num", ntr, num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    return Pipeline([("pre", pre), ("clf", clf)])


def cv(makefn, X, seed=42, n_splits=5):
    oof = np.zeros(len(X))
    for a, b in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        p = makefn(); p.fit(X.iloc[a], y[a]); oof[b] = p.predict_proba(X.iloc[b])[:, 1]
    f1 = f1_score(y, (oof >= 0.5).astype(int), zero_division=0)
    auc = roc_auc_score(y, oof)
    return f1, auc, 0.6 * f1 + 0.4 * auc, oof


MODELS = {
 "logreg":  (lambda: LogisticRegression(max_iter=4000, C=0.3), True),
 "rf":      (lambda: RandomForestClassifier(n_estimators=350, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", n_jobs=NJ, random_state=42), False),
 "et":      (lambda: ExtraTreesClassifier(n_estimators=350, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", n_jobs=NJ, random_state=42), False),
 "lgbm":    (lambda: LGBMClassifier(n_estimators=400, num_leaves=15, learning_rate=0.04, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0, min_child_samples=20, n_jobs=NJ, random_state=42, verbose=-1), False),
 "xgb":     (lambda: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0, min_child_weight=5, tree_method="hist", n_jobs=NJ, random_state=42, eval_metric="logloss"), False),
 "hgb":     (lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=20, random_state=42), False),
 "mlp":     (lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1.0, max_iter=800, random_state=42), True),
}
if HAS_CAT:
    MODELS["cat"] = (lambda: CatBoostClassifier(iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=5,
                                                verbose=0, random_seed=42, allow_writing_files=False), False)

FSETS = {
 "demo_time":        ("demo", "time"),
 "demo_time_clim":   ("demo", "time", "clim"),
 "demo_time_clim_geo": ("demo", "time", "clim", "geo"),
 "demo_clim":        ("demo", "clim"),
}

rows = []
for fname, use in FSETS.items():
    X, num, cat = build(tr, use)
    for mname, (mf, scale) in MODELS.items():
        t0 = time.time()
        try:
            f1, auc, comb, oof = cv(lambda: make(mf(), num, cat, scale), X)
        except Exception as e:
            print("FAIL", fname, mname, repr(e)[:120], flush=True); continue
        rows.append(dict(fset=fname, model=mname, f1=f1, auc=auc, combined=comb,
                         pos=float((oof >= 0.5).mean()), nfeat=len(num) + len(cat),
                         secs=round(time.time() - t0, 1)))
        pd.DataFrame(rows).to_csv("experiments/v2_bench.csv", index=False)
        print(f"{fname:22s} {mname:7s} f1={f1:.4f} auc={auc:.4f} comb={comb:.4f} "
              f"pos={(oof>=0.5).mean():.3f} nf={len(num)+len(cat)} ({time.time()-t0:.0f}s)", flush=True)

print("\nDONE")
print(pd.DataFrame(rows).sort_values("combined", ascending=False).to_string(index=False))
