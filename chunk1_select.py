"""Chunk 1: feature selection only (fast, 3-fold instead of 5 to save time)."""
import sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from src.data_loader import load_data, merge_climate
from src.features_advanced import add_features_advanced, compute_location_climatology, ADV_NUM, ADV_CAT
from src.encoding import target_encode_oof
from src.validation import make_preprocess, fold_metrics, combined_score

tr_raw, te_raw, cl = load_data()
mtr = merge_climate(tr_raw, cl)
clim = compute_location_climatology(mtr)
d = add_features_advanced(mtr, clim)
lo, _ = target_encode_oof(d, pd.Series(d["is_climate_sensitive"]), "location", n_splits=5, smooth=30)
d["loc_te"] = lo
y = d["is_climate_sensitive"].values
NUM = ADV_NUM + ["loc_te"]; CAT = ADV_CAT
X = d[NUM + CAT]

def mk_rf():
    return RandomForestClassifier(n_estimators=600, min_samples_leaf=2, max_features="sqrt",
                                  max_depth=20, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=42)

ti, vi = train_test_split(np.arange(len(X)), test_size=0.25, stratify=y, random_state=42)
p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_rf())])
p.fit(X.iloc[ti], y[ti])
pi = permutation_importance(p, X.iloc[vi], y[vi], n_repeats=4, random_state=42,
                            scoring="roc_auc", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=NUM + CAT).sort_values(ascending=False)
imp.to_csv("experiments/perm_importance_full.csv")
print("TOP 25 by permutation AUC importance:")
print(imp.head(25).round(5).to_string(), flush=True)
print("\nweak/negative-importance count:", int((imp <= 0).sum()), "of", len(imp), flush=True)

def strat3(num, cat):
    oof = np.zeros(len(X))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=42).split(X, y):
        q = Pipeline([("pre", make_preprocess(num, cat)), ("clf", mk_rf())])
        q.fit(X.iloc[a], y[a]); oof[b] = q.predict_proba(X.iloc[b])[:, 1]
    m = fold_metrics(y, oof)
    return m, combined_score(m["f1"], m["auc"])

full = strat3(NUM, CAT)
print(f"\nALL {len(NUM)+len(CAT)} feats : F1={full[0]['f1']:.4f} AUC={full[0]['auc']:.4f} comb={full[1]:.4f}", flush=True)
for k in [15, 25, 40]:
    sel = list(imp.head(k).index)
    ns = [c for c in sel if c in NUM]; cs = [c for c in sel if c in CAT]
    m, c = strat3(ns, cs)
    print(f"top{k:<3d} ({len(ns)}num/{len(cs)}cat): F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={c:.4f}", flush=True)
print("DONE chunk1")