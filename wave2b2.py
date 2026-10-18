"""Wave 2B-2: tuned ET only (fixed sampler bug: index into python list)."""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(11)
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from src.data_loader import load_data, merge_climate
from src.feature_engineering import add_features, BASE_CATEGORICAL, BASE_NUMERIC
from src.features_v2 import add_features_v2, V2_EXTRA
from src.validation import make_preprocess, fold_metrics, combined_score

tr_raw, te_raw, cl = load_data()
tr = add_features_v2(add_features(merge_climate(tr_raw, cl)))
y = tr["is_climate_sensitive"].values
NUM = BASE_NUMERIC + V2_EXTRA
CAT = BASE_CATEGORICAL
X = tr[NUM + CAT]
rng = np.random.RandomState(11)
MF = ["sqrt", "log2", 0.6, 1.0]
MD = [None, 16, 28]
NE = [500, 800, 1200]
best, best_kw = -1, None
for i in range(6):
    kw = dict(n_estimators=NE[rng.randint(3)], min_samples_leaf=int(rng.randint(1, 6)),
              max_features=MF[rng.randint(4)], max_depth=MD[rng.randint(3)],
              class_weight="balanced_subsample", n_jobs=-1, random_state=42)
    oo = np.zeros(len(X))
    for ti, vi in StratifiedKFold(3, shuffle=True, random_state=42).split(X, y):
        p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", ExtraTreesClassifier(**kw))])
        p.fit(X.iloc[ti], y[ti]); oo[vi] = p.predict_proba(X.iloc[vi])[:, 1]
    m = fold_metrics(y, oo); c = combined_score(m["f1"], m["auc"])
    print(f"  et iter{i}: comb={c:.4f} {kw}", flush=True)
    if c > best: best, best_kw = c, kw
print("ET best:", round(best, 4), best_kw, flush=True)
o = np.zeros(len(X))
for ti, vi in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", ExtraTreesClassifier(**best_kw))])
    p.fit(X.iloc[ti], y[ti]); o[vi] = p.predict_proba(X.iloc[vi])[:, 1]
m = fold_metrics(y, o); print(f"extratrees_tuned: F1={m['f1']:.4f} AUC={m['auc']:.4f} "
      f"comb={combined_score(m['f1'], m['auc']):.4f}", flush=True)
oof = pd.read_csv("experiments/oof_predictions.csv")
oof["proba_et_tuned"] = np.round(o, 6)
oof.to_csv("experiments/oof_predictions.csv", index=False)
print("DONE et")
