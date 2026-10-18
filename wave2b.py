"""Wave 2B: location in-fold encoding, age-band mixture, native CatBoost, tuned ET."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(42)
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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
oof = pd.read_csv("experiments/oof_predictions.csv")

def rep(p, tag):
    m = fold_metrics(y, p); c = combined_score(m["f1"], m["auc"])
    print(f"{tag}: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={c:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f}", flush=True)
    return c

def mk_xgb(s=42, n=400):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=n, max_depth=3, learning_rate=0.07248, subsample=0.7599,
                         colsample_bytree=0.6187, reg_lambda=7.803, scale_pos_weight=1.5,
                         n_jobs=-1, random_state=s, tree_method="hist", eval_metric="logloss")
def mk_rf(s=42):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, max_features="log2",
                                  max_depth=None, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=s)

# ---- 1. location in-fold target encoding (+frequency), RF ----
loc = tr_raw["location"].values
def loc_encode_oof(n_splits=5, seed=42, smooth=30):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    te_mean = np.zeros(len(X)); freq = np.zeros(len(X))
    glob = y.mean()
    fmap = pd.Series(loc).value_counts(normalize=True).to_dict()
    for ti, vi in skf.split(X, y):
        m = pd.DataFrame({"loc": loc[ti], "y": y[ti]}).groupby("loc")["y"].agg(["mean", "count"])
        enc = ((m["mean"] * m["count"] + glob * smooth) / (m["count"] + smooth)).to_dict()
        te_mean[vi] = pd.Series(loc[vi]).map(enc).fillna(glob).values
        freq[vi] = pd.Series(loc[vi]).map(fmap).fillna(0).values
    return te_mean, freq
loc_te, loc_fr = loc_encode_oof()
Xloc = X.copy(); Xloc["loc_te"] = loc_te; Xloc["loc_freq"] = loc_fr
NUMLOC = NUM + ["loc_te", "loc_freq"]
o = np.zeros(len(X))
for ti, vi in StratifiedKFold(5, shuffle=True, random_state=42).split(Xloc, y):
    p = Pipeline([("pre", make_preprocess(NUMLOC, CAT)), ("clf", mk_rf())])
    p.fit(Xloc.iloc[ti], y[ti]); o[vi] = p.predict_proba(Xloc.iloc[vi])[:, 1]
rep(o, "rf_locTE_strat")
oof["proba_rf_locTE"] = np.round(o, 6)
# grouped honesty check
g = np.zeros(len(X))
for ti, vi in GroupKFold(5).split(Xloc, y, tr_raw["location"]):
    # recompute encoding inside group fold to be honest
    trloc, teloc = loc[ti], loc[vi]
    m = pd.DataFrame({"loc": trloc, "y": y[ti]}).groupby("loc")["y"].agg(["mean", "count"])
    enc = ((m["mean"] * m["count"] + y[ti].mean() * 30) / (m["count"] + 30)).to_dict()
    Xtr = Xloc.iloc[ti].copy(); Xva = Xloc.iloc[vi].copy()
    Xva["loc_te"] = pd.Series(teloc).map(enc).fillna(y[ti].mean()).values
    p = Pipeline([("pre", make_preprocess(NUMLOC, CAT)), ("clf", mk_rf())])
    p.fit(Xtr, y[ti]); g[vi] = p.predict_proba(Xva)[:, 1]
rep(g, "rf_locTE_grouped")

# ---- 2. age-band mixture of XGB experts (bands: <5 / 5-60 / 60+) ----
bands = np.where(tr["age"].values < 5, 0, np.where(tr["age"].values >= 60, 2, 1))
o = np.zeros(len(X))
for ti, vi in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    for b in [0, 1, 2]:
        tri = ti[bands[ti] == b]; vai = vi[bands[vi] == b]
        if len(vai) == 0: continue
        p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_xgb(n=300))])
        if len(tri) > 20: p.fit(X.iloc[tri], y[tri])
        else:
            p.fit(X.iloc[ti], y[ti])
        o[vai] = p.predict_proba(X.iloc[vai])[:, 1]
rep(o, "xgb_agebands")
oof["proba_xgb_agebands"] = np.round(o, 6)

# ---- 3. native CatBoost (categorical, no one-hot) ----
from catboost import CatBoostClassifier, Pool
cat_cols = CAT
num_imp = SimpleImputer(strategy="median")
Xn = pd.DataFrame(num_imp.fit_transform(X[NUM]), columns=NUM, index=X.index)
Xcat = pd.concat([Xn, X[CAT].astype(str)], axis=1)
cat_idx = [Xcat.columns.get_loc(c) for c in CAT]
o = np.zeros(len(X)); t0 = time.time()
for ti, vi in StratifiedKFold(5, shuffle=True, random_state=42).split(Xcat, y):
    m = CatBoostClassifier(iterations=1100, depth=6, learning_rate=0.04684, l2_leaf_reg=4.999,
                           auto_class_weights="Balanced", random_seed=42,
                           verbose=False, allow_writing_files=False)
    m.fit(Xcat.iloc[ti], y[ti], cat_features=cat_idx); o[vi] = m.predict_proba(Xcat.iloc[vi])[:, 1]
print(f"(cat-native {time.time()-t0:.0f}s)", flush=True)
rep(o, "catboost_native")
oof["proba_cat_native"] = np.round(o, 6)

# ---- 4. tuned ET quick search (6 iters, 3-fold) then 5-fold ----
from sklearn.ensemble import ExtraTreesClassifier
rng = np.random.RandomState(11)
best, best_kw = -1, None
for i in range(6):
    kw = dict(n_estimators=int(rng.choice([500, 800, 1200])), min_samples_leaf=int(rng.randint(1, 6)),
              max_features=rng.choice(["sqrt", "log2", 0.6]), max_depth=rng.choice([None, 16, 28]),
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
rep(o, "extratrees_tuned")
oof["proba_et_tuned"] = np.round(o, 6)

oof.to_csv("experiments/oof_predictions.csv", index=False)
print("DONE wave2B")
