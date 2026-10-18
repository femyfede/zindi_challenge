"""TabPFN: pretrained tabular foundation model. Designed for small tabular data
(~10k rows, <=100 features) which is exactly this regime (3146 rows, ~90 features).
Evaluated under BOTH stratified 5-fold and rolling year-by-year CV, identical to
every other model tested, so the comparison is apples-to-apples.
"""
import sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from src.data_loader import load_data, merge_climate
from src.features_advanced import add_features_advanced, compute_location_climatology, ADV_NUM, ADV_CAT
from src.encoding import target_encode_oof, target_encode_full
from src.validation import fold_metrics, combined_score

tr_raw, te_raw, cl = load_data()
mtr = merge_climate(tr_raw, cl)
clim = compute_location_climatology(mtr)
d = add_features_advanced(mtr, clim)
lo, _ = target_encode_oof(d, pd.Series(d["is_climate_sensitive"]), "location", n_splits=5, smooth=30)
d["loc_te"] = lo
y = d["is_climate_sensitive"].values
NUM = ADV_NUM + ["loc_te"]

X = d[NUM].copy()
# TabPFN needs float32, no NaN/inf
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True)).astype(np.float32)
print("X:", X.shape, "| any nan:", bool(X.isna().any().any()), flush=True)

from tabpfn import TabPFNClassifier

def mk():
    return TabPFNClassifier(n_estimators=8, device="cpu", random_state=42)

def cv_strat():
    oof = np.zeros(len(X))
    for i, (ti, vi) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
        t0 = time.time()
        m = mk(); m.fit(X.iloc[ti], y[ti])
        oof[vi] = m.predict_proba(X.iloc[vi])[:, 1]
        print(f"  strat fold{i} done {time.time()-t0:.0f}s", flush=True)
    return oof

def cv_rolling(n_initial=5, n_splits=5):
    oof = np.zeros(len(X)); mask = np.zeros(len(X), bool)
    years = sorted(d["death_year"].unique()); e = n_initial - 1
    for _ in range(n_splits):
        if e >= len(years) - 1: break
        ti = d.index[d["death_year"].isin(years[:e + 1])].values
        vi = d.index[d["death_year"] == years[e + 1]].values
        m = mk(); m.fit(X.iloc[ti], y[ti])
        oof[vi] = m.predict_proba(X.iloc[vi])[:, 1]; mask[vi] = True
        e += 1
    return oof, mask

print("\n=== TabPFN stratified 5-fold ===", flush=True)
os_ = cv_strat()
ms = fold_metrics(y, os_)
print(f"TabPFN STRAT: F1={ms['f1']:.4f} AUC={ms['auc']:.4f} comb={combined_score(ms['f1'], ms['auc']):.4f} "
      f"prec={ms['precision']:.4f} rec={ms['recall']:.4f}", flush=True)

print("\n=== TabPFN rolling year-by-year ===", flush=True)
orl, mrl = cv_rolling()
mr = fold_metrics(y[mrl], orl[mrl])
print(f"TabPFN ROLL: F1={mr['f1']:.4f} AUC={mr['auc']:.4f} comb={combined_score(mr['f1'], mr['auc']):.4f} "
      f"(n={mrl.sum()})", flush=True)

# reference: tuned RF on identical features
oof_rf = np.zeros(len(X))
for ti, vi in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    from sklearn.pipeline import Pipeline
    from src.validation import make_preprocess
    p = Pipeline([("pre", make_preprocess(NUM, ADV_CAT)),
                  ("clf", RandomForestClassifier(n_estimators=1200, min_samples_leaf=2,
                       max_features="sqrt", max_depth=20, class_weight="balanced_subsample",
                       n_jobs=-1, random_state=42))])
    p.fit(d.iloc[ti][NUM + ADV_CAT], y[ti])
    oof_rf[vi] = p.predict_proba(d.iloc[vi][NUM + ADV_CAT])[:, 1]
mr2 = fold_metrics(y, oof_rf)
print(f"\nREF RF STRAT: F1={mr2['f1']:.4f} AUC={mr2['auc']:.4f} comb={combined_score(mr2['f1'], mr2['auc']):.4f}", flush=True)

# blend
for w in [0.3, 0.5, 0.7]:
    b = w * os_ + (1 - w) * oof_rf
    mb = fold_metrics(y, b)
    print(f"BLEND tabpfn={w:.1f}/rf={1-w:.1f}: F1={mb['f1']:.4f} AUC={mb['auc']:.4f} "
          f"comb={combined_score(mb['f1'], mb['auc']):.4f}", flush=True)

np.save("experiments/tabpfn_oof.npy", os_)
pd.DataFrame({"model": ["tabpfn", "rf_ref"], "cv": ["strat5", "strat5"],
              "F1": [round(ms["f1"], 4), round(mr2["f1"], 4)],
              "AUC": [round(ms["auc"], 4), round(mr2["auc"], 4)],
              "comb": [round(combined_score(ms["f1"], ms["auc"]), 4),
                       round(combined_score(mr2["f1"], mr2["auc"]), 4)]}).to_csv(
    "experiments/tabpfn_result.csv", index=False)
print("DONE tabpfn")