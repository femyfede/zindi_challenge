"""Time-aware CV with CORRECT aggregation: only rows that received a prediction count.
The previous 0.45 was an artifact of zero-filling unpredicted rows.
Also tests a 80/20 chronological holdout, which matches the test-time range better."""
import sys, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.data_loader import load_data, merge_climate
from src.features_advanced import add_features_advanced, compute_location_climatology, ADV_NUM, ADV_CAT
from src.encoding import target_encode_oof, target_encode_full
from src.validation import make_preprocess, fold_metrics, combined_score

tr_raw, te_raw, cl = load_data()
tr = add_features_advanced(merge_climate(tr_raw, cl))
y = tr["is_climate_sensitive"].values

NUM = ADV_NUM
CAT = ADV_CAT
X = tr[NUM + CAT]
ys = pd.to_datetime(tr["deathdate"])

def agg_masked(y_true, oof, mask):
    """Metrics over ONLY rows with a real prediction."""
    m = fold_metrics(y_true[mask], oof[mask])
    m["combined"] = combined_score(m["f1"], m["auc"])
    m["n_predicted"] = int(mask.sum())
    return m

def rolling_eval(make_fn, n_initial_years=5, n_splits=5, label=""):
    years = sorted(tr["death_year"].unique())
    oof = np.zeros(len(X)); mask = np.zeros(len(X), bool); ys = tr["deathdate"]
    idx_end = n_initial_years - 1
    for i in range(n_splits):
        if idx_end >= len(years) - 1: break
        tr_years = years[:idx_end + 1]; te_year = years[idx_end + 1]
        ti = tr.index[tr["death_year"].isin(tr_years)].values
        vi = tr.index[tr["death_year"] == te_year].values
        p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", make_fn())])
        p.fit(X.iloc[ti], y[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:, 1]; mask[vi] = True
        idx_end += 1
    m = agg_masked(y, oof, mask)
    print(f"  ROLLING {label}: n={m['n_predicted']} F1={m['f1']:.4f} AUC={m['auc']:.4f} "
          f"comb={m['combined']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f}", flush=True)
    return m

def chrono_holdout(make_fn, frac=0.2, label="", seed=42):
    """Sort by date, last frac% is the holdout. Matches the actual test date range."""
    order = np.argsort(ys.values, kind="stable")
    n_ho = int(len(X) * frac)
    vi = order[-n_ho:]; ti = order[:-n_ho]
    p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", make_fn())])
    p.fit(X.iloc[ti], y[ti]); oof = p.predict_proba(X.iloc[vi])[:, 1]
    m = fold_metrics(y[vi], oof); m["combined"] = combined_score(m["f1"], m["auc"])
    m["n_predicted"] = len(vi)
    print(f"  CHRONO80/20 {label}: n={m['n_predicted']} train_max={ys.iloc[ti].max().date()} "
          f"test_min={ys.iloc[vi].min().date()} F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={m['combined']:.4f}", flush=True)
    return m

def strat_eval(make_fn, label=""):
    oof = np.zeros(len(X))
    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", make_fn())])
        p.fit(X.iloc[ti], y[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:, 1]
    m = fold_metrics(y, oof); m["combined"] = combined_score(m["f1"], m["auc"]); m["n_predicted"] = len(X)
    print(f"  STRAT5 {label}: F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={m['combined']:.4f}", flush=True)
    return m

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
mk = {
 "xgb": lambda: XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.03, subsample=0.7,
                             colsample_bytree=0.7, reg_lambda=1.0, scale_pos_weight=1.5,
                             n_jobs=-1, random_state=42, tree_method="hist", eval_metric="logloss"),
 "rf": lambda: RandomForestClassifier(n_estimators=1200, min_samples_leaf=2, max_features="sqrt",
                                      max_depth=20, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=42),
 "lgbm": lambda: LGBMClassifier(n_estimators=1200, num_leaves=63, learning_rate=0.03,
                                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=3,
                                lambda_l2=1.5, class_weight="balanced", n_jobs=-1,
                                random_state=42, verbose=-1),
}
out = {}
for name, fn in mk.items():
    print(f"[{name}]", flush=True)
    out[name] = {"strat": strat_eval(fn, name), "chrono": chrono_holdout(fn, label=name),
                 "rolling": rolling_eval(fn, label=name)}
pd.DataFrame([{"model": k, "cv": c, "F1": round(v["f1"],4), "AUC": round(v["auc"],4),
               "combined": round(v["combined"],4), "n": v["n_predicted"]}
              for k, d in out.items() for c, v in d.items()]).to_csv(
    "experiments/cv_strategy_comparison.csv", index=False)
print("\nDONE cv_compare")
