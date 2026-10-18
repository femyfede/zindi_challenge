"""Phase-1 screening: single 5-fold CV, many model/feature combos. Writes CSV incrementally."""
import numpy as np, pandas as pd, warnings, time, sys, os
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from bench import load

OUT = "experiments/phase1.csv"
tr, te = load()
y = tr["is_climate_sensitive"].values
dt = pd.to_datetime(tr["deathdate"])
NJ = 4


def build(fsname):
    d = tr.drop(columns=["is_climate_sensitive", "ID"]).copy()
    d["death_year"] = dt.dt.year; d["death_month"] = dt.dt.month; d["doy"] = dt.dt.dayofyear
    d["age_log"] = np.log1p(d["age"])
    d["temp_range"] = d["max_temperature"] - d["min_temperature"]
    d["t_anom"] = d["tavg_7d"] - d["tavg_90d"]
    d["rain_ratio"] = d["rain_sum_7d"] / (d["rain_sum_30d"] + 1)
    d["rain_30_90_diff"] = d["rain_sum_30d"] - d["rain_sum_90d"] / 3.0
    d["ndvi_chg"] = d["ndvi_30d"] - d["ndvi_90d"]
    d["dry_days"] = 30 - d["rain_days_30d"]
    d["is_under5"] = (d["age"] <= 5).astype(int); d["is_elderly"] = (d["age"] >= 60).astype(int)
    d["is_infant"] = (d["age"] <= 1).astype(int); d["is_child"] = ((d["age"] > 5) & (d["age"] <= 18)).astype(int)
    d["month_sin"] = np.sin(2 * np.pi * d["death_month"] / 12); d["month_cos"] = np.cos(2 * np.pi * d["death_month"] / 12)
    d["is_female"] = (d["gender"] == "Female").astype(int); d["is_rural"] = (d["zone"] == "Rural").astype(int)
    cat = ["gender", "zone"]
    always_drop = ["location", "deathdate", "hot_days_30d"]
    drop = list(always_drop)
    if fsname in ("nogeo", "noage_year"):
        drop += ["latitude", "longitude", "elevation", "slope"]
    if fsname == "noage_year":
        drop += ["age", "age_log", "is_under5", "is_elderly", "is_infant", "is_child", "death_year"]
    num = [c for c in d.columns if c not in cat + drop]
    if fsname == "lean":
        num = ["age", "death_year", "death_month", "is_female", "is_rural"]
    return d[num + cat], num, cat


MODELS = {
 "logreg":     (lambda: LogisticRegression(max_iter=3000, C=1.0), True),
 "rf":         (lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", n_jobs=NJ, random_state=42), False),
 "rf_depth8":  (lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=4, max_features=0.4, max_depth=8, class_weight="balanced_subsample", n_jobs=NJ, random_state=42), False),
 "rf_nobal":   (lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt", n_jobs=NJ, random_state=42), False),
 "et":         (lambda: ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", n_jobs=NJ, random_state=42), False),
 "xgb":        (lambda: XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0, min_child_weight=3, tree_method="hist", n_jobs=NJ, random_state=42, eval_metric="logloss"), False),
 "xgb_d3":     (lambda: XGBClassifier(n_estimators=600, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, min_child_weight=5, tree_method="hist", n_jobs=NJ, random_state=42, eval_metric="logloss"), False),
 "lgbm":       (lambda: LGBMClassifier(n_estimators=500, num_leaves=15, learning_rate=0.04, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0, min_child_samples=20, n_jobs=NJ, random_state=42, verbose=-1), False),
 "lgbm_l7":    (lambda: LGBMClassifier(n_estimators=700, num_leaves=7, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, min_child_samples=30, n_jobs=NJ, random_state=42, verbose=-1), False),
 "hgb":        (lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=20, random_state=42), False),
}


def run_cv(clf, X, num, cat, scale, seed=42, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    for a, b in skf.split(X, y):
        ntr = Pipeline([("imp", SimpleImputer(strategy="median"))] +
                       ([("sc", StandardScaler())] if scale else []))
        pre = ColumnTransformer([
            ("num", ntr, num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
        Pipeline([("pre", pre), ("clf", clf())]).fit(X.iloc[a], y[a]).predict_proba  # noqa
        p = Pipeline([("pre", pre), ("clf", clf())])
        p.fit(X.iloc[a], y[a]); oof[b] = p.predict_proba(X.iloc[b])[:, 1]
    f1 = f1_score(y, (oof >= 0.5).astype(int), zero_division=0)
    auc = roc_auc_score(y, oof)
    return f1, auc, 0.6 * f1 + 0.4 * auc, oof


rows = []
for fsname in ["full", "nogeo", "lean"]:
    X, num, cat = build(fsname)
    for mname, (mf, scale) in MODELS.items():
        if fsname == "lean" and mname not in ("logreg", "rf", "xgb", "lgbm", "hgb"):
            continue
        t0 = time.time()
        try:
            f1, auc, comb, oof = run_cv(mf, X, num, cat, scale)
        except Exception as e:
            print("FAIL", fsname, mname, e, flush=True); continue
        rows.append(dict(fset=fsname, model=mname, f1=f1, auc=auc, combined=comb,
                         posrate=float((oof >= 0.5).mean()), secs=round(time.time() - t0, 1)))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"{fsname:8s} {mname:10s} f1={f1:.4f} auc={auc:.4f} comb={comb:.4f} pos={((oof>=0.5).mean()):.3f} ({time.time()-t0:.0f}s)", flush=True)

print("\nDONE")
print(pd.DataFrame(rows).sort_values("combined", ascending=False).to_string(index=False))
