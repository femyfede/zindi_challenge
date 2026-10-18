"""Model benchmark across feature sets, repeated 5-fold CV, exact competition metric."""
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from bench import load, repeated_cv, show

tr, te = load()
y = tr["is_climate_sensitive"]
dt = pd.to_datetime(tr["deathdate"])


def featset(name):
    d = tr.drop(columns=["is_climate_sensitive", "ID"]).copy()
    d["death_year"] = dt.dt.year
    d["death_month"] = dt.dt.month
    d["doy"] = dt.dt.dayofyear
    d["age_log"] = np.log1p(d["age"])
    d["temp_range"] = d["max_temperature"] - d["min_temperature"]
    d["t_anom"] = d["tavg_7d"] - d["tavg_90d"]
    d["rain_ratio"] = d["rain_sum_7d"] / (d["rain_sum_30d"] + 1)
    d["rain90_3"] = d["rain_sum_90d"] / 3.0
    d["rain_30_90_diff"] = d["rain_sum_30d"] - d["rain_sum_90d"] / 3.0
    d["ndvi_chg"] = d["ndvi_30d"] - d["ndvi_90d"]
    d["dry_days"] = 30 - d["rain_days_30d"]
    d["is_under5"] = (d["age"] <= 5).astype(int)
    d["is_elderly"] = (d["age"] >= 60).astype(int)
    d["is_infant"] = (d["age"] <= 1).astype(int)
    d["year_x_age"] = d["death_year"] * d["age"]
    cat = ["gender", "zone"]
    drop = ["location", "deathdate", "hot_days_30d"]
    num = [c for c in d.columns if c not in cat + drop]
    if name == "raw_nogeo":
        drop += ["latitude", "longitude", "elevation", "slope"]
        num = [c for c in d.columns if c not in cat + drop]
    if name == "raw_geo":
        pass
    X = d[num + cat]
    return X, num, cat


def mkpipe(clf, num, cat, scale=False, qt=False):
    if qt:
        ntr = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("qt", QuantileTransformer(output_distribution="normal", n_quantiles=200))])
    elif scale:
        ntr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    else:
        ntr = SimpleImputer(strategy="median")
    pre = ColumnTransformer([
        ("num", ntr, num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    return Pipeline([("pre", pre), ("clf", clf)])


MODELS = {
 "logreg":      lambda: LogisticRegression(max_iter=3000, C=1.0, class_weight=None),
 "logreg_bal":  lambda: LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
 "rf800":       lambda: RandomForestClassifier(n_estimators=800, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=42),
 "rf800_plain": lambda: RandomForestClassifier(n_estimators=800, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=42),
 "et800":       lambda: ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=42),
 "xgb":         lambda: XGBClassifier(n_estimators=700, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0, min_child_weight=3, n_jobs=-1, random_state=42, eval_metric="logloss"),
 "xgb_bal":     lambda: XGBClassifier(n_estimators=700, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0, min_child_weight=3, n_jobs=-1, random_state=42, eval_metric="logloss", scale_pos_weight=0.544),
 "lgbm":        lambda: LGBMClassifier(n_estimators=700, num_leaves=15, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0, min_child_samples=20, n_jobs=-1, random_state=42, verbose=-1),
 "hgb":         lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=20, random_state=42),
}

rows = []
for fsname in ["raw_nogeo", "raw_geo"]:
    X, num, cat = featset(fsname)
    for mname, mf in MODELS.items():
        t0 = time.time()
        fp = lambda a, b, c, mf=mf, num=num, cat=cat, mname=mname: mkpipe(
            mf(), num, cat,
            scale=mname.startswith("logreg"),
            qt=mname.startswith("logreg")).fit(a, b).predict_proba(c)[:, 1]
        m, per, oofs = repeated_cv(fp, X, y, repeats=3)
        m.update(model=mname, fset=fsname, secs=round(time.time() - t0, 1))
        rows.append(m)
        print(f"{fsname:10s} {mname:12s} f1={m['f1']:.4f} auc={m['auc']:.4f} comb={m['combined']:.4f} +-{m['comb_sd']:.4f}  ({m['secs']}s)")

res = pd.DataFrame(rows)
res.to_csv("experiments/bench_models.csv", index=False)
show(res.drop(columns=["fset"])[["model", "f1", "auc", "combined", "f1_sd", "auc_sd", "comb_sd"]], "BENCH (3x repeated 5-fold)")
