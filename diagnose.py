"""Step 1-2: freeze baseline + diagnose shift + OOF error analysis (no test labels used)."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(42)
os.makedirs("experiments", exist_ok=True)

# ---- freeze baseline (recorded from run_solution.py stratified 5-fold OOF) ----
base = pd.DataFrame([
 dict(experiment="baseline", model="rf",         features="v1_no_geo", cv="strat5", F1=0.7972, AUC=0.8111, combined=0.8028, precision=0.8042, recall=0.7904, notes="final model, 400 trees leaf2 balanced_subsample"),
 dict(experiment="baseline", model="extratrees", features="v1_no_geo", cv="strat5", F1=0.7893, AUC=0.8048, combined=0.7955, precision=0.8131, recall=0.7670, notes="500 trees leaf2"),
 dict(experiment="baseline", model="xgboost",   features="v1_no_geo", cv="strat5", F1=0.7920, AUC=0.7986, combined=0.7946, precision=0.7825, recall=0.8017, notes="600/d6/lr.04, NO imbalance handling"),
 dict(experiment="baseline", model="lightgbm",  features="v1_no_geo", cv="strat5", F1=0.7905, AUC=0.7970, combined=0.7931, precision=0.8026, recall=0.7787, notes="800/63 leaves/lr.03 balanced"),
 dict(experiment="baseline", model="histgb",    features="v1_no_geo", cv="strat5", F1=0.7806, AUC=0.7961, combined=0.7868, precision=0.8010, recall=0.7611, notes=""),
 dict(experiment="baseline", model="logreg",    features="v1_no_geo", cv="strat5", F1=0.7680, AUC=0.8036, combined=0.7822, precision=0.8319, recall=0.7132, notes="balanced, scaled"),
])
base.to_csv("experiments/baseline_results.csv", index=False)
print("baseline frozen")

# ---- shift diagnosis ----
import sys; sys.path.insert(0, ".")
from src.data_loader import load_data, merge_climate
from src.feature_engineering import add_features
tr_raw, te_raw, cl = load_data()
tr = add_features(merge_climate(tr_raw, cl)); te = add_features(merge_climate(te_raw, cl))
num = ["age","avg_temperature","max_temperature","min_temperature","precipitation","rain_sum_7d",
       "rain_sum_30d","rain_sum_90d","rain_days_30d","max_daily_rain_30d","tavg_7d","tavg_30d",
       "tavg_90d","tmax_30d","tmin_30d","temp_range_mean_30d","ndvi_30d","ndvi_90d",
       "temp_range","t_anom_7_30","rain_intensity_30d","ndvi_change"]
rows = []
for c in num:
    a, b = tr[c].values, te[c].values
    rows.append(dict(feature=c, train_mean=round(a.mean(),4), test_mean=round(b.mean(),4),
                     std_mean_diff=round((b.mean()-a.mean())/(a.std()+1e-9),3),
                     train_std=round(a.std(),4), test_std=round(b.std(),4)))
shift = pd.DataFrame(rows).sort_values("std_mean_diff", key=lambda s: s.abs(), ascending=False)
shift.to_csv("experiments/train_test_shift.csv", index=False)
print(shift.head(12).to_string(index=False))
print("\nzone train%:", (tr_raw.zone.value_counts(normalize=True).round(3).to_dict()),
      "test%:", (te_raw.zone.value_counts(normalize=True).round(3).to_dict()))
print("age groups train%:\n", (pd.cut(tr_raw.age,[0,1,5,18,40,60,200]).value_counts(normalize=True).round(3)))
print("age groups test%:\n", (pd.cut(te_raw.age,[0,1,5,18,40,60,200]).value_counts(normalize=True).round(3)))

# ---- OOF error analysis for RF baseline ----
from src.validation import make_preprocess, fold_metrics
from src.models import make_rf
from src.feature_engineering import BASE_NUMERIC, BASE_CATEGORICAL
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
X, y = tr[BASE_NUMERIC+BASE_CATEGORICAL], tr["is_climate_sensitive"]
oof = np.zeros(len(X))
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for ti, vi in skf.split(X, y):
    p = Pipeline([("pre", make_preprocess(BASE_NUMERIC, BASE_CATEGORICAL)), ("clf", make_rf())])
    p.fit(X.iloc[ti], y.iloc[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:,1]
pred = (oof >= 0.5).astype(int)
pd.DataFrame({"ID": tr_raw["ID"], "y": y.values, "proba": oof, "pred": pred}).to_csv(
    "experiments/oof_rf_baseline.csv", index=False)
ana = tr_raw.assign(y=y.values, pred=pred, oof=oof)
ana["age_band"] = pd.cut(ana.age, [-1,1,5,18,40,60,500],
                          labels=["infant","1-5","5-18","18-40","40-60","60+"])
print("\n--- error by age_band (F1-relevant: FP/FN) ---")
print(ana.groupby("age_band", observed=True).apply(
    lambda g: pd.Series({"n":len(g),"pos_rate":round(g.y.mean(),3),
      "F1":round(fold_metrics(g.y.values,g.oof.values)["f1"],4),
      "FN_rate":round(((g.y==1)&(g.pred==0)).sum()/max((g.y==1).sum(),1),3),
      "FP_rate":round(((g.y==0)&(g.pred==1)).sum()/max((g.y==0).sum(),1),3)}))
      .to_string())
print("\n--- error by zone/gender ---")
for c in ["zone","gender"]:
    print(ana.groupby(c)[["y","pred","oof"]].apply(lambda g: pd.Series({"n":len(g),
      "F1":round(fold_metrics(g["y"].values,g["oof"].values)["f1"],4)})).to_string())
print("\nOOF overall:", {k: round(v,4) for k,v in fold_metrics(y.values,oof).items() if k!='cm'})
print("DONE diagnose")
