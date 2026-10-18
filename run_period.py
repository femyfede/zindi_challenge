"""Is climate real signal, or just a proxy for the year/recording break?
Test incremental value of each feature family INSIDE each period."""
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from bench import load
from features2 import build

NJ = 4
tr, te = load()
dt = pd.to_datetime(tr["deathdate"])
y = tr["is_climate_sensitive"].values
BREAK = pd.Timestamp("2017-07-01")

RF = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                                    class_weight="balanced_subsample", n_jobs=NJ, random_state=42)


def cv_auc(cols, mask, seed=42, n_splits=5):
    X = tr.loc[mask, cols]
    yy = y[mask]
    if len(np.unique(yy)) < 2 or len(yy) < 150:
        return np.nan, 0
    oof = np.zeros(len(X))
    for a, b in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, yy):
        oof[b] = RF().fit(X.iloc[a], yy[a]).predict_proba(X.iloc[b])[:, 1]
    return roc_auc_score(yy, oof), len(yy)


Xall, num, cat = build(tr, ("demo", "time", "clim", "geo"))
demo_time = [c for c in Xall.columns if not c.startswith(("rain", "tavg", "tmax", "tmin", "ndvi",
           "temp_range", "t_anom", "dry_days", "rain_log", "is_dry", "max_daily", "avg_temp",
           "max_temp", "min_temp", "precip", "latitude", "longitude", "elevation", "slope"))]
clim_only = [c for c in Xall.columns if c.startswith(("rain", "tavg", "tmax", "tmin", "ndvi",
            "temp_range", "t_anom", "dry_days", "rain_log", "is_dry", "max_daily", "avg_temp",
            "max_temp", "min_temp", "precip"))]
geo_only = ["latitude", "longitude", "elevation", "slope"]

periods = {"ALL": np.ones(len(tr), bool),
           "pre(<=2017-06)": (dt < BREAK).values,
           "post(>=2017-07)": (dt >= BREAK).values}

print(f"{'period':16s} {'cols':16s} {'auc':>7s} {'n':>6s}  base")
for pname, m in periods.items():
    yr = dt[m].dt.year
    print(f"  period {pname}: years {yr.min()}-{yr.max()}, base rate {y[m].mean():.3f}")
print()
rows = []
for pname, m in periods.items():
    for cname, cols in [("demo+time", demo_time), ("clim only", clim_only),
                        ("geo only", geo_only), ("demo+time+clim", demo_time + clim_only),
                        ("demo+time+geo", demo_time + geo_only)]:
        cols = [c for c in cols if c in Xall.columns]
        t0 = time.time()
        a, n = cv_auc(cols, m)
        rows.append(dict(period=pname, cols=cname, auc=a, n=n, base=y[m].mean() if m.any() else np.nan))
        print(f"{pname:16s} {cname:16s} {a:7.4f} {n:6d}  ({time.time()-t0:.0f}s)", flush=True)

pd.DataFrame(rows).to_csv("experiments/period_conditioned.csv", index=False)

print("\n=== age-0 / age-1 / age 2-5 rate by period (correct bins) ===")
for lbl, mask in [("age==0", tr.age.values == 0), ("age==1", tr.age.values == 1),
                  ("age 2-5", (tr.age >= 2) & (tr.age <= 5)), ("age>5", tr.age.values > 5)]:
    pre = mask & (dt < BREAK).values; post = mask & (dt >= BREAK).values
    print(f"  {lbl:9s} pre n={pre.sum():4d} rate={y[pre].mean() if pre.sum() else float('nan'):.3f} | "
          f"post n={post.sum():4d} rate={y[post].mean() if post.sum() else float('nan'):.3f}")
