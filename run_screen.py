"""Univariate signal screen + first model benchmark."""
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from bench import load, repeated_cv, show

tr, te = load()
y = tr["is_climate_sensitive"]
d = tr.drop(columns=["is_climate_sensitive", "ID", "location"]).copy()
dt = pd.to_datetime(tr["deathdate"])
d["death_year"] = dt.dt.year; d["death_month"] = dt.dt.month
d["doy"] = dt.dt.dayofyear
d["age_log"] = np.log1p(tr.age)

print("n =", len(tr), " pos rate =", round(y.mean(), 4))
print("\n=== UNIVARIATE AUC (|auc-0.5| desc) ===")
rows = []
for c in d.columns:
    if d[c].dtype == object:
        v = pd.factorize(d[c])[0]
    else:
        v = d[c].values
    if len(np.unique(v)) < 2:
        continue
    try:
        a = roc_auc_score(y, v)
    except Exception:
        continue
    rows.append((c, round(a, 4), round(abs(a - 0.5), 4), int(d[c].nunique())))
u = pd.DataFrame(rows, columns=["feat", "auc", "absauc", "nuniq"]).sort_values("absauc", ascending=False)
print(u.head(35).to_string(index=False))

print("\n=== TARGET RATE BY MONTH / YEAR / ZONE / GENDER ===")
tr2 = tr.copy(); tr2["m"] = dt.dt.month; tr2["y_"] = dt.dt.year
print(tr2.groupby("m").is_climate_sensitive.agg(["size", "mean"]).T.to_string())
print(tr2.groupby("y_").is_climate_sensitive.agg(["size", "mean"]).T.to_string())
print(tr2.groupby("gender").is_climate_sensitive.agg(["size", "mean"]).T.to_string())
print(tr2.groupby("zone").is_climate_sensitive.agg(["size", "mean"]).T.to_string())

print("\n=== LOCATION base rates (train) vs test locations ===")
lr = tr.groupby("location").is_climate_sensitive.agg(["size", "mean"]).sort_values("mean")
print(lr.to_string())
print("\ntest locs:")
print(te.location.value_counts().to_string())
