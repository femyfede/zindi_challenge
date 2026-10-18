"""Pin down the structural break and quantify each feature family's honest contribution."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, f1_score
from bench import load

tr, te = load()
y = tr["is_climate_sensitive"].values
dt = pd.to_datetime(tr["deathdate"])
tr["d"] = dt

inf = tr.age <= 1
print("=== infant(age<=1) climate-sensitive rate by QUARTER ===")
q = tr["d"].dt.to_period("Q")
g = tr.groupby(q).apply(lambda x: pd.Series({
    "n": len(x), "inf_share": (x.age <= 1).mean(), "inf_rate": x.loc[x.age <= 1, "is_climate_sensitive"].mean(),
    "u5_rate": x.loc[x.age <= 5, "is_climate_sensitive"].mean(),
    "a5_rate": x.loc[x.age > 5, "is_climate_sensitive"].mean()}), include_groups=False)
print(g.round(3).to_string())

print("\n=== overall rate by HALF-YEAR (to see base-rate stability) ===")
h = tr["d"].dt.to_period("Y").astype(str) + "H" + ((tr["d"].dt.month > 6).astype(int) + 1).astype(str)
print(tr.groupby(h).is_climate_sensitive.agg(["size", "mean"]).round(3).to_string())

print("\n=== share of test rows before/after 2017-07-01 ===")
cut = pd.Timestamp("2017-07-01")
print("train pre :", int((dt < cut).sum()), " post:", int((dt >= cut).sum()))
dte = pd.to_datetime(te["deathdate"])
print("test  pre :", int((dte < cut).sum()), " post:", int((dte >= cut).sum()))
print("\ntest infant share pre/post:", round((te.age[dte < cut] <= 1).mean(), 3), round((te.age[dte >= cut] <= 1).mean(), 3))
print("train infant share pre/post:", round(tr.age[dt < cut].le(1).mean(), 3), round(tr.age[dt >= cut].le(1).mean(), 3))

print("\n=== coordinate overlap check ===")
trc = set(zip(tr.latitude.round(4), tr.longitude.round(4)))
tec = set(zip(te.latitude.round(4), te.longitude.round(4)))
print("train coords:", len(trc), "test coords:", len(tec), "shared:", len(trc & tec))
print("test locations in train locations:", len(set(te.location) & set(tr.location)))

print("\n=== age x period cell table (the structure) ===")
tr["period"] = np.where(dt < cut, "pre", "post")
tr["ageb"] = pd.cut(tr.age, [0, 1, 5, 17, 44, 59, 69, 74, 84, 200],
                    labels=["0-1", "2-5", "6-17", "18-44", "45-59", "60-69", "70-74", "75-84", "85+"]).astype(str)
print(pd.crosstab(tr.ageb, tr.period, values=tr.is_climate_sensitive, aggfunc=["size", "mean"]).round(3).to_string())
