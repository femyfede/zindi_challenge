"""Age structure + honest upper-bound analysis for F1@0.5 and AUC."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from bench import load

tr, te = load()
y = tr["is_climate_sensitive"].values
dt = pd.to_datetime(tr["deathdate"])
tr["yr"] = dt.dt.year

print("=== target rate by EXACT age (counts >= 5) ===")
g = tr.groupby("age").is_climate_sensitive.agg(["size", "sum", "mean"])
g = g[g["size"] >= 5]
print(g.head(30).to_string())
print("\nage==0 : n=%d rate=%.4f" % ((tr.age == 0).sum(), y[tr.age.values == 0].mean()))
print("age 1-5 : n=%d rate=%.4f" % (((tr.age >= 1) & (tr.age <= 5)).sum(), y[((tr.age >= 1) & (tr.age <= 5)).values].mean()))
print("age>5   : n=%d rate=%.4f" % ((tr.age > 5).sum(), y[tr.age.values > 5].mean()))

print("\n=== age0 x year ===")
a0 = tr[tr.age == 0].groupby("yr").is_climate_sensitive.agg(["size", "mean"])
print(a0.to_string())
print("\n=== age>5 x year ===")
print(tr[tr.age > 5].groupby("yr").is_climate_sensitive.agg(["size", "mean"]).to_string())

print("\n=== F1 of trivial policies (whole train) ===")
p = y.mean()
def rep(name, pred):
    print(f"  {name:34s} F1={f1_score(y, pred):.4f} posrate={pred.mean():.3f}")
rep("all ones", np.ones(len(y), int))
rep("all zeros", np.zeros(len(y), int))
rep("age<=5", (tr.age <= 5).values.astype(int))
rep("age<=1", (tr.age <= 1).values.astype(int))
rep("not (age>=70 & age<75)", (~((tr.age >= 70) & (tr.age < 75)).values).astype(int))

print("\n=== Bayes-ish F1 upper bound via age x gender x zone x year-decile rate table (OOF) ===")
tr["dec"] = (tr["yr"] // 2).astype(str)
tr["ageb"] = pd.cut(tr.age, [0, 1, 3, 6, 15, 30, 45, 60, 75, 90, 200]).astype(str)
tr["cell"] = tr["ageb"] + "|" + tr["gender"] + "|" + tr["zone"] + "|" + tr["dec"]
oof = np.zeros(len(tr))
for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y):
    gm = pd.Series(y[a]).groupby(tr["cell"].values[a]).agg(["sum", "count"])
    prior = y[a].mean()
    e = (gm["sum"] + 20 * prior) / (gm["count"] + 20)
    oof[b] = pd.Series(tr["cell"].values[b]).map(e).fillna(prior).values
f1 = f1_score(y, (oof >= 0.5).astype(int)); auc = roc_auc_score(y, oof)
print(f"  cell-rate model: F1={f1:.4f} AUC={auc:.4f} combined={0.6*f1+0.4*auc:.4f} posrate={(oof>=0.5).mean():.3f}")

print("\n=== how much can AUC move with year alone (in-sample-ish OOF) ===")
enc = np.zeros(len(tr))
for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y):
    m = pd.Series(y[a]).groupby(tr["yr"].values[a]).mean()
    enc[b] = pd.Series(tr["yr"].values[b]).map(m).fillna(y[a].mean()).values
print("  year TE AUC:", round(roc_auc_score(y, enc), 4))
