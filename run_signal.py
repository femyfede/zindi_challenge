"""Where does the signal live? OOF AUC of hierarchical / batch group encodings + structure probes."""
import numpy as np, pandas as pd, warnings, re
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from bench import load

tr, te = load()
y = tr["is_climate_sensitive"].values
dt = pd.to_datetime(tr["deathdate"])
tr["yr"] = dt.dt.year; tr["mo"] = dt.dt.month
tr["district"] = tr["location"].str.split(",").str[-2].str.strip()
tr["facility"] = tr["location"].str.split(",").str[0].str.strip()
tr["q"] = tr["yr"].astype(str) + "Q" + ((tr["mo"] - 1) // 3 + 1).astype(str)
print("districts (train):"); print(tr.district.value_counts().to_string())
te_dt = pd.to_datetime(te["deathdate"])
te["district"] = te["location"].str.split(",").str[-2].str.strip()
print("\ndistricts (test):"); print(te.district.value_counts().to_string())
print("\noverlap:", set(tr.district) & set(te.district))

skf = pd.Series(np.arange(len(tr)))


def oof_group_auc(col, smooth=10.0, min_n=1):
    """Leak-free group mean rate via 5-fold OOF; report AUC of the encoding."""
    from sklearn.model_selection import StratifiedKFold
    enc = np.full(len(tr), np.nan)
    g = tr[col].values
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y):
        gm = pd.Series(y[a]).groupby(g[a]).agg(["sum", "count"])
        prior = y[a].mean()
        e = (gm["sum"] + smooth * prior) / (gm["count"] + smooth)
        enc[b] = pd.Series(g[b]).map(e).fillna(prior).values
    n_groups = tr.groupby(col).ngroups
    sizes = tr.groupby(col).size()
    return roc_auc_score(y, enc), n_groups, int((sizes >= 8).sum()), enc


print("\n=== OOF AUC of group target-encoding (smooth=10) ===")
print(f"{'grouping':34s} {'auc':>7s} {'ngrp':>6s} {'n>=8':>6s}")
cands = ["district", "location", "facility", "yr", "mo", "q", "yr_month",
         "location_yr", "location_q", "location_mo", "location_yr_month",
         "district_yr", "district_yr_month", "date", "deathdate"]
MON = tr["mo"].map(lambda v: f"{int(v):02d}")
tr["yr_month"] = tr["yr"].astype(str) + "-" + MON
tr["location_yr"] = tr["location"] + "|" + tr["yr"].astype(str)
tr["location_q"] = tr["location"] + "|" + tr["q"]
tr["location_mo"] = tr["location"] + "|" + MON
tr["location_yr_month"] = tr["location"] + "|" + tr.yr_month
tr["district_yr"] = tr["district"] + "|" + tr["yr"].astype(str)
tr["district_yr_month"] = tr["district"] + "|" + tr.yr_month
for c in cands:
    if c not in tr.columns:
        continue
    a, ng, nb, enc = oof_group_auc(c)
    print(f"{c:34s} {a:7.4f} {ng:6d} {nb:6d}")

print("\n=== age fine structure ===")
print("age value counts head/tail:")
print(tr.age.value_counts().sort_index().head(15).to_string())
print("n distinct ages:", tr.age.nunique(), " non-integer:", int((tr.age % 1 != 0).sum()))
print("\nage -> target rate (bins of 5):")
b = pd.cut(tr.age, list(range(0, 96, 5)) + [200])
print(tr.groupby(b, observed=True).is_climate_sensitive.agg(["size", "mean"]).to_string())

print("\n=== temporal autocorrelation within location ===")
for lag in [1, 2, 3, 5, 10]:
    s = tr.sort_values(["location", "deathdate"])
    yy = s.is_climate_sensitive.values
    print(f"  lag {lag}: same-label rate {np.mean(yy[lag:] == yy[:-lag]):.4f}")

print("\n=== batch/date purity in TRAIN (within location) ===")
g = tr.groupby(["location", "deathdate"]).is_climate_sensitive.agg(["size", "mean"])
big = g[g["size"] >= 3]
print("clusters(location,date) size>=3:", len(big),
      " purity(frac==1 or 0):", round(((big["mean"] == 0) | (big["mean"] == 1)).mean(), 4))
p = tr.is_climate_sensitive.mean()
print("expected purity by chance:", round(p**4 + (1 - p)**4, 4))
print(big.sort_values("size", ascending=False).head(12).to_string())
