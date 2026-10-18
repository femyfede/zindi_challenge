import pandas as pd, numpy as np

tr = pd.read_csv("Train.csv")
te = pd.read_csv("Test.csv")
cl = pd.read_csv("climate_features.csv")
pd.set_option("display.width", 200)

print("=== ORDERING ===")
print("train head dates:", tr.deathdate.head(8).tolist())
print("test  head dates:", te.deathdate.head(8).tolist())
print("clim  head dates:", cl.deathdate.head(8).tolist())
print("train sorted by date?", tr.deathdate.is_monotonic_increasing)
print("test  sorted by date?", te.deathdate.is_monotonic_increasing)
print("clim  sorted by date?", cl.deathdate.is_monotonic_increasing)

# where do test IDs appear in climate_features?
pos_cl = {i: k for k, i in enumerate(cl.ID)}
pos_tr = {i: k for k, i in enumerate(tr.ID)}
te_pos = np.array([pos_cl[i] for i in te.ID])
tr_pos = np.array([pos_cl[i] for i in tr.ID])
print("\nclim pos of train rows: first 12 ->", tr_pos[:12])
print("clim pos of test  rows: first 12 ->", te_pos[:12])
print("train clim-pos sorted == 0..n?", np.array_equal(np.sort(tr_pos), np.arange(len(tr_pos))))
print("test clim-pos sorted == 0..n?", np.array_equal(np.sort(te_pos), np.arange(len(te_pos))))

print("\n=== ID STRUCTURE ===")
hexs = tr.ID.str.replace("ID_", "", regex=False)
print("hex unique:", hexs.nunique(), "of", len(hexs))
print("len distribution:", hexs.str.len().value_counts().to_dict())
hx = hexs.apply(lambda s: int(s, 16))
print("hex as int: min %d max %d" % (hx.min(), hx.max()))
for shift in range(0, 28):
    b = (hx >> shift) & 1
    if b.nunique() > 1:
        from sklearn.metrics import roc_auc_score
        a = roc_auc_score(tr.is_climate_sensitive, b)
        if abs(a - 0.5) > 0.05:
            print("  bit", shift, "auc", round(a, 4))
print("hex-int monotonic corr with label:", np.corrcoef(hx, tr.is_climate_sensitive)[0, 1].round(4))
print("hex-int corr with date:", np.corrcoef(hx, pd.to_datetime(tr.deathdate).astype("int64"))[0, 1].round(4))
print("hex-int corr with age:", np.corrcoef(hx, tr.age)[0, 1].round(4))

print("\n=== DUPLICATES ===")
feat = ["zone","gender","age","avg_temperature","max_temperature","min_temperature","precipitation","latitude","longitude","location"]
allr = pd.concat([tr.assign(_s=1), te.assign(_s=0)])
print("exact dup feature rows within train:", tr.duplicated(feat).sum())
d = allr.duplicated(feat, keep=False)
if d.any():
    g = allr[d].groupby(feat)["_s"].nunique()
    print("feature-identical groups spanning train+test:", (g > 1).sum(), " total rows:", d.sum())

print("\n=== AGE ===")
print(tr.age.describe())
print("non-integer ages:", (tr.age % 1 != 0).sum())
print("age==0:", (tr.age == 0).sum(), " age<1:", (tr.age < 1).sum())
print("age bins target rate:")
print(tr.groupby(pd.cut(tr.age, [0,1,3,5,10,18,30,45,60,75,90,120]), observed=True).is_climate_sensitive.agg(["size","mean"]))

print("\n=== TARGET BY CAT ===")
print(tr.groupby("gender").is_climate_sensitive.agg(["size","mean"]))
print(tr.groupby("zone").is_climate_sensitive.agg(["size","mean"]))
print("\n=== TARGET BY YEAR ===")
tr["yr"] = pd.to_datetime(tr.deathdate).dt.year
te["yr"] = pd.to_datetime(te.deathdate).dt.year
print(tr.groupby("yr").is_climate_sensitive.agg(["size","mean"]))
print("test year counts:", te.yr.value_counts().sort_index().to_dict())
