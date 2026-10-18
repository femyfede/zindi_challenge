import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

tr = pd.read_csv("Train.csv"); te = pd.read_csv("Test.csv"); cl = pd.read_csv("climate_features.csv")
pd.set_option("display.width", 250)

pos_cl = {i: k for k, i in enumerate(cl.ID)}
te_pos = np.array([pos_cl[i] for i in te.ID])
tr_pos = np.array([pos_cl[i] for i in tr.ID])
print("test clim positions: min %d max %d  monotone-in-test-order? %s" % (te_pos.min(), te_pos.max(), bool(np.all(np.diff(te_pos) > 0))))
print("train block == 0..3145 exactly:", np.array_equal(np.sort(tr_pos), np.arange(3146)))
print("test block == 3146..4175 exactly:", np.array_equal(np.sort(te_pos), np.arange(3146, 4176)))
print("\nfirst 20 te_pos:", te_pos[:20].tolist())

# what is the TEST BLOCK ordered by?
blk = cl.iloc[3146:].copy()
blk["te_row"] = te_pos - 3146
m = te.copy(); m["climpos"] = te_pos
print("\ntest-block vs Test.csv: date monotonic within block?", pd.to_datetime(blk.deathdate).is_monotonic_increasing)
print("test-block dates head:", blk.deathdate.head(12).tolist())
print("Test.csv   dates head:", m.deathdate.head(12).tolist())

# is the test block sorted by any column?
blk2 = blk.merge(m[["ID","deathdate","location","zone","age","latitude"]].rename(columns={"deathdate":"dd_te"}), on="ID")
print("\ntest block == Test.csv row order? ", bool(np.array_equal(blk2.te_row.values, np.arange(1030))))

# ID hex order within each block
def hx(s): return int(s.replace("ID_",""), 16)
for nm, blkdf in [("train blk", cl.iloc[:3146]), ("test blk", cl.iloc[3146:])]:
    h = np.array([hx(i) for i in blkdf.ID])
    print(f"{nm}: hex monotonic increasing? {bool(np.all(np.diff(h)>0))}")

print("\n=== ID bits vs label (train) ===")
h = np.array([hx(i) for i in tr.ID], dtype=object)
h = np.array([int(x) for x in h], dtype=np.uint64)
for shift in range(0, 32):
    b = ((h >> np.uint64(shift)) & np.uint64(1)).astype(int)
    if b.min() != b.max():
        a = roc_auc_score(tr.is_climate_sensitive, b)
        if abs(a - 0.5) > 0.04:
            print("  bit", shift, "auc", round(a, 4))
print("hex corr w/ label", np.corrcoef(h.astype(float), tr.is_climate_sensitive)[0,1].round(4))
print("hex corr w/ date ", np.corrcoef(h.astype(float), pd.to_datetime(tr.deathdate).astype("int64"))[0,1].round(4))
print("hex corr w/ age  ", np.corrcoef(h.astype(float), tr.age)[0,1].round(4))

print("\n=== RUN STRUCTURE (train block, date order) ===")
y = tr.is_climate_sensitive.values
print("adjacent agreement in train (date-sorted):", (y[1:]==y[:-1]).mean().round(4), " base p:", round(y.mean(),4))
print("expected agreement if random:", round(y.mean()**2+(1-y.mean())**2,4))

print("\n=== DUP CHECK ===")
feat = ["zone","gender","age","avg_temperature","max_temperature","min_temperature","precipitation","latitude","longitude","location"]
both = pd.concat([tr.assign(s=1), te.assign(s=0)], ignore_index=True)
d = both.duplicated(feat, keep=False)
print("rows in dup groups:", int(d.sum()))
if d.any():
    gg = both[d].groupby(feat).s.nunique()
    print("dup groups spanning train+test:", int((gg>1).sum()))
# location-level latitude/longitude duplicates
print("\nlocations per (lat,lon) train:", tr.groupby(["latitude","longitude"]).location.nunique().value_counts().to_dict())
