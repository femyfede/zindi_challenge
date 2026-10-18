"""DIAGNOSTIC: why does local CV (0.83) not transfer to the leaderboard?
Compares the TRAIN distribution the model was validated on vs the TEST distribution
it actually has to predict. Uses test FEATURES only - never test labels."""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.data_loader import load_data, merge_climate
from src.features_advanced import add_features_advanced, compute_location_climatology, ADV_NUM, ADV_CAT
from src.encoding import target_encode_oof, target_encode_full

tr_raw, te_raw, cl = load_data()
tr = merge_climate(tr_raw, cl); te = merge_climate(te_raw, cl)
clim = compute_location_climatology(tr)
tr = add_features_advanced(tr, clim); te = add_features_advanced(te, clim)

y = tr["is_climate_sensitive"].values
loc_oof, _ = target_encode_oof(tr, pd.Series(y), "location", n_splits=5, smooth=30)
tr["loc_te"] = loc_oof
_, lte, _, _ = target_encode_full(tr, pd.Series(y), te, "location", smooth=30)
te["loc_te"] = lte
NUM = ADV_NUM + ["loc_te"]; CAT = ADV_CAT

# ---- 1. How much of TEST is location-seen vs location-unseen? ----
seen = set(tr["location"].unique())
te["loc_seen"] = te["location"].isin(seen).astype(int)
print("=" * 70)
print("1. LOCATION OVERLAP")
print(f"   Test rows at a location seen in train: {te['loc_seen'].sum()}/{len(te)} "
      f"({te['loc_seen'].mean():.1%})")
print(f"   Test rows at UNSEEN locations       : {(1-te['loc_seen']).sum()}/{len(te)} "
      f"({(1-te['loc_seen']).mean():.1%})")

# ---- 2. Age composition (dominant feature) ----
print("=" * 70)
print("2. AGE COMPOSITION  (age drives the label)")
bands = [-1, 1, 5, 18, 40, 60, 500]
labs = ["infant(<=1)", "1-5", "5-18", "18-40", "40-60", "60+"]
a_tr = pd.cut(tr["age"], bands, labels=labs).value_counts(normalize=True).reindex(labs)
a_te = pd.cut(te["age"], bands, labels=labs).value_counts(normalize=True).reindex(labs)
cmp = pd.DataFrame({"train%": (a_tr * 100).round(1), "test%": (a_te * 100).round(1)})
cmp["delta_pp"] = (cmp["test%"] - cmp["train%"]).round(1)
print(cmp.to_string())
print(f"\n   HARD band (60+, F1 was only 0.13): train {(a_tr['60+']*100):.1f}%  "
      f"test {(a_te['60+']*100):.1f}%")

# ---- 3. Zone / gender ----
print("=" * 70)
print("3. ZONE & GENDER")
for c in ["zone", "gender"]:
    tr_p = tr[c].value_counts(normalize=True)
    te_p = te[c].value_counts(normalize=True)
    d = pd.DataFrame({"train%": (tr_p * 100).round(1), "test%": (te_p * 100).round(1)})
    d["delta_pp"] = (d["test%"] - d["train%"]).round(1)
    print(f"\n{c}:"); print(d.to_string())

# ---- 4. Train-year vs test-year (base rate drifts DOWN over time) ----
print("=" * 70)
print("4. DEATH YEAR  (target base rate drifts 0.77 in 2009 -> 0.46 in 2019)")
tr["yr"] = pd.to_datetime(tr["deathdate"]).dt.year
te["yr"] = pd.to_datetime(te["deathdate"]).dt.year
yrs = sorted(set(tr["yr"]) | set(te["yr"]))
tab = pd.DataFrame({"train_n": tr["yr"].value_counts(), "test_n": te["yr"].value_counts()}
                  ).reindex(yrs).fillna(0).astype(int)
tab["train_base_rate"] = tr.groupby("yr")["is_climate_sensitive"].mean().reindex(yrs).round(3)
tab["test_share%"] = (tab["test_n"] / tab["test_n"].sum() * 100).round(1)
print(tab.to_string())

# expected test base rate = weighted by test year distribution using train rates
rates = tr.groupby("yr")["is_climate_sensitive"].mean()
w = te["yr"].map(rates).dropna()
print(f"\n   Train overall base rate          : {y.mean():.4f}")
print(f"   Expected test base rate (by year): {w.mean():.4f}  <- TRUE likely test prevalence")

# ---- 5. Predicted probability distribution: OOF(train) vs test ----
print("=" * 70)
print("5. PREDICTED PROBABILITY DISTRIBUTION (OOF on train vs actual test preds)")
from sklearn.model_selection import StratifiedKFold
from src.validation import make_preprocess
X = tr[NUM + CAT]
oof = np.zeros(len(X))
for ti, vi in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    p = Pipeline([("pre", make_preprocess(NUM, CAT)),
                  ("clf", RandomForestClassifier(n_estimators=1200, min_samples_leaf=2,
                        max_features="sqrt", max_depth=20, class_weight="balanced_subsample",
                        n_jobs=-1, random_state=42))])
    p.fit(X.iloc[ti], y[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:, 1]
full = Pipeline([("pre", make_preprocess(NUM, CAT)),
                 ("clf", RandomForestClassifier(n_estimators=1200, min_samples_leaf=2,
                       max_features="sqrt", max_depth=20, class_weight="balanced_subsample",
                       n_jobs=-1, random_state=42))]).fit(X, y)
tp = full.predict_proba(te[NUM + CAT])[:, 1]
pd_ = pd.DataFrame({
    "OOF_train": pd.Series(oof).describe(percentiles=[.25, .5, .75]),
    "TEST_pred": pd.Series(tp).describe(percentiles=[.25, .5, .75])}).round(3)
print(pd_.to_string())
print(f"\n   Predicted positives: OOF {float((oof>=.5).mean()):.4f}  vs  TEST {float((tp>=.5).mean()):.4f}")
print(f"   Mean confidence   : OOF {oof.mean():.4f}  vs  TEST {tp.mean():.4f}")

# ---- 6. Probability by seen vs unseen test location ----
print("=" * 70)
print("6. TEST PREDICTIONS SPLIT BY LOCATION SEEN/UNSEEN")
d = pd.DataFrame({"proba": tp, "seen": te["loc_seen"].values})
print(d.groupby("seen")["proba"].describe().round(3).to_string())
print(f"\n   pos_rate on SEEN test rows   : {float((d[d.seen==1].proba>=.5).mean()):.4f}")
print(f"   pos_rate on UNSEEN test rows : {float((d[d.seen==0].proba>=.5).mean()):.4f}")

# ---- 7. Per-age-band positive rate on test vs train (predicted) ----
print("=" * 70)
print("7. PREDICTED POSITIVE RATE BY AGE BAND: train(OOF) vs test")
o = pd.DataFrame({"band": pd.cut(tr["age"], bands, labels=labs), "p": oof, "y": y})
t = pd.DataFrame({"band": pd.cut(te["age"], bands, labels=labs), "p": tp})
g = o.groupby("band", observed=True).apply(
    lambda g: pd.Series({"n": len(g), "true_rate": g.y.mean(),
                         "oof_pred_rate": (g.p >= .5).mean()}))
h = t.groupby("band", observed=True).apply(lambda g: pd.Series({"n": len(g), "test_pred_rate": (g.p >= .5).mean()}))
print(pd.concat([g, h], axis=1).round(3).to_string())
print("\nDONE diagnostic")
