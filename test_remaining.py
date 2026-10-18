"""Remaining untried avenues, all evaluated under rolling year-by-year CV
(the scheme that best matches test-time geography) AND stratified 5-fold:
  A) feature selection (permutation importance) to cut variance from ~90 noisy features
  B) dedicated specialist model for the hardest age band (60+, F1 was only 0.13)
  C) monotonic constraints on physically-signed climate features (XGB/LGBM)
  D) fine age binning (finer than 6 bands)
  E) isotonic + Platt calibration fitted on rolling OOF
"""
import sys, warnings, time, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.data_loader import load_data, merge_climate
from src.features_advanced import add_features_advanced, compute_location_climatology, ADV_NUM, ADV_CAT
from src.encoding import target_encode_oof
from src.validation import make_preprocess, fold_metrics, combined_score

tr_raw, te_raw, cl = load_data()
mtr = merge_climate(tr_raw, cl)
clim = compute_location_climatology(mtr)
d = add_features_advanced(mtr, clim)
lo, _ = target_encode_oof(d, pd.Series(d["is_climate_sensitive"]), "location", n_splits=5, smooth=30)
d["loc_te"] = lo
y = d["is_climate_sensitive"].values
NUM = ADV_NUM + ["loc_te"]
CAT = ADV_CAT
X = d[NUM + CAT]

def mk_rf():
    return RandomForestClassifier(n_estimators=1200, min_samples_leaf=2, max_features="sqrt",
                                  max_depth=20, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=42)

def cv_rolling(num, cat, make_fn, Xdf=None, n_initial=5, n_splits=5):
    Xd = Xdf if Xdf is not None else X
    oof = np.zeros(len(Xd)); mask = np.zeros(len(Xd), bool)
    years = sorted(d["death_year"].unique()); e = n_initial - 1
    for _ in range(n_splits):
        if e >= len(years) - 1: break
        ti = d.index[d["death_year"].isin(years[:e + 1])].values
        vi = d.index[d["death_year"] == years[e + 1]].values
        if Xdf is None:
            p = Pipeline([("pre", make_preprocess(num, cat)), ("clf", make_fn())])
            p.fit(X.iloc[ti], y[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:, 1]
        else:
            p = make_fn(); p.fit(Xd.iloc[ti][num], y[ti]); oof[vi] = p.predict_proba(Xd.iloc[vi][num])[:, 1]
        mask[vi] = True; e += 1
    return oof, mask

def strat(num, cat, make_fn, Xdf=None):
    Xd = Xdf if Xdf is not None else X
    oof = np.zeros(len(Xd))
    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=42).split(Xd, y):
        if Xdf is None:
            p = Pipeline([("pre", make_preprocess(num, cat)), ("clf", make_fn())])
            p.fit(X.iloc[ti], y[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:, 1]
        else:
            p = make_fn(); p.fit(Xd.iloc[ti][num], y[ti]); oof[vi] = p.predict_proba(Xd.iloc[vi][num])[:, 1]
    return oof

def show(tag, m, extra=""):
    print(f"{tag:34s} F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={combined_score(m['f1'],m['auc']):.4f} {extra}", flush=True)

res = {}
# ---------- baseline ----------
o_roll, m_roll = cv_rolling(NUM, CAT, mk_rf)
mb = fold_metrics(y[m_roll], o_roll[m_roll]); res["baseline_rf_roll"] = mb
show("BASELINE rf (rolling)", mb)
o_str = strat(NUM, CAT, mk_rf); mstr = fold_metrics(y, o_str); res["baseline_rf_strat"] = mstr
show("BASELINE rf (strat5)", mstr)

# ---------- A) feature selection ----------
print("\n--- A) permutation importance on 80/20 split ---", flush=True)
from sklearn.model_selection import train_test_split
ti, vi = train_test_split(np.arange(len(X)), test_size=0.25, stratify=y, random_state=42)
psel = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_rf())])
psel.fit(X.iloc[ti], y[ti])
from sklearn.inspection import permutation_importance
pi = permutation_importance(psel, X.iloc[vi], y[vi], n_repeats=5, random_state=42,
                            scoring="roc_auc", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=NUM + CAT).sort_values(ascending=False)
print(imp.head(20).round(5).to_string(), flush=True)
imp.to_csv("experiments/perm_importance_full.csv")
for topk in [20, 30, 45]:
    sel = list(imp.head(topk).index)
    num_s = [c for c in sel if c in NUM]; cat_s = [c for c in sel if c in CAT]
    orl, mrl = cv_rolling(num_s, cat_s, mk_rf)
    m1 = fold_metrics(y[mrl], orl[mrl])
    o1 = strat(num_s, cat_s, mk_rf); m2 = fold_metrics(y, o1)
    res[f"top{topk}"] = m1
    show(f"A) top{topk} (rolling)", m1); show(f"A) top{topk} (strat5)", m2)

# ---------- B) age-band specialist blend ----------
print("\n--- B) age-band specialist ---", flush=True)
bands = np.digitize(d["age"].values, [5, 18, 40, 60])
o_spec = o_roll.copy()
for b in [0, 1, 2, 3]:
    vb = np.where(bands == b)[0]
    vbm = vb[m_roll[vb]]
    if len(vbm) < 30: continue
    sub = d.iloc[vbm][NUM + CAT]
    ys = y[vbm]
    oo = np.zeros(len(vbm))
    for a, b2 in StratifiedKFold(3, shuffle=True, random_state=42).split(sub, ys):
        p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk_rf())])
        p.fit(sub.iloc[a], ys[a]); oo[b2] = p.predict_proba(sub.iloc[b2])[:, 1]
    o_spec[vbm] = oo
    print(f"   band{b} n={len(vbm)} F1={fold_metrics(ys,oo)['f1']:.4f}", flush=True)
mspec = fold_metrics(y[m_roll], o_spec[m_roll]); res["ageband_specialist"] = mspec
show("B) age-band specialist (roll)", mspec)
for w in [0.2, 0.35, 0.5]:
    bl = (1 - w) * o_roll + w * o_spec
    mbl = fold_metrics(y[m_roll], bl[m_roll])
    show(f"B) blend w_specialist={w}", mbl)

# ---------- C) monotonic constraints ----------
print("\n--- C) monotonic constraints ---", flush=True)
# physically signed: hotter -> more stress(+), more rain -> wetter(-ndvi stress),
# higher ndvi -> generally better veg(+). We keep constraints weak & defensible.
MONO = {}
for c, sgn in [("tavg_30d", 1), ("tmax_30d", 1), ("heat_30d", 1), ("APP_T_max_30d", 1),
               ("temp_range", 1), ("is_dry_30d", 1), ("dry_days_30d", 1),
               ("rain_intensity_30d", -1), ("VPD_mean_30d", 1), ("ws10m_mean_30d", -1)]:
    if c in NUM: MONO[c] = sgn
mono_num = [c for c in MONUM if False] if False else [c for c in MONO]
print("monotone feats:", len(mono_num), flush=True)
def mk_xgb_mono():
    return XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.03, subsample=0.7,
                         colsample_bytree=0.7, reg_lambda=1.0, scale_pos_weight=1.5,
                         n_jobs=-1, random_state=42, tree_method="hist",
                         eval_metric="logloss", monotone_constraints=pd.Series(MONO).to_dict())
try:
    orl2, mrl2 = cv_rolling(NUM, CAT, mk_xgb_mono)
    mmono = fold_metrics(y[mrl2], orl2[mrl2]); res["xgb_mono"] = mmono
    show("C) xgb monotone (rolling)", mmono)
    bl = 0.5 * orl2 + 0.5 * o_roll
    mb2 = fold_metrics(y[mrl2], bl[mrl2]); show("C) blend mono+rf (rolling)", mb2)
except Exception as e:
    print("mono failed:", str(e)[:150], flush=True)

# ---------- D) finer age bins ----------
print("\n--- D) finer age bins ---", flush=True)
d2 = d.copy()
d2["age_fine"] = pd.cut(d2["age"], [-1, 1, 2, 3, 5, 10, 18, 30, 45, 60, 75, 200],
                        labels=False).astype(float)
NUM2 = NUM + ["age_fine"]
X2 = d[NUM2 + CAT]
or3, mrl3 = cv_rolling(NUM2, CAT, mk_rf, Xdf=X2)
mfine = fold_metrics(y[mrl3], or3[mrl3]); res["fine_age"] = mfine
show("D) fine age bins (rolling)", mfine)

# ---------- E) calibration on rolling OOF ----------
print("\n--- E) calibration fitted on rolling OOF ---", flush=True)
bl_oof = o_roll[m_roll]; yy = y[m_roll]
for name, cal in [("isotonic", IsotonicRegression(out_of_bounds="clip")),
                  ("platt", LogisticRegression(max_iter=1000))]:
    cal.fit(bl_oof.reshape(-1, 1), yy)
    pc = cal.predict(bl_oof.reshape(-1, 1))
    mc = fold_metrics(yy, pc)
    show(f"E) {name} (in-sample-ish)", mc)

json.dump({k: {kk: float(vv) for kk, vv in v.items() if kk != "cm"}
           for k, v in res.items()}, open("experiments/final_avenues.json", "w"), indent=2)
print("\nDONE avenues")