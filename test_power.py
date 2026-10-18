"""Does NASA POWER external data improve over the provided climate_features.csv?
Builds POWER features once, caches to disk, then runs an honest A/B under
identical stratified 5-fold CV and identical rolling year-by-year CV."""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.data_loader import load_data, merge_climate
from src.features_advanced import add_features_advanced, compute_location_climatology, ADV_NUM, ADV_CAT
from src.encoding import target_encode_oof, target_encode_full
from src.validation import make_preprocess, fold_metrics, combined_score
from power_features import _prep_daily, build_record_features, PW_BASE, PW_COUNTS

CACHE = "data_cache/power_record_features.csv"

def get_power_features(tr_raw, te_raw):
    if os.path.exists(CACHE):
        f = pd.read_csv(CACHE, index_col=0)
        f.index = f.index.astype(int)
        if len(f) == len(tr_raw) + len(te_raw):
            print("loaded cached POWER features", f.shape, flush=True)
            return f.iloc[:len(tr_raw)].reset_index(drop=True), f.iloc[len(tr_raw):].reset_index(drop=True)
    print("building POWER features (first run ~3 min)...", flush=True)
    daily = _prep_daily()
    allrec = pd.concat([tr_raw, te_raw], ignore_index=True)
    t0 = time.time()
    f = build_record_features(allrec, daily)
    f.index = allrec.index
    f.to_csv(CACHE)
    print(f"built {f.shape} in {time.time()-t0:.0f}s", flush=True)
    return f.iloc[:len(tr_raw)].reset_index(drop=True), f.iloc[len(tr_raw):].reset_index(drop=True)

tr_raw, te_raw, cl = load_data()
pw_tr, pw_te = get_power_features(tr_raw, te_raw)

# only well-populated columns
pw_cols_tr = [c for c in PW_BASE + PW_COUNTS if c in pw_tr.columns]
keep = [c for c in pw_cols_tr if pw_tr[c].notna().mean() > 0.9 and pw_tr[c].std() > 1e-9]
print(f"using {len(keep)}/{len(pw_cols_tr)} POWER columns", flush=True)
# derived stress composites
def add_pw_composites(d):
    o = pd.DataFrame(index=d.index)
    o["pw_vpd_anom_7_30"] = d["pw_VPD_mean_7d"] - d["pw_VPD_mean_30d"]
    o["pw_hum_anom_7_30"] = d["pw_RH2M_mean_7d"] - d["pw_RH2M_mean_30d"]
    o["pw_dew_anom_7_30"] = d["pw_T2MDEW_mean_7d"] - d["pw_T2MDEW_mean_30d"]
    o["pw_heat_humidity_30d"] = d["pw_APP_T_max_30d"] * d["pw_RH2M_mean_30d"] / 100.0
    o["pw_vpd_x_ndvi"] = d["pw_VPD_mean_30d"] * d["ndvi_30d"]
    o["pw_dry_hot_30d"] = d["pw_vpd_days_30d"] * (d["pw_APP_T_max_30d"] / 18.0)
    o["pw_dtr_30d"] = d["pw_T2M_MAX_mean_30d"] - d["pw_T2M_MIN_mean_30d"]
    if "pw_wet_days_30d" in d.columns:
        o["pw_rain_int_power"] = d["pw_PRECTOTCORR_sum_30d"] / (d["pw_wet_days_30d"] + 1.0)
    else:
        o["pw_rain_int_power"] = d["pw_PRECTOTCORR_max_30d"]
    return o

def prep(df, clim):
    d = add_features_advanced(df, clim)
    return d

def build(df, clim, pw, use_pw):
    d = prep(df, clim)
    base = ADV_NUM.copy()
    if use_pw:
        pw2 = pw.copy()
        pw2["ndvi_30d"] = d["ndvi_30d"].values
        comp = add_pw_composites(pw2)
        comp.index = d.index
        d = pd.concat([d.reset_index(drop=True), keep_feats.reset_index(drop=True),
                       comp.reset_index(drop=True)], axis=1)
        return d, base + keep + list(comp.columns)
    return d, base

clim = compute_location_climatology(merge_climate(tr_raw, cl))
keep_feats = pw_tr[keep].copy()
keep_feats.index = range(len(pw_tr))
_mtr = merge_climate(tr_raw, cl); _mte = merge_climate(te_raw, cl)
tr_p, TR_NUM = build(_mtr.copy(), clim, pw_tr, True)
te_p, _ = build(_mte.copy(), clim, pw_te, True)
tr_n, NUM_NOPW = build(_mtr.copy(), clim, pw_tr, False)
te_n, _ = build(_mte.copy(), clim, pw_te, False)
CAT = ADV_CAT

y = tr_p["is_climate_sensitive"].values
def rep(p, mask=None):
    if mask is None: mask = np.ones(len(p), bool)
    return fold_metrics(y[mask], p[mask])

def cv_strat(make_fn, num, df, seed=42):
    X = df[num + CAT]; oof = np.zeros(len(X))
    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        p = Pipeline([("pre", make_preprocess(num, CAT)), ("clf", make_fn())])
        p.fit(X.iloc[ti], y[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:, 1]
    return oof

def cv_rolling(make_fn, num, df, n_initial=5, n_splits=5):
    X = df[num + CAT]
    oof = np.zeros(len(X)); mask = np.zeros(len(X), bool)
    years = sorted(df["death_year"].unique()); idx_end = n_initial - 1
    for _ in range(n_splits):
        if idx_end >= len(years) - 1: break
        ti = df.index[df["death_year"].isin(years[:idx_end + 1])].values
        vi = df.index[df["death_year"] == years[idx_end + 1]].values
        p = Pipeline([("pre", make_preprocess(num, CAT)), ("clf", make_fn())])
        p.fit(X.iloc[ti], y[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:, 1]; mask[vi] = True
        idx_end += 1
    return oof, mask

keep_feats = pw_tr[keep].copy()
keep_feats.index = range(len(pw_tr))

keep_feats = pw_tr[keep].copy()
keep_feats.index = range(len(pw_tr))

M = {
 "rf": lambda: RandomForestClassifier(n_estimators=1200, min_samples_leaf=2, max_features="sqrt",
                                      max_depth=20, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=42),
 "xgb": lambda: XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.03, subsample=0.7,
                             colsample_bytree=0.7, reg_lambda=1.0, scale_pos_weight=1.5,
                             n_jobs=-1, random_state=42, tree_method="hist", eval_metric="logloss"),
 "lgbm": lambda: LGBMClassifier(n_estimators=1200, num_leaves=63, learning_rate=0.03,
                                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=3,
                                lambda_l2=1.5, class_weight="balanced", n_jobs=-1,
                                random_state=42, verbose=-1),
}
rows = []; store = {}
for tag, num, df in [("without_POWER", NUM_NOPW, tr_n), ("with_POWER", TR_NUM, tr_p)]:
    for mname, fn in M.items():
        t0 = time.time()
        o = cv_strat(fn, num, df)
        ms = rep(o); cs = combined_score(ms["f1"], ms["auc"])
        orl, mrl = cv_rolling(fn, num, df)
        mr = rep(orl, mrl); cr = combined_score(mr["f1"], mr["auc"])
        store[(tag, mname)] = o
        rows.append(dict(features=tag, model=mname, strat_F1=round(ms["f1"],4),
                         strat_AUC=round(ms["auc"],4), strat_comb=round(cs,4),
                         roll_F1=round(mr["f1"],4), roll_AUC=round(mr["auc"],4),
                         roll_comb=round(cr,4), secs=round(time.time()-t0)))
        print(f"{tag:14s} {mname:5s} STRAT F1={ms['f1']:.4f} AUC={ms['auc']:.4f} c={cs:.4f} | "
              f"ROLL F1={mr['f1']:.4f} AUC={mr['auc']:.4f} c={cr:.4f} ({time.time()-t0:.0f}s)", flush=True)

res = pd.DataFrame(rows)
res.to_csv("experiments/power_ablation.csv", index=False)
print("\nDELTA (with_POWER - without_POWER):")
for mname in M:
    a = res[(res.features == "without_POWER") & (res.model == mname)].iloc[0]
    b = res[(res.features == "with_POWER") & (res.model == mname)].iloc[0]
    print(f"  {mname:5s} strat {b.strat_comb-a.strat_comb:+.4f}   rolling {b.roll_comb-a.roll_comb:+.4f}")
np.save("experiments/power_oof.npy",
        np.array([[store[("with_POWER", m)][i] for i in range(len(y))] for m in M]))
print("DONE power_ablation")
