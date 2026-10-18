"""Chunk 2: monotonic constraints + age-band specialist, rolling CV (masked)."""
import sys, warnings, time
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
from src.encoding import target_encode_oof
from src.validation import make_preprocess, fold_metrics, combined_score

tr_raw, te_raw, cl = load_data()
mtr = merge_climate(tr_raw, cl)
clim = compute_location_climatology(mtr)
d = add_features_advanced(mtr, clim)
lo, _ = target_encode_oof(d, pd.Series(d["is_climate_sensitive"]), "location", n_splits=5, smooth=30)
d["loc_te"] = lo
y = d["is_climate_sensitive"].values
NUM = ADV_NUM + ["loc_te"]; CAT = ADV_CAT
X = d[NUM + CAT]

def mk_rf():
    return RandomForestClassifier(n_estimators=600, min_samples_leaf=2, max_features="sqrt",
                                  max_depth=20, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=42)

def roll(make_fn, Xdf=None):
    Xd = Xdf if Xdf is not None else X
    oof = np.zeros(len(Xd)); mask = np.zeros(len(Xd), bool)
    years = sorted(d["death_year"].unique()); e = 4
    for _ in range(5):
        if e >= len(years) - 1: break
        ti = d.index[d["death_year"].isin(years[:e + 1])].values
        vi = d.index[d["death_year"] == years[e + 1]].values
        p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", make_fn())])
        p.fit(X.iloc[ti], y[ti]); oof[vi] = p.predict_proba(X.iloc[vi])[:, 1]
        mask[vi] = True; e += 1
    return oof, mask

def show(tag, m):
    print(f"{tag:36s} F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={combined_score(m['f1'],m['auc']):.4f}", flush=True)

t0 = time.time(); o_rf, m_ = roll(mk_rf); show("BASELINE rf (rolling)", fold_metrics(y[m_], o_rf[m_]))
print("  (%.0fs)" % (time.time() - t0), flush=True)

# ---- monotonic XGB ----
MONO = {}
for c, s in [("tavg_30d", 1), ("tmax_30d", 1), ("heat_30d", 1), ("temp_range", 1),
             ("is_dry_30d", 1), ("dry_days_30d", 1), ("t2m_max_mean", 1),
             ("max_temperature", 1), ("temp_range_mean_30d", 1),
             ("rain_intensity_30d", -1), ("wet_days", -1), ("rain_sum_30d", -1),
             ("ndvi_30d", 1), ("ndvi_90d", 1), ("avg_temperature", 1)]:
    if c in NUM: MONO[c] = s
print(f"\nmonotone constraints on {len(MONO)} features (pre-transformed, no ColumnTransformer)", flush=True)

def roll_plain(make_fn, num_cols):
    """Same rolling split but on a plain numeric matrix (no ColumnTransformer),
    so XGBoost sees real column names for monotone_constraints."""
    Xn = d[num_cols].copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan).fillna(Xn.median(numeric_only=True)).astype(np.float32)
    oof = np.zeros(len(Xn)); mask = np.zeros(len(Xn), bool)
    years = sorted(d["death_year"].unique()); e = 4
    for _ in range(5):
        if e >= len(years) - 1: break
        ti = d.index[d["death_year"].isin(years[:e + 1])].values
        vi = d.index[d["death_year"] == years[e + 1]].values
        m = make_fn(); m.fit(Xn.iloc[ti], y[ti])
        oof[vi] = m.predict_proba(Xn.iloc[vi])[:, 1]; mask[vi] = True; e += 1
    return oof, mask

def mk_mono():
    return XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.03, subsample=0.7,
                         colsample_bytree=0.7, reg_lambda=1.0, scale_pos_weight=1.5,
                         n_jobs=-1, random_state=42, tree_method="hist",
                         eval_metric="logloss", monotone_constraints=dict(MONO))
t0 = time.time(); o_mono, _ = roll_plain(mk_mono, NUM)
m_mono = fold_metrics(y[m_], o_mono[m_]); show("xgb MONOTONE (rolling)", m_mono)
print("  (%.0fs)" % (time.time() - t0), flush=True)

# ---- xgb free + lgbm for blending ----
def mk_xgb():
    return XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.03, subsample=0.7,
                         colsample_bytree=0.7, reg_lambda=1.0, scale_pos_weight=1.5,
                         n_jobs=-1, random_state=42, tree_method="hist", eval_metric="logloss")
def mk_lgb():
    return LGBMClassifier(n_estimators=800, num_leaves=63, learning_rate=0.03,
                          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=3,
                          lambda_l2=1.5, class_weight="balanced", n_jobs=-1,
                          random_state=42, verbose=-1)
o_xgb, _ = roll(mk_xgb); show("xgb free (rolling)", fold_metrics(y[m_], o_xgb[m_]))
o_lgb, _ = roll(mk_lgb); show("lgbm (rolling)", fold_metrics(y[m_], o_lgb[m_]))

# ---- greedy blend on rolling OOF ----
cands = {"rf": o_rf, "xgb": o_xgb, "lgbm": o_lgb, "xgb_mono": o_mono}
def c_of(p): 
    m = fold_metrics(y[m_], p[m_]); return combined_score(m["f1"], m["auc"]), m
order = sorted(cands, key=lambda k: c_of(cands[k])[0], reverse=True)
print("\ngreedy blend order:", [(k, round(c_of(cands[k])[0], 4)) for k in order], flush=True)
ens, mem, bc = cands[order[0]].copy(), [order[0]], c_of(cands[order[0]])[0]
for k in order[1:]:
    t = (ens * len(mem) + cands[k]) / (len(mem) + 1)
    c = c_of(t)[0]
    if c > bc + 1e-4:
        ens, mem, bc = t, mem + [k], c
        print(f"  + {k} -> {c:.4f}", flush=True)
show(f"GREEDY BLEND {'+'.join(mem)}", fold_metrics(y[m_], ens[m_]))
np.save("experiments/roll_oof_final.npy", np.column_stack([cands[k] for k in cands]))
print("DONE chunk2")