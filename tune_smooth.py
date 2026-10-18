"""Tune location TE smooth, add lat/lon, more aggressive XGB params, stacking meta."""
import os, sys, time, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
np.random.seed(42)
sys.path.insert(0, ".")
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from src.data_loader import load_data, merge_climate
from src.features_v3 import add_features_v3, V3_NUM, V3_CAT
from src.encoding import target_encode_oof, target_encode_full, spatial_features
from src.validation import make_preprocess, fold_metrics, combined_score

os.makedirs("experiments", exist_ok=True)
tr_raw, te_raw, cl = load_data()
tr = add_features_v3(merge_climate(tr_raw, cl))
te = add_features_v3(merge_climate(te_raw, cl))

# spatial
tr = spatial_features(tr, n_clusters=8)
te = spatial_features(te, n_clusters=8)

y = tr["is_climate_sensitive"].values
NUM_BASE = V3_NUM + [f"dist_cluster_{c}" for c in range(8)] + ["dist_nearest_cluster"]
CAT = V3_CAT + ["geo_cluster"]

def test_smooth(smooth):
    # location TE
    loc_oof, loc_glob = target_encode_oof(tr, y, "location", n_splits=3, smooth=smooth)
    tr["loc_te"] = loc_oof
    _, loc_te_test, _, _ = target_encode_full(tr, pd.Series(y), te, "location", smooth=smooth)
    te["loc_te"] = loc_te_test
    # freq
    loc_freq = tr["location"].value_counts(normalize=True).to_dict()
    tr["loc_freq"] = tr["location"].map(loc_freq).fillna(0)
    te["loc_freq"] = te["location"].map(loc_freq).fillna(0)
    zone_freq = tr["zone"].value_counts(normalize=True).to_dict()
    tr["zone_freq"] = tr["zone"].map(zone_freq)
    te["zone_freq"] = te["zone"].map(zone_freq)
    
    NUM = NUM_BASE + ["loc_te", "loc_freq", "zone_freq", "latitude", "longitude"]
    X = tr[NUM + CAT]
    
    from xgboost import XGBClassifier
    def mk_xgb(): return XGBClassifier(n_estimators=800, max_depth=5, learning_rate=0.04,
                                       subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,
                                       scale_pos_weight=1.5, n_jobs=-1, random_state=42,
                                       tree_method="hist", eval_metric="logloss")
    def mk_rf():
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=1000, min_samples_leaf=2,
                                      max_features="log2", class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=42)
    
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    def oof(mk):
        o = np.zeros(len(X))
        for ti, vi in skf.split(X, y):
            p = Pipeline([("pre", make_preprocess(NUM, CAT)), ("clf", mk())])
            p.fit(X.iloc[ti], y[ti]); o[vi] = p.predict_proba(X.iloc[vi])[:, 1]
        m = fold_metrics(y, o); m["combined"] = combined_score(m["f1"], m["auc"])
        return m, o
    
    m_xgb, oof_xgb = oof(mk_xgb)
    m_rf, oof_rf = oof(mk_rf)
    c_xgb = m_xgb["combined"]; c_rf = m_rf["combined"]
    
    # stack
    Z = np.column_stack([oof_xgb, oof_rf])
    oof_stack = np.zeros(len(X))
    for ti, vi in skf.split(Z, y):
        lr = LogisticRegression(max_iter=1000, C=1.0).fit(Z[ti], y[ti])
        oof_stack[vi] = lr.predict_proba(Z[vi])[:, 1]
    m_stack = fold_metrics(y, oof_stack); m_stack["combined"] = combined_score(m_stack["f1"], m_stack["auc"])
    
    # ensemble
    oof_ens = (oof_xgb + oof_rf) / 2
    m_ens = fold_metrics(y, oof_ens); m_ens["combined"] = combined_score(m_ens["f1"], m_ens["auc"])
    
    return {"smooth": smooth, "xgb": c_xgb, "rf": c_rf, "stack": m_stack["combined"],
            "ens": m_ens["combined"], "f1_xgb": m_xgb["f1"], "f1_rf": m_rf["f1"]}

for s in [10, 20, 30, 50, 100, 200]:
    r = test_smooth(s)
    print(f"smooth={s}: xgb={r['xgb']:.4f} rf={r['rf']:.4f} stack={r['stack']:.4f} ens={r['ens']:.4f}", flush=True)