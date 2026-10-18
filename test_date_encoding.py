"""A/B test the date-level target encoding under stratified 5-fold CV.
Encoding is rebuilt inside every fold from training rows only."""
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
from src.encoding import target_encode_oof, target_encode_full
from src.validation import make_preprocess, fold_metrics, combined_score
from src.date_encoding import add_date_features, DATE_FEATS

tr_raw, te_raw, cl = load_data()
mtr = merge_climate(tr_raw, cl); mte = merge_climate(te_raw, cl)
clim = compute_location_climatology(mtr)

y_full = tr_raw["is_climate_sensitive"].values

def prep_train(df):
    d = add_features_advanced(df, clim)
    lo, _ = target_encode_oof(d, pd.Series(y_full), "location", n_splits=5, smooth=30)
    d["loc_te"] = lo
    return d

def prep_test(df):
    d = add_features_advanced(df, clim)
    _, lo, _, _ = target_encode_full(mtr, pd.Series(y_full), d, "location", smooth=30)
    d["loc_te"] = lo
    return d

tr = prep_train(mtr)
te = prep_test(mte)
y = tr["is_climate_sensitive"].values
NUM = ADV_NUM + ["loc_te"]
CAT = ADV_CAT
ALLNUM = NUM + DATE_FEATS
print("base features:", len(NUM), "| with date feats:", len(ALLNUM), flush=True)

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

def run(make_fn, num, use_date, seed=42):
    """Stratified 5-fold; date features rebuilt inside each fold from train rows only."""
    oof = np.zeros(len(tr))
    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y):
        fit_df = tr.iloc[ti]; y_fit = y[ti]
        tgt = tr.iloc[vi]
        if use_date:
            # rebuild the exact same column set for fit and val, index-aligned
            base_num = [c for c in num if c not in DATE_FEATS]
            fit_all = fit_df[base_num + CAT].copy()
            val_all = tgt[base_num + CAT].copy()
            fit_all = pd.concat([fit_all, add_date_features(fit_df, fit_df, y_fit)], axis=1)
            val_all = pd.concat([val_all, add_date_features(tgt, fit_df, y_fit)], axis=1)
            X_fit, X_val = fit_all, val_all
        else:
            X_fit = fit_df[num + CAT]
            X_val = tgt[num + CAT]
        p = Pipeline([("pre", make_preprocess(num, CAT)), ("clf", make_fn())])
        p.fit(X_fit, y_fit); oof[vi] = p.predict_proba(X_val)[:, 1]
    return oof

store = {}
rows = []
for mname, fn in M.items():
    for tag, num, ud in [("no_date", NUM, False), ("with_date", ALLNUM, True)]:
        t0 = time.time()
        o = run(fn, num, ud)
        m = fold_metrics(y, o); c = combined_score(m["f1"], m["auc"])
        store[(mname, tag)] = o
        rows.append(dict(model=mname, feats=tag, F1=round(m["f1"], 4), AUC=round(m["auc"], 4),
                         comb=round(c, 4), prec=round(m["precision"], 4),
                         rec=round(m["recall"], 4), secs=round(time.time() - t0)))
        print(f"{mname:5s} {tag:10s} F1={m['f1']:.4f} AUC={m['auc']:.4f} comb={c:.4f} "
              f"prec={m['precision']:.4f} rec={m['recall']:.4f} ({time.time()-t0:.0f}s)", flush=True)

print("\nDELTA (with_date - no_date):")
for mname in M:
    a = [r for r in rows if r['model'] == mname and r['feats'] == 'no_date'][0]
    b = [r for r in rows if r['model'] == mname and r['feats'] == 'with_date'][0]
    print(f"  {mname:5s} F1 {b['F1']-a['F1']:+.4f}  AUC {b['AUC']-a['AUC']:+.4f}  comb {b['comb']-a['comb']:+.4f}")

# ensemble of with_date models
Z = np.column_stack([store[(m, "with_date")] for m in M])
for k in (1, 2, 3):
    ens = Z[:, :k].mean(axis=1)
    m = fold_metrics(y, ens); c = combined_score(m["f1"], m["auc"])
    print(f"ENSEMBLE with_date top{k} ({'+'.join(list(M)[:k])}): F1={m['f1']:.4f} "
          f"AUC={m['auc']:.4f} comb={c:.4f}")

pd.DataFrame(rows).to_csv("experiments/date_encoding_ablation.csv", index=False)
np.save("experiments/date_oof.npy", Z)
print("DONE date_ablation")