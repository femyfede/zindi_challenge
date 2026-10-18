"""Round 7 — the genuinely untried levers.

A) Ensemble over RANDOMISED age-bin definitions (each gives a different `cell`
   partition => decorrelated errors; averaging should beat any single binning).
B) Small-weight blend with tree models (trees have lower AUC but different bias).
C) Recalibration (isotonic / Platt) of the final score - affects F1 only.
D) Separate models per period (pre/post) instead of a shared spline + cell.
"""
import numpy as np, pandas as pd, warnings, time, itertools
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as LR2
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from bench import load
from features2 import build

BREAK = pd.Timestamp("2017-07-01")
SEEDS = (42, 7, 123, 2024, 99)
rng = np.random.default_rng(0)
tr, te = load()
y = tr["is_climate_sensitive"].values


def oof_scores(make_pipe, D, seeds=SEEDS):
    outs = []
    for s in seeds:
        o = np.zeros(len(D))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(D, y):
            m = make_pipe(); m.fit(D.iloc[a], y[a]); o[b] = m.predict_proba(D.iloc[b])[:, 1]
        outs.append(o)
    return np.array(outs)


def met(o):
    per = [(f1_score(y, (x >= .5).astype(int)), roc_auc_score(y, x)) for x in o]
    m = o.mean(axis=0)
    f1 = f1_score(y, (m >= .5).astype(int)); auc = roc_auc_score(y, m)
    return dict(f1=f1, auc=auc, comb=0.6 * f1 + 0.4 * auc,
                sm=float(np.mean([0.6 * a + 0.4 * b for a, b in per])), pos=float((m >= .5).mean()))


def lin_frame(bins, gender=True):
    d = pd.DataFrame(index=tr.index)
    dt = pd.to_datetime(tr["deathdate"])
    post = (dt >= BREAK).astype(float)
    d["gender"] = tr["gender"]; d["zone"] = tr["zone"]
    d["age"] = tr["age"].astype(float)
    d["year_cont"] = (dt.dt.year + (dt.dt.dayofyear - 1) / 365.25).astype(float)
    d["doy"] = dt.dt.dayofyear.astype(float)
    d["age_x_post"] = tr["age"].astype(float) * post
    ab = pd.cut(tr["age"], bins).astype(str)
    cell = ab + "|" + np.where(dt >= BREAK, "post", "pre")
    d["cell"] = cell
    cats = ["gender", "zone", "cell"]
    if gender:
        d["cell_g"] = cell + "|" + tr["gender"].astype(str); cats.append("cell_g")
    return d, cats


def lin_pipe(cats, C):
    spec = ColumnTransformer([
        ("sp_age", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age"]),
        ("sp_agepost", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age_x_post"]),
        ("sp_year", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["year_cont"]),
        ("sp_doy", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["doy"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
    return Pipeline([("sp", spec), ("clf", LogisticRegression(max_iter=10000, C=C))])


B20 = [-0.01, 0, 1, 2, 3, 4, 5, 8, 12, 18, 25, 35, 45, 55, 62, 68, 73, 78, 85, 200]

print("=== A) ensemble over randomised age-bin definitions ===")
POOL = [B20]
for _ in range(11):
    inner = sorted(set([0, 1, 2, 3, 4, 5] + [int(v) for v in rng.integers(6, 20, 8)] +
                      [int(v) for v in rng.integers(20, 95, 5)]))
    POOL.append([-0.01] + inner + [200])

BAG = {}
for i, b in enumerate(POOL):
    D, cats = lin_frame(b, True)
    o = oof_scores(lambda: lin_pipe(cats, 0.02), D)
    BAG[i] = o
    if i == 0:
        r0 = met(o); print(f"  pool[0] (B20, reference) f1={r0['f1']:.4f} auc={r0['auc']:.4f} comb={r0['comb']:.4f}")
    if i % 4 == 0:
        print(f"   built pool[{i}]", flush=True)

for k in [3, 5, 8, 12]:
    o = np.mean([BAG[i] for i in range(k)], axis=0)
    r = met(o)
    print(f"  BAG first {k:2d} binnings: f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f} "
          f"sm={r['sm']:.4f} pos={r['pos']:.3f}", flush=True)

BEST_BAG = np.mean([BAG[i] for i in range(12)], axis=0)

print("\n=== B) tree-blend weight sweep on top of the bagged linear score ===")
Ttr, Tnum, Tcat = build(tr, ("demo", "time"))


def tree_pipe(clf):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), Tnum),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), Tcat)])
    return Pipeline([("pre", pre), ("clf", clf)])


TREE = {"lgbm": lambda: LGBMClassifier(n_estimators=500, num_leaves=15, learning_rate=0.03,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                       reg_lambda=3.0, min_child_samples=20, n_jobs=4,
                                       random_state=42, verbose=-1),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04,
                                                      max_leaf_nodes=15, l2_regularization=1.0,
                                                      min_samples_leaf=20, random_state=42),
        "xgb": lambda: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
                                     subsample=0.8, colsample_bytree=0.7, reg_lambda=3.0,
                                     min_child_weight=5, tree_method="hist", n_jobs=4,
                                     random_state=42, eval_metric="logloss")}
TOOF = {}
for k, f in TREE.items():
    TOOF[k] = oof_scores(lambda f=f: tree_pipe(f()), Ttr, seeds=(42, 7, 123))
    r = met(TOOF[k]); print(f"  {k} alone: f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f}")

for k, to in TOOF.items():
    for w in [0.05, 0.1, 0.15, 0.2, 0.3]:
        o = (1 - w) * BEST_BAG + w * to
        r = met(o)
        print(f"  bag + {w:.2f}*{k:5s}: f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f} "
              f"sm={r['sm']:.4f} pos={r['pos']:.3f}", flush=True)

print("\n=== C) recalibration of the bagged score ===")
for cal in ["isotonic", "platt"]:
    cal_o = np.zeros_like(BEST_BAG)
    for i in range(BEST_BAG.shape[0]):
        if cal == "isotonic":
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        else:
            ir = LR2(max_iter=1000)
        ir.fit(BEST_BAG[i], y)
        cal_o[i] = ir.predict(BEST_BAG[i])
    r = met(cal_o)
    print(f"  {cal:9s}: f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f} sm={r['sm']:.4f} pos={r['pos']:.3f}")

print("\n=== D) separate model per period ===")
dt = pd.to_datetime(tr["deathdate"])
pre_m = (dt < BREAK).values
for name, mask in [("pre", pre_m), ("post", ~pre_m)]:
    D, cats = lin_frame(B20, True)
    yy = y[mask]
    o = np.zeros(mask.sum())
    for s in (42, 7, 123):
        oo = np.zeros(mask.sum())
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(D[mask], yy):
            m = lin_pipe(cats, 0.02); m.fit(D[mask].iloc[a], yy[a]); oo[b] = m.predict_proba(D[mask].iloc[b])[:, 1]
        o += oo / 3
    print(f"  {name:5s} n={mask.sum():4d} pos={yy.mean():.3f} f1={f1_score(yy,(o>=.5).astype(int)):.4f} "
          f"auc={roc_auc_score(yy,o):.4f}")
