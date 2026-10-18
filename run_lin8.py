"""Round 8 — squeeze AUC. 3 seeds throughout so members are directly blendable.

Targets, in order of expected value:
  1. weight sweep: bagged-linear + tree blend (trees add decorrelated ranking signal)
  2. scipy-optimised blend weights on the combined score
  3. self-training / pseudo-labelling on the test features
  4. bagged bins + tree blend combined
"""
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from bench import load
from features2 import build

BREAK = pd.Timestamp("2017-07-01")
SEEDS = (42, 7, 123)
rng = np.random.default_rng(0)
tr, te = load()
y = tr["is_climate_sensitive"].values


def met(o):
    per = [(f1_score(y, (x >= .5).astype(int)), roc_auc_score(y, x)) for x in o]
    m = o.mean(axis=0)
    f1 = f1_score(y, (m >= .5).astype(int)); auc = roc_auc_score(y, m)
    return dict(f1=f1, auc=auc, comb=0.6 * f1 + 0.4 * auc,
                sm=float(np.mean([0.6 * a + 0.4 * b for a, b in per])), pos=float((m >= .5).mean()))


def lin_frame(bins):
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
    d["cell"] = cell; d["cell_g"] = cell + "|" + tr["gender"].astype(str)
    return d, ["gender", "zone", "cell", "cell_g"]


def lin_pipe(cats, C):
    spec = ColumnTransformer([
        ("sp_age", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age"]),
        ("sp_agepost", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age_x_post"]),
        ("sp_year", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["year_cont"]),
        ("sp_doy", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["doy"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
    return Pipeline([("sp", spec), ("clf", LogisticRegression(max_iter=10000, C=C))])


def oof(make_pipe, D, seeds=SEEDS):
    outs = []
    for s in seeds:
        o = np.zeros(len(D))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(D, y):
            m = make_pipe(); m.fit(D.iloc[a], y[a]); o[b] = m.predict_proba(D.iloc[b])[:, 1]
        outs.append(o)
    return np.array(outs)


B20 = [-0.01, 0, 1, 2, 3, 4, 5, 8, 12, 18, 25, 35, 45, 55, 62, 68, 73, 78, 85, 200]
POOL = [B20]
for _ in range(11):
    inner = sorted(set([0, 1, 2, 3, 4, 5] + [int(v) for v in rng.integers(6, 20, 8)] +
                      [int(v) for v in rng.integers(20, 95, 5)]))
    POOL.append([-0.01] + inner + [200])

print("=== bagged-bins linear ===", flush=True)
BAGOOF = []
for i, b in enumerate(POOL):
    D, cats = lin_frame(b)
    BAGOOF.append(oof(lambda cats=cats: lin_pipe(cats, 0.02), D))
LIN = np.mean(BAGOOF, axis=0)
r = met(LIN); print(f"  BAG12: f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f} sm={r['sm']:.4f}", flush=True)
r = met(BAGOOF[0]); print(f"  B20  : f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f}", flush=True)

Ttr, Tnum, Tcat = build(tr, ("demo", "time"))
def tree_pipe(clf):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), Tnum),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), Tcat)])
    return Pipeline([("pre", pre), ("clf", clf)])

print("\n=== trees ===", flush=True)
TP = {
 "lgbm": lambda: LGBMClassifier(n_estimators=500, num_leaves=15, learning_rate=0.03, subsample=0.8,
                                subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0,
                                min_child_samples=20, n_jobs=4, random_state=42, verbose=-1),
 "hgb": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                               l2_regularization=1.0, min_samples_leaf=20, random_state=42),
 "xgb": lambda: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04, subsample=0.8,
                              colsample_bytree=0.7, reg_lambda=3.0, min_child_weight=5,
                              tree_method="hist", n_jobs=4, random_state=42, eval_metric="logloss"),
 "cat": lambda: CatBoostClassifier(iterations=600, depth=5, learning_rate=0.04, l2_leaf_reg=5,
                                   verbose=0, random_seed=42, allow_writing_files=False),
 "et": lambda: ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                                     class_weight="balanced_subsample", n_jobs=4, random_state=42),
 "rf": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=3, max_features="sqrt",
                                       class_weight="balanced_subsample", n_jobs=4, random_state=42),
}
TO = {}
for k, f in TP.items():
    TO[k] = oof(lambda f=f: tree_pipe(f()), Ttr)
    r = met(TO[k]); print(f"  {k:5s} f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f}", flush=True)

print("\n=== blend weight sweep (linear bag + tree) ===")
best_w, best_s = 0.0, -1
for k, to in TO.items():
    for w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        r = met((1 - w) * LIN + w * to)
        star = ""
        if r["sm"] > best_s:
            best_s, best_w, best_k = r["sm"], w, k; star = "  <--"
        print(f"  LIN + {w:.2f}*{k:5s}: f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f} "
              f"sm={r['sm']:.4f} pos={r['pos']:.3f}{star}", flush=True)
print(f"  best single-tree blend: {best_k} w={best_w} sm={best_s:.4f}")

print("\n=== multi-tree equal-weight blend added to LIN ===")
for k in [2, 3, 4]:
    keys = list(TO)[:k]
    avg = np.mean([TO[x] for x in keys], axis=0)
    for w in [0.15, 0.25, 0.35]:
        r = met((1 - w) * LIN + w * avg)
        print(f"  LIN + {w:.2f}*avg({'+'.join(keys)}): f1={r['f1']:.4f} auc={r['auc']:.4f} "
              f"comb={r['comb']:.4f} sm={r['sm']:.4f} pos={r['pos']:.3f}", flush=True)

print("\n=== scipy-optimised weights (lin + 6 trees) on combined score ===")
names = ["LIN"] + list(TO)
M = np.array([o.mean(axis=0) for o in [LIN] + [TO[k] for k in TO]])


def loss(w):
    w = np.abs(w); w = w / w.sum()
    p = w @ M
    return -(0.6 * f1_score(y, (p >= .5).astype(int)) + 0.4 * roc_auc_score(y, p))


best = None
for _ in range(12):
    w0 = rng.random(len(names)) ** 2
    r = minimize(loss, w0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6})
    if best is None or r.fun < best.fun:
        best = r
w = np.abs(best.x); w = w / w.sum()
p = w @ M
r = met(np.array([p] * 3))
print("  weights: " + ", ".join(f"{n}={wi:.3f}" for n, wi in zip(names, w)))
print(f"  optimised (in-sample on OOF, so optimistic): f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f}")
np.save("experiments/blend_w.npy", w)
