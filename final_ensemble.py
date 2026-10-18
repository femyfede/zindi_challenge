"""Final ensemble: shared folds across a diverse model pool, greedy + uniform selection,
then refit on full train and write a verified submission."""
import numpy as np, pandas as pd, warnings, time, json, itertools, pickle
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier)
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from bench import load
from features2 import build

SEEDS = (42, 7, 123)
NJ = 4
tr, te = load()
y = tr["is_climate_sensitive"].values
BREAK = pd.Timestamp("2017-07-01")
AGE_BINS = [-0.01, 0, 1, 3, 6, 12, 18, 30, 45, 60, 70, 75, 85, 200]
AGE_LABELS = ["a0", "a1", "a23", "a46", "a712", "a1318", "a1930", "a3145",
              "a4660", "a6170", "a7175", "a7685", "a85p"]

# ---------------- design matrices ----------------
def spline_frame(df, cell_g=False, cell_z=False):
    d = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["deathdate"])
    post = (dt >= BREAK).astype(float)
    d["gender"] = df["gender"]; d["zone"] = df["zone"]
    d["age"] = df["age"].astype(float)
    d["year_cont"] = (dt.dt.year + (dt.dt.dayofyear - 1) / 365.25).astype(float)
    d["doy"] = dt.dt.dayofyear.astype(float)
    d["age_x_post"] = df["age"].astype(float) * post
    ab = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    cell = ab + "|" + np.where(dt >= BREAK, "post", "pre")
    d["cell"] = cell
    if cell_g:
        d["cell_g"] = cell + "|" + df["gender"].astype(str)
    if cell_z:
        d["cell_z"] = cell + "|" + df["zone"].astype(str)
    return d


def spline_spec(cell_g=False, cell_z=False, k_age=10, k_year=6, k_doy=6):
    cats = ["gender", "zone", "cell"] + (["cell_g"] if cell_g else []) + (["cell_z"] if cell_z else [])
    return ColumnTransformer([
        ("sp_age", SplineTransformer(n_knots=k_age, degree=3, include_bias=False), ["age"]),
        ("sp_agepost", SplineTransformer(n_knots=k_age, degree=3, include_bias=False), ["age_x_post"]),
        ("sp_year", SplineTransformer(n_knots=k_year, degree=3, include_bias=False), ["year_cont"]),
        ("sp_doy", SplineTransformer(n_knots=k_doy, degree=3, include_bias=False), ["doy"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])


def _flags(k):
    return {"cell_g": "g" in k, "cell_z": "z" in k}


Dtr = {k: spline_frame(tr, **_flags(k)) for k in ["g", "gz"]}
Dte = {k: spline_frame(te, **_flags(k)) for k in ["g", "gz"]}
Ttr, Tnum, Tcat = build(tr, ("demo", "time"))
Tte, _, _ = build(te, ("demo", "time"))


def tree_spec(scale=False):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))] +
                         ([("sc", StandardScaler())] if scale else [])), Tnum),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), Tcat)])


def mk(kind, **kw):
    if kind == "lin":
        return Pipeline([("sp", spline_spec(cell_g=kw.get("cell_g"), cell_z=kw.get("cell_z"))),
                         ("clf", LogisticRegression(max_iter=10000, C=kw["C"]))])
    if kind == "mlp":
        return Pipeline([("sp", tree_spec(scale=True)),
                         ("clf", MLPClassifier((64, 32), alpha=1.0, max_iter=800, random_state=kw.get("rs", 42)))])
    if kind == "rf":
        return Pipeline([("sp", tree_spec()), ("clf", RandomForestClassifier(
            n_estimators=600, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=NJ, random_state=kw.get("rs", 42)))])
    if kind == "lgbm":
        return Pipeline([("sp", tree_spec()), ("clf", LGBMClassifier(
            n_estimators=500, num_leaves=15, learning_rate=0.03, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=3.0, min_child_samples=20, n_jobs=NJ,
            random_state=kw.get("rs", 42), verbose=-1))])
    if kind == "hgb":
        return Pipeline([("sp", tree_spec()), ("clf", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.0,
            min_samples_leaf=20, random_state=kw.get("rs", 42)))])
    if kind == "cat":
        return Pipeline([("sp", tree_spec()), ("clf", CatBoostClassifier(
            iterations=600, depth=5, learning_rate=0.04, l2_leaf_reg=5, verbose=0,
            random_seed=kw.get("rs", 42), allow_writing_files=False))])
    raise ValueError(kind)


POOL = {
    "lin_C02_g":   ("lin", dict(C=0.02, cell_g=True), "g"),
    "lin_C03_g":   ("lin", dict(C=0.03, cell_g=True), "g"),
    "lin_C10_g":   ("lin", dict(C=0.10, cell_g=True), "g"),
    "lin_C03_gz":  ("lin", dict(C=0.03, cell_g=True, cell_z=True), "gz"),
    "lin_C10_gz":  ("lin", dict(C=0.10, cell_g=True, cell_z=True), "gz"),
    "lin_C003":    ("lin", dict(C=0.03), "g"),
    "mlp":         ("mlp", dict(), None),
    "rf":          ("rf", dict(), None),
    "lgbm":        ("lgbm", dict(), None),
    "hgb":         ("hgb", dict(), None),
    "cat":         ("cat", dict(), None),
}

SPLITS = [(s, a, b) for s in SEEDS for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(Ttr, y)]


def getX(key, frame_tr, frame_te, idx):
    return frame_tr.iloc[idx] if frame_tr is not None else frame_te


def oof_for(name):
    kind, kw, dkey = POOL[name]
    D = Dtr[dkey] if dkey else Ttr
    oof = np.zeros(len(D))
    for s, a, b in SPLITS:
        m = mk(kind, rs=42 + s, **kw)
        m.fit(D.iloc[a], y[a])
        oof[b] = m.predict_proba(D.iloc[b])[:, 1]
    return oof


def met(p):
    per = []
    for s in SEEDS:
        mask = np.zeros(len(y), bool)
        for ss, a, b in SPLITS:
            if ss == s:
                mask[b] = True
        f1 = f1_score(y[mask], (p[mask] >= 0.5).astype(int), zero_division=0)
        per.append((f1, roc_auc_score(y[mask], p[mask])))
    f1 = f1_score(y, (p >= 0.5).astype(int), zero_division=0)
    auc = roc_auc_score(y, p)
    sm = float(np.mean([0.6 * a + 0.4 * b for a, b in per]))
    return dict(f1=f1, auc=auc, comb=0.6 * f1 + 0.4 * auc, comb_seedmean=sm,
                pos=float((p >= 0.5).mean()))


t0 = time.time()
OOF, rows = {}, []
for name in POOL:
    t1 = time.time()
    OOF[name] = oof_for(name)
    r = met(OOF[name]); r["model"] = name; r["secs"] = round(time.time() - t1, 1)
    rows.append(r)
    print(f"{name:12s} f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f} "
          f"(seedmean {r['comb_seedmean']:.4f}) pos={r['pos']:.3f} [{r['secs']}s]", flush=True)
    pd.DataFrame(rows).to_csv("experiments/pool.csv", index=False)

ind = pd.DataFrame(rows).sort_values("comb", ascending=False)
print("\n=== INDIVIDUAL ===")
print(ind.to_string(index=False))

# ---- uniform and greedy ensembles over rank-averaged members ----
def rk(v):
    return (rankdata(v) - 0.5) / len(v)

print("\n=== SEARCHING ENSEMBLES ===")
cands = {}
names = list(OOF)
for k in range(2, 6):
    for combo in itertools.combinations(names, k):
        if "lin_" in combo[0] and sum(n.startswith("lin_") for n in combo) == k:
            pass
        p = np.mean([rk(OOF[n]) for n in combo], axis=0)
        m = met(p)
        cands[combo] = m
top = sorted(cands.items(), key=lambda kv: -kv[1]["comb_seedmean"])[:15]
for combo, m in top:
    print(f"  {m['comb_seedmean']:.4f} (pool {m['comb']:.4f}) f1={m['f1']:.4f} auc={m['auc']:.4f} "
          f"pos={m['pos']:.3f}  {'+'.join(combo)}")
print(f"\n({time.time()-t0:.0f}s)")
json.dump({k: v for k, v in top}, open("experiments/ens_top.json", "w"), indent=1, default=str)
