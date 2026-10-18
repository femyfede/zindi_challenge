"""Sweep 3: finer age cells, period-specific year/season terms, deeper crosses."""
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from bench import load

BREAK = pd.Timestamp("2017-07-01")
SEEDS = (42, 7, 123)
tr, te = load()
y = tr["is_climate_sensitive"].values

BINS = {
 "13": [-0.01, 0, 1, 3, 6, 12, 18, 30, 45, 60, 70, 75, 85, 200],
 "20": [-0.01, 0, 1, 2, 3, 4, 5, 8, 12, 18, 25, 35, 45, 55, 62, 68, 73, 78, 85, 200],
 "8":  [-0.01, 0, 2, 6, 18, 45, 65, 78, 200],
 "6":  [-0.01, 0, 1, 6, 18, 45, 78, 200],
}


def frame(df, cfg):
    d = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["deathdate"])
    post = (dt >= BREAK).astype(float)
    d["gender"] = df["gender"]; d["zone"] = df["zone"]
    d["age"] = df["age"].astype(float)
    d["year_cont"] = (dt.dt.year + (dt.dt.dayofyear - 1) / 365.25).astype(float)
    d["doy"] = dt.dt.dayofyear.astype(float)
    d["age_x_post"] = df["age"].astype(float) * post
    d["year_x_post"] = d["year_cont"] * post
    d["doy_x_post"] = d["doy"] * post
    bins = BINS[cfg["bins"]]
    ab = pd.cut(df["age"], bins).astype(str)
    per = np.where(dt >= BREAK, "post", "pre")
    cell = ab + "|" + per
    d["cell"] = cell
    if cfg.get("gz"):
        d["cell_g"] = cell + "|" + df["gender"].astype(str)
    if cfg.get("z"):
        d["cell_z"] = cell + "|" + df["zone"].astype(str)
    if cfg.get("gz_full"):
        d["cell_gz"] = cell + "|" + df["gender"].astype(str) + "|" + df["zone"].astype(str)
    if cfg.get("post_only"):
        d["per"] = per
    if cfg.get("season"):
        sea = dt.dt.month.map(lambda m: "d" if m in (12, 1, 2, 6, 7, 8) else "w").astype(str)
        d["sea"] = sea
        if cfg.get("cell_sea"):
            d["cell_sea"] = cell + "|" + sea
    return d


def spec(cfg):
    t = [("sp_age", SplineTransformer(n_knots=cfg["k_age"], degree=3, include_bias=False), ["age"]),
         ("sp_agepost", SplineTransformer(n_knots=cfg["k_age"], degree=3, include_bias=False), ["age_x_post"]),
         ("sp_year", SplineTransformer(n_knots=cfg["k_year"], degree=3, include_bias=False), ["year_cont"])]
    if cfg.get("year_post"):
        t.append(("sp_ypost", SplineTransformer(n_knots=cfg["k_year"], degree=3, include_bias=False), ["year_x_post"]))
    if cfg.get("doy_post"):
        t.append(("sp_dpost", SplineTransformer(n_knots=5, degree=3, include_bias=False), ["doy_x_post"]))
    else:
        t.append(("sp_doy", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["doy"]))
    cats = ["gender", "zone", "cell"]
    for k in ("cell_g", "cell_z", "cell_gz", "per", "sea", "cell_sea"):
        if cfg.get(k):
            cats.append(k)
    t.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
    return ColumnTransformer(t)


def rep_cv(cfg):
    D = frame(tr, cfg)
    oofs = []
    for s in SEEDS:
        oof = np.zeros(len(D))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(D, y):
            p = Pipeline([("sp", spec(cfg)), ("clf", LogisticRegression(max_iter=10000, C=cfg["C"]))])
            p.fit(D.iloc[a], y[a]); oof[b] = p.predict_proba(D.iloc[b])[:, 1]
        oofs.append(oof)
    per = [(f1_score(y, (o >= 0.5).astype(int), zero_division=0), roc_auc_score(y, o)) for o in oofs]
    m = np.mean(oofs, axis=0)
    f1 = f1_score(y, (m >= 0.5).astype(int), zero_division=0); auc = roc_auc_score(y, m)
    return dict(f1=f1, auc=auc, comb=0.6 * f1 + 0.4 * auc,
                sm=float(np.mean([0.6 * a + 0.4 * b for a, b in per])),
                pos=float((m >= 0.5).mean()))


BASE = dict(k_age=10, k_year=6, gz=True)
VARIANTS = [
    {}, {"year_post": 1}, {"doy_post": 1}, {"year_post": 1, "doy_post": 1},
    {"z": 1}, {"gz_full": 1}, {"post_only": 1}, {"season": 1}, {"cell_sea": 1, "season": 1},
    {"year_post": 1, "gz_full": 1}, {"year_post": 1, "season": 1},
]
rows = []
t0 = time.time()
for bins in ["13", "20", "8", "6"]:
    for v in VARIANTS:
        for C in [0.02, 0.03, 0.06]:
            cfg = {**BASE, **v, "bins": bins, "C": C}
            tag = ",".join(v.keys()) or "base"
            try:
                r = rep_cv(cfg)
            except Exception as e:
                print("FAIL", bins, tag, C, repr(e)[:80], flush=True); continue
            rows.append({**r, **v, "bins": bins, "C": C, "tag": tag})
            print(f"bins={bins:>2s} C={C:<5} {tag:22s} f1={r['f1']:.4f} auc={r['auc']:.4f} "
                  f"comb={r['comb']:.4f} sm={r['sm']:.4f} pos={r['pos']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv("experiments/lin3.csv", index=False)

res = pd.DataFrame(rows).sort_values("sm", ascending=False)
res.to_csv("experiments/lin3.csv", index=False)
print(f"\n({time.time()-t0:.0f}s)\nTOP 20 by seed-mean combined:")
print(res.head(20).to_string(index=False))
