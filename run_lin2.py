"""Sweep 2: richer interaction structure + shrinkage sweep for the spline-logistic model."""
import numpy as np, pandas as pd, warnings, time, itertools, json, sys
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from bench import load

BREAK = pd.Timestamp("2017-07-01")
tr, te = load()
y = tr["is_climate_sensitive"].values
AGE_BINS = [-0.01, 0, 1, 3, 6, 12, 18, 30, 45, 60, 70, 75, 85, 200]
AGE_LABELS = ["a0", "a1", "a23", "a46", "a712", "a1318", "a1930", "a3145",
              "a4660", "a6170", "a7175", "a7685", "a85p"]
CLIM = ["rain_sum_7d", "rain_sum_30d", "rain_sum_90d", "rain_days_30d",
        "tavg_7d", "tavg_30d", "tavg_90d", "tmax_30d", "tmin_30d",
        "ndvi_30d", "ndvi_90d", "temp_range_mean_30d"]


def frame(df, cfg):
    d = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["deathdate"])
    post = (dt >= BREAK).astype(float)
    d["gender"] = df["gender"]; d["zone"] = df["zone"]
    d["age"] = df["age"].astype(float)
    d["year_cont"] = (dt.dt.year + (dt.dt.dayofyear - 1) / 365.25).astype(float)
    d["doy"] = dt.dt.dayofyear.astype(float)
    d["age_x_post"] = df["age"].astype(float) * post
    ab = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    per = np.where(dt >= BREAK, "post", "pre")
    cell = ab + "|" + per
    d["cell"] = cell
    if cfg.get("cell_g"):
        d["cell_g"] = cell + "|" + df["gender"].astype(str)
    if cfg.get("cell_z"):
        d["cell_z"] = cell + "|" + df["zone"].astype(str)
    if cfg.get("cell_season"):
        sea = dt.dt.month.map(lambda m: "dry" if m in (12, 1, 2, 6, 7, 8) else "wet").astype(str)
        d["cell_sea"] = cell + "|" + sea
        d["season"] = sea
    if cfg.get("age_season"):
        sea = dt.dt.month.map(lambda m: "dry" if m in (12, 1, 2, 6, 7, 8) else "wet").astype(str)
        d["ageband_sea"] = ab + "|" + sea
    if cfg.get("yearchunk"):
        d["ychunk"] = (dt.dt.year // cfg["yearchunk"]).astype(str)
    for c in cfg.get("clim", []):
        d[c] = df[c].astype(float)
    return d


def spec(cfg):
    t = [("sp_age", SplineTransformer(n_knots=cfg["k_age"], degree=3, include_bias=False), ["age"]),
         ("sp_agepost", SplineTransformer(n_knots=cfg["k_post_age"], degree=3, include_bias=False), ["age_x_post"]),
         ("sp_year", SplineTransformer(n_knots=cfg["k_year"], degree=3, include_bias=False), ["year_cont"])]
    if cfg.get("k_doy", 0):
        t.append(("sp_doy", SplineTransformer(n_knots=cfg["k_doy"], degree=3, include_bias=False), ["doy"]))
    cats = ["gender", "zone", "cell"]
    for k in ("cell_g", "cell_z", "cell_season", "season", "ageband_sea", "ychunk"):
        if cfg.get(k):
            cats.append(k)
    nums = [c for c in cfg.get("clim", [])]
    if nums:
        t.append(("num", StandardScaler(), nums))
    t.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
    return ColumnTransformer(t)


def build_pipe(cfg, C):
    return Pipeline([("sp", spec(cfg)), ("clf", LogisticRegression(max_iter=10000, C=C))])


def rep_cv(cfg, seeds=(42, 7, 123), n_splits=5):
    D = frame(tr, cfg)
    oofs = []
    for s in seeds:
        oof = np.zeros(len(D))
        for a, b in StratifiedKFold(n_splits, shuffle=True, random_state=s).split(D, y):
            p = build_pipe(cfg, cfg["C"]); p.fit(D.iloc[a], y[a]); oof[b] = p.predict_proba(D.iloc[b])[:, 1]
        oofs.append(oof)
    per = [(f1_score(y, (o >= 0.5).astype(int), zero_division=0), roc_auc_score(y, o)) for o in oofs]
    m = np.mean(oofs, axis=0)
    f1 = f1_score(y, (m >= 0.5).astype(int), zero_division=0); auc = roc_auc_score(y, m)
    return dict(f1=f1, auc=auc, comb=0.6 * f1 + 0.4 * auc,
                comb_seedmean=float(np.mean([0.6 * a + 0.4 * b for a, b in per])),
                comb_seedmin=float(np.min([0.6 * a + 0.4 * b for a, b in per])),
                pos=float((m >= 0.5).mean()))


BASE = dict(k_age=10, k_post_age=10, k_year=6, k_doy=6)
CFGS = []
for C in [0.01, 0.02, 0.03, 0.05, 0.1]:
    for extra in [{}, {"cell_g": 1}, {"cell_z": 1}, {"cell_g": 1, "cell_z": 1},
                  {"cell_season": 1}, {"age_season": 1},
                  {"cell_g": 1, "cell_z": 1, "cell_season": 1},
                  {"cell_g": 1, "cell_z": 1, "clim": CLIM}]:
        CFGS.append({**BASE, **extra, "C": C})

rows = []
t0 = time.time()
for cfg in CFGS:
    tag = ",".join(k for k in ("cell_g", "cell_z", "cell_season", "age_season") if cfg.get(k)) or "base"
    tag += "|clim" if cfg.get("clim") else ""
    try:
        r = rep_cv(cfg)
    except Exception as e:
        print("FAIL", tag, cfg["C"], repr(e)[:90], flush=True); continue
    rows.append({**r, **cfg, "tag": tag, "has_clim": bool(cfg.get("clim"))})
    print(f"{tag:42s} C={cfg['C']:<5} f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f} "
          f"(seedmean {r['comb_seedmean']:.4f}) pos={r['pos']:.3f}", flush=True)
    pd.DataFrame([{k: v for k, v in r_.items() if k != "clim"} | {"tag": r_["tag"], "has_clim": r_["has_clim"]} for r_ in rows]) \
        .to_csv("experiments/lin2.csv", index=False)

res = pd.DataFrame([{k: v for k, v in r.items() if k != "clim"} for r in rows]).sort_values("comb", ascending=False)
res.to_csv("experiments/lin2.csv", index=False)
print(f"\n({time.time()-t0:.0f}s)\nTOP 15:")
print(res.head(15).to_string(index=False))
