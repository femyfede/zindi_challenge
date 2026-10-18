"""Tune a properly-specified smooth additive model (splines) + interaction search.
Linear models are cheap here, so we can sweep aggressively with repeated CV."""
import numpy as np, pandas as pd, warnings, itertools, time
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


def frame(df, cfg):
    d = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["deathdate"])
    is_post = (dt >= BREAK).astype(float)
    d["gender"] = df["gender"]; d["zone"] = df["zone"]
    d["age"] = df["age"].astype(float)
    d["year_cont"] = (dt.dt.year + (dt.dt.dayofyear - 1) / 365.25).astype(float)
    d["doy"] = dt.dt.dayofyear.astype(float)
    d["is_post"] = is_post
    d["age_x_post"] = df["age"].astype(float) * is_post
    if cfg.get("cell"):
        ab = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
        per = np.where(dt >= BREAK, "post", "pre")
        d["cell_ap"] = ab + "|" + per
        if cfg.get("cell_gz"):
            d["cell_ap_g"] = d["cell_ap"] + "|" + df["gender"].astype(str)
            d["cell_ap_z"] = d["cell_ap"] + "|" + df["zone"].astype(str)
        if cfg.get("cell_post_only"):
            d["cell_p"] = per
    if cfg.get("yearchunk"):
        d["ychunk"] = (dt.dt.year // cfg["yearchunk"]).astype(str)
    if cfg.get("month"):
        d["month"] = dt.dt.month.astype(str)
    for c in cfg.get("extra", []):
        d[c] = df[c].astype(float)
    return d


def spec(cfg):
    tr_ = []
    sp_age = ("sp_age", SplineTransformer(n_knots=cfg["k_age"], degree=3, include_bias=False), ["age"])
    tr_.append(sp_age)
    if cfg.get("k_post_age", 0):
        tr_.append(("sp_agepost", SplineTransformer(n_knots=cfg["k_post_age"], degree=3, include_bias=False), ["age_x_post"]))
    tr_.append(("sp_year", SplineTransformer(n_knots=cfg["k_year"], degree=3, include_bias=False), ["year_cont"]))
    if cfg.get("k_doy", 0):
        tr_.append(("sp_doy", SplineTransformer(n_knots=cfg["k_doy"], degree=3, include_bias=False), ["doy"]))
    cats = ["gender", "zone"]
    if cfg.get("cell"):
        cats.append("cell_ap")
    if cfg.get("cell_gz"):
        cats += ["cell_ap_g", "cell_ap_z"]
    if cfg.get("cell_post_only"):
        cats.append("cell_p")
    if cfg.get("yearchunk"):
        cats.append("ychunk")
    if cfg.get("month"):
        cats.append("month")
    passthru = [c for c in cfg.get("extra", [])]
    tr_.append(("num", StandardScaler(), passthru))
    tr_.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
    return ColumnTransformer(tr_)


def build_pipe(cfg, C):
    return Pipeline([("sp", spec(cfg)),
                     ("clf", LogisticRegression(max_iter=8000, C=C))])


def rep_cv(cfg, C, seeds=(42, 7, 123), n_splits=5):
    D = frame(tr, cfg)
    f1s, aucs, oofs = [], [], []
    for s in seeds:
        oof = np.zeros(len(D))
        for a, b in StratifiedKFold(n_splits, shuffle=True, random_state=s).split(D, y):
            p = build_pipe(cfg, C); p.fit(D.iloc[a], y[a]); oof[b] = p.predict_proba(D.iloc[b])[:, 1]
        f1s.append(f1_score(y, (oof >= 0.5).astype(int), zero_division=0))
        aucs.append(roc_auc_score(y, oof)); oofs.append(oof)
    oof = np.mean(oofs, axis=0)
    f1 = f1_score(y, (oof >= 0.5).astype(int), zero_division=0); auc = roc_auc_score(y, oof)
    return dict(f1=f1, auc=auc, comb=0.6 * f1 + 0.4 * auc,
                f1_m=np.mean(f1s), auc_m=np.mean(aucs), comb_m=np.mean([0.6 * a + 0.4 * b for a, b in zip(f1s, aucs)]),
                pos=float((oof >= 0.5).mean()))


CFGS = []
for k_age in [5, 7, 10]:
    for k_year in [4, 6]:
        for cell in [False, True]:
            for C in [0.03, 0.1, 0.3, 1.0]:
                CFGS.append(dict(k_age=k_age, k_year=k_year, cell=cell, C=C,
                                 k_post_age=k_age, k_doy=6))
rows = []
t0 = time.time()
for cfg in CFGS:
    try:
        r = rep_cv(cfg, cfg["C"])
    except Exception as e:
        print("FAIL", cfg, repr(e)[:100]); continue
    r.update({k: cfg[k] for k in ("k_age", "k_year", "cell", "C")})
    rows.append(r)
    print(f"k_age={cfg['k_age']} k_year={cfg['k_year']} cell={int(cfg['cell'])} C={cfg['C']:<5} "
          f"f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f} (mean-of-seeds comb={r['comb_m']:.4f}) "
          f"pos={r['pos']:.3f}", flush=True)

res = pd.DataFrame(rows).sort_values("comb", ascending=False)
res.to_csv("experiments/lin_tune.csv", index=False)
print(f"\n({time.time()-t0:.0f}s)\nTOP 12:")
print(res.head(12).to_string(index=False))
