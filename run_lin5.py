"""Final push: age x season cells, explicit climate interactions, and tighter regularisation."""
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
B20 = [-0.01, 0, 1, 2, 3, 4, 5, 8, 12, 18, 25, 35, 45, 55, 62, 68, 73, 78, 85, 200]
B8 = [-0.01, 0, 2, 6, 18, 45, 65, 78, 200]


def frame(df, cfg):
    d = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["deathdate"])
    post = (dt >= BREAK).astype(float)
    d["gender"] = df["gender"]; d["zone"] = df["zone"]
    d["age"] = df["age"].astype(float)
    d["year_cont"] = (dt.dt.year + (dt.dt.dayofyear - 1) / 365.25).astype(float)
    d["doy"] = dt.dt.dayofyear.astype(float)
    d["age_x_post"] = df["age"].astype(float) * post
    per = np.where(dt >= BREAK, "post", "pre")
    ab20 = pd.cut(df["age"], B20).astype(str)
    d["cell"] = ab20 + "|" + per
    d["cell_g"] = d["cell"] + "|" + df["gender"].astype(str)
    if cfg.get("mcell"):
        ab = pd.cut(df["age"], B8).astype(str)
        mo = dt.dt.month.astype(str)
        d["month_cell"] = ab + "|" + mo
        d["mo"] = mo
        d["month_cell_p"] = ab + "|" + per + "|" + mo
    if cfg.get("doy_int"):
        sp = SplineTransformer(n_knots=5, degree=3, include_bias=False).fit_transform(
            df[["age"]].values)
        doysp = SplineTransformer(n_knots=6, degree=3, include_bias=False).fit_transform(
            dt.dt.dayofyear.values.reshape(-1, 1).astype(float))
        prod = sp * doysp[:, [0]]
        for j in range(prod.shape[1]):
            d[f"age_doy_{j}"] = prod[:, j]
    for c in cfg.get("clim", []):
        d[c] = df[c].astype(float)
    if cfg.get("age_clim"):
        for c in cfg.get("clim", [])[:4]:
            d[f"age_x_{c}"] = df["age"].astype(float) * df[c].astype(float)
    return d


def spec(cfg):
    t = [("sp_age", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age"]),
         ("sp_agepost", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age_x_post"]),
         ("sp_year", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["year_cont"]),
         ("sp_doy", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["doy"])]
    cats = ["gender", "zone", "cell", "cell_g"]
    for k in ("month_cell", "mo", "month_cell_p"):
        if cfg.get(k):
            cats.append(k)
    extra = [c for c in dcols if c.startswith("age_doy_")]
    clim = list(cfg.get("clim", [])) + [f"age_x_{c}" for c in cfg.get("clim", [])[:4]] if cfg.get("age_clim") else list(cfg.get("clim", []))
    if clim:
        t.append(("clim", StandardScaler(), clim))
    t.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
    return ColumnTransformer(t)


dcols = []
BASE = {"k": 10}


def run(cfg, C):
    global dcols
    D = frame(tr, cfg); dcols = list(D.columns)
    outs = []
    for s in SEEDS:
        o = np.zeros(len(D))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(D, y):
            m = Pipeline([("sp", spec(cfg)), ("clf", LogisticRegression(max_iter=10000, C=C))])
            m.fit(D.iloc[a], y[a]); o[b] = m.predict_proba(D.iloc[b])[:, 1]
        outs.append(o)
    per = [(f1_score(y, (o >= .5).astype(int)), roc_auc_score(y, o)) for o in outs]
    m = np.mean(outs, axis=0)
    f1 = f1_score(y, (m >= .5).astype(int)); auc = roc_auc_score(y, m)
    return dict(f1=f1, auc=auc, comb=0.6 * f1 + 0.4 * auc,
                sm=float(np.mean([0.6 * a + 0.4 * b for a, b in per])), pos=float((m >= .5).mean()))


CLIM = ["rain_sum_7d", "rain_sum_30d", "rain_days_30d", "tavg_7d", "tavg_30d",
        "tmax_30d", "ndvi_30d", "ndvi_90d", "temp_range_mean_30d", "max_daily_rain_30d"]
VAR = [({}, "base"),
       ({"mcell": 1}, "age8 x month"),
       ({"mcell": 1, "month_cell_p": 1}, "age8 x month x period"),
       ({"doy_int": 1}, "age x doy spline"),
       ({"clim": CLIM}, "+climate"),
       ({"clim": CLIM, "age_clim": 1}, "+climate+agexclim"),
       ({"mcell": 1, "clim": CLIM}, "age8xmonth +climate")]
rows = []
t0 = time.time()
for v, tag in VAR:
    for C in [0.01, 0.02, 0.03]:
        try:
            r = run(v, C)
        except Exception as e:
            print("FAIL", tag, C, repr(e)[:90], flush=True); continue
        rows.append({**r, "tag": tag, "C": C})
        print(f"{tag:24s} C={C:<5} f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f} "
              f"sm={r['sm']:.4f} pos={r['pos']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv("experiments/lin5.csv", index=False)

res = pd.DataFrame(rows).sort_values("sm", ascending=False)
print(f"\n({time.time()-t0:.0f}s)\nTOP 10:")
print(res.head(10).to_string(index=False))
print("\nBASELINE for reference: cell_g C=0.02 -> f1=0.8195 auc=0.8253 comb=0.8218 sm=0.8215")
