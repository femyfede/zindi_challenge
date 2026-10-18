"""Round 6: age-group x continuous-year interactions, finer period cells, facility-size feature.

Rationale: age>5 climate-sensitive rate declines WITHIN the pre period too
(2008 .69 -> 2013 .30), so a single global year spline cannot capture a
different time trajectory per age group. Also tests a transductive
facility-size feature (row count per location over train+test combined).
"""
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
SEEDS = (42, 7, 123, 2024, 99)
tr, te = load()
y = tr["is_climate_sensitive"].values
B20 = [-0.01, 0, 1, 2, 3, 4, 5, 8, 12, 18, 25, 35, 45, 55, 62, 68, 73, 78, 85, 200]

# facility size over the COMBINED train+test pool (observable for every row)
_all = pd.concat([tr[["location"]], te[["location"]]])
_fsize = _all.location.value_counts()
FLOOR = 1.0 / len(tr)


def frame(df, cfg):
    d = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["deathdate"])
    post = (dt >= BREAK).astype(float)
    yc = dt.dt.year + (dt.dt.dayofyear - 1) / 365.25
    d["gender"] = df["gender"]; d["zone"] = df["zone"]
    d["age"] = df["age"].astype(float)
    d["year_cont"] = yc.astype(float)
    d["doy"] = dt.dt.dayofyear.astype(float)
    d["age_x_post"] = df["age"].astype(float) * post

    age = df["age"].values.astype(float)
    groups = {"u5": (age <= 5).astype(float), "a6_18": ((age > 5) & (age <= 18)).astype(float),
              "a18_45": ((age > 18) & (age <= 45)).astype(float), "a45p": (age > 45).astype(float)}
    if cfg.get("grp_year"):
        for g, v in groups.items():
            d[f"yr_x_{g}"] = yc.values * v
    if cfg.get("grp_yearspline"):
        for g, v in groups.items():
            d[f"yr_x_{g}"] = (yc.values * v)
    if cfg.get("fsize"):
        d["facility_size"] = df["location"].map(_fsize).fillna(FLOOR).astype(float)
        d["log_fsize"] = np.log1p(d["facility_size"])

    per = np.where(dt >= BREAK, "post", "pre")
    ab20 = pd.cut(df["age"], B20).astype(str)
    d["cell"] = ab20 + "|" + per
    d["cell_g"] = d["cell"] + "|" + df["gender"].astype(str)
    w = cfg.get("ywin", 0)
    if w:
        ab8 = pd.cut(df["age"], B8).astype(str)
        d["cell_y8"] = ab8 + "|" + (dt.dt.year // w).astype(str)
    return d


B8 = [-0.01, 0, 2, 6, 18, 45, 65, 78, 200]


def spec(cfg):
    t = [("sp_age", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age"]),
         ("sp_agepost", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age_x_post"]),
         ("sp_year", SplineTransformer(n_knots=cfg.get("k_year", 6), degree=3, include_bias=False), ["year_cont"]),
         ("sp_doy", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["doy"])]
    if cfg.get("grp_yearspline"):
        for g in ("u5", "a6_18", "a18_45", "a45p"):
            t.append((f"sp_y_{g}", SplineTransformer(n_knots=4, degree=3, include_bias=False), [f"yr_x_{g}"]))
    cats = ["gender", "zone", "cell", "cell_g"]
    if cfg.get("cell_y8"):
        cats.append("cell_y8")
    nums = []
    if cfg.get("grp_year"):
        nums += [f"yr_x_{g}" for g in ("u5", "a6_18", "a18_45", "a45p")]
    if cfg.get("fsize"):
        nums += ["facility_size", "log_fsize"]
    if nums:
        t.append(("num", StandardScaler(), nums))
    t.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
    return ColumnTransformer(t)


def run(cfg, C):
    D = frame(tr, cfg)
    outs = []
    for s in SEEDS:
        o = np.zeros(len(D))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(D, y):
            m = Pipeline([("sp", spec(cfg)), ("clf", LogisticRegression(max_iter=10000, C=C))])
            m.fit(D.iloc[a], y[a]); o[b] = m.predict_proba(D.iloc[b])[:, 1]
        outs.append(o)
    outs = np.array(outs)
    per = [(f1_score(y, (o >= .5).astype(int)), roc_auc_score(y, o)) for o in outs]
    m = outs.mean(axis=0)
    f1 = f1_score(y, (m >= .5).astype(int)); auc = roc_auc_score(y, m)
    return dict(f1=f1, auc=auc, comb=0.6 * f1 + 0.4 * auc,
                sm=float(np.mean([0.6 * a + 0.4 * b for a, b in per])),
                mn=float(np.min([0.6 * a + 0.4 * b for a, b in per])), pos=float((m >= .5).mean()))


VAR = [({}, "base"),
       ({"grp_year": 1}, "agegrp x year"),
       ({"grp_yearspline": 1}, "agegrp x year SPLINE"),
       ({"fsize": 1}, "+facility size"),
       ({"fsize": 1, "grp_year": 1}, "agegrp x year + fsize"),
       ({"grp_year": 1, "ywin": 3}, "agegrp x year + age8 x yr3bin"),
       ({"ywin": 3}, "age8 x yr3bin"),
       ({"ywin": 2}, "age8 x yr2bin"),
       ({"grp_year": 1, "k_year": 8}, "agegrp x year, k_year=8")]
rows = []
t0 = time.time()
for v, tag in VAR:
    for C in [0.01, 0.02, 0.03]:
        try:
            r = run(v, C)
        except Exception as e:
            print("FAIL", tag, C, repr(e)[:90], flush=True); continue
        rows.append({**r, "tag": tag, "C": C})
        print(f"{tag:26s} C={C:<5} f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f} "
              f"sm={r['sm']:.4f} min={r['mn']:.4f} pos={r['pos']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv("experiments/lin6.csv", index=False)

res = pd.DataFrame(rows).sort_values("sm", ascending=False)
print(f"\n({time.time()-t0:.0f}s)\nTOP 12:")
print(res.head(12).to_string(index=False))
print("\nPREVIOUS BEST (singleA C=0.02): f1=0.8187 auc=0.8250 comb=0.8212 seedmean=0.8214")
