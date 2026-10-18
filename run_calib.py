"""Honest test of per-age-band isotonic recalibration.

The previous run fitted isotonic on the same OOF rows it scored -> inflated.
Here the calibrator is fitted on an INNER CV over the outer-train portion only,
then applied to the untouched outer validation fold.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from bench import load

BREAK = pd.Timestamp("2017-07-01")
SEEDS = (42, 7, 123)
tr, te = load()
y = tr["is_climate_sensitive"].values
B20 = [-0.01, 0, 1, 2, 3, 4, 5, 8, 12, 18, 25, 35, 45, 55, 62, 68, 73, 78, 85, 200]
CATS = ["gender", "zone", "cell", "cell_g"]


def frame(df):
    d = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["deathdate"])
    post = (dt >= BREAK).astype(float)
    d["gender"] = df["gender"]; d["zone"] = df["zone"]
    d["age"] = df["age"].astype(float)
    d["year_cont"] = (dt.dt.year + (dt.dt.dayofyear - 1) / 365.25).astype(float)
    d["doy"] = dt.dt.dayofyear.astype(float)
    d["age_x_post"] = df["age"].astype(float) * post
    cell = pd.cut(df["age"], B20).astype(str) + "|" + np.where(dt >= BREAK, "post", "pre")
    d["cell"] = cell
    d["cell_g"] = cell + "|" + df["gender"].astype(str)
    return d


def pipe(C=0.02):
    spec = ColumnTransformer([
        ("sp_age", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age"]),
        ("sp_agepost", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age_x_post"]),
        ("sp_year", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["year_cont"]),
        ("sp_doy", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["doy"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATS)])
    return Pipeline([("sp", spec), ("clf", LogisticRegression(max_iter=10000, C=C))])


D = frame(tr)


def fit_inner_oof(indices, seed):
    """inner OOF probs for the rows in `indices` (used only to fit the calibrator)"""
    sub = D.iloc[indices]
    yy = y[indices]
    o = np.zeros(len(indices))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(sub, yy):
        m = pipe(); m.fit(sub.iloc[a], yy[a]); o[b] = m.predict_proba(sub.iloc[b])[:, 1]
    return o


BANDSETS = {"global": None,
            "age3": np.array([-0.01, 1, 18, 200]),
            "age5": np.array([-0.01, 1, 6, 30, 65, 200]),
            "age8": np.array([-0.01, 0, 1, 3, 6, 18, 45, 65, 200]),
            "ageXperiod": "period"}

res = {k: [] for k in BANDSETS}
res["raw"] = []
for s in SEEDS:
    raw = np.zeros(len(tr))
    cal = {k: np.zeros(len(tr)) for k in BANDSETS}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(D, y):
        inner = fit_inner_oof(a, s + 1000)
        m = pipe(); m.fit(D.iloc[a], y[a])
        pv = m.predict_proba(D.iloc[b])[:, 1]
        raw[b] = pv
        for k, edges in BANDSETS.items():
            if isinstance(edges, str):
                dtb = pd.to_datetime(tr["deathdate"]).values[b]
                dta = pd.to_datetime(tr["deathdate"]).values[a]
                kb = (dtb >= np.datetime64(BREAK)).astype(int)
                ka = (dta >= np.datetime64(BREAK)).astype(int)
            elif edges is None:
                kb = np.zeros(len(b), int); ka = np.zeros(len(a), int)
            else:
                kb = pd.cut(tr["age"].values[b], edges, labels=False)
                kb = np.nan_to_num(kb, nan=len(edges) - 2).astype(int)
                ka = pd.cut(tr["age"].values[a], edges, labels=False)
                ka = np.nan_to_num(ka, nan=len(edges) - 2).astype(int)
            out = np.zeros(len(b))
            ngrp = kb.max() + 1
            for g in range(ngrp):
                mb, ma = kb == g, ka == g
                if ma.sum() < 60 or len(np.unique(y[a][ma])) < 2:
                    out[mb] = pv[mb]; continue
                ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(inner[ma], y[a][ma])
                out[mb] = ir.predict(pv[mb])
            cal[k][b] = out
    res["raw"].append(raw)
    for k in BANDSETS:
        res[k].append(cal[k])


def met(o):
    per = [(f1_score(y, (x >= .5).astype(int)), roc_auc_score(y, x)) for x in o]
    m = o.mean(axis=0)
    f1 = f1_score(y, (m >= .5).astype(int)); auc = roc_auc_score(y, m)
    return f1, auc, 0.6 * f1 + 0.4 * auc, float(np.mean([0.6 * a + 0.4 * b for a, b in per]))


print("=== HONESTLY nested per-band isotonic calibration ===")
for k, o in res.items():
    f1, auc, comb, sm = met(np.array(o))
    tag = " (baseline, no calibration)" if k == "raw" else ""
    print(f"  {k:10s}: f1={f1:.4f} auc={auc:.4f} comb={comb:.4f} sm={sm:.4f} "
          f"pos={(np.array(o).mean(0)>=.5).mean():.3f}{tag}")

print("\nFor reference, the LEAKY version (fit+score on same rows) reported:")
print("  age5      : f1=0.8232 auc=0.8379 comb=0.8291 sm=0.8269   <-- inflated")
print("  age3      : f1=0.8202 auc=0.8355 comb=0.8263 sm=0.8252   <-- inflated")
