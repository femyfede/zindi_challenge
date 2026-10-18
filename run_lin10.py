"""Round 10 — the last untried model classes.

1. GAM with BAND-SPECIFIC year splines (each age band gets its own time
   trajectory). The crude linear version was tested and failed; the spline
   version never actually ran.
2. F1-focused sample weighting: up-weight rows sitting near the 0.5 cut,
   since F1 (60% of the metric) is dominated by boundary errors.
3. Elastic-net penalty sweep (different regularisation shape).
4. Per-age-band isotonic recalibration of the final score (F1 only).
5. Separate specialist model per period.
"""
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
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
BANDS = [-0.01, 0, 1, 3, 6, 18, 30, 45, 65, 80, 200]
NB = len(BANDS) - 1


def base_frame(df, gam=False):
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
    if gam:
        b = pd.cut(df["age"], BANDS, labels=False).fillna(NB - 1).astype(int).values
        yc = d["year_cont"].values
        for k in range(NB):
            m = (b == k).astype(float)
            d[f"byr_{k}"] = yc * m
    return d


def make_spec(gam):
    t = [("sp_age", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age"]),
         ("sp_agepost", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age_x_post"]),
         ("sp_year", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["year_cont"]),
         ("sp_doy", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["doy"])]
    if gam:
        for k in range(NB):
            t.append((f"sp_byr{k}", SplineTransformer(n_knots=4, degree=3, include_bias=False), [f"byr_{k}"]))
    t.append(("cat", OneHotEncoder(handle_unknown="ignore"), CATS))
    return ColumnTransformer(t)


def make_pipe(gam, C, l1=0.0):
    return Pipeline([("sp", make_spec(gam)),
                     ("clf", LogisticRegression(max_iter=20000, C=C, penalty="elasticnet",
                                                l1_ratio=l1, solver="saga"))])


def met(o):
    per = [(f1_score(y, (x >= .5).astype(int)), roc_auc_score(y, x)) for x in o]
    m = o.mean(axis=0)
    f1 = f1_score(y, (m >= .5).astype(int)); auc = roc_auc_score(y, m)
    return f1, auc, 0.6 * f1 + 0.4 * auc, float(np.mean([0.6 * a + 0.4 * b for a, b in per]))


REF = None
def run(gam, C, l1=0.0, seeds=SEEDS, weight=None):
    D = base_frame(tr, gam)
    outs = []
    for s in seeds:
        o = np.zeros(len(D))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(D, y):
            m = make_pipe(gam, C, l1)
            if weight is None:
                m.fit(D.iloc[a], y[a])
            else:
                inner = np.zeros(len(a)); m.fit(D.iloc[a], y[a])
                p = m.predict_proba(D.iloc[a])[:, 1]
                w = 1.0 + weight * (1.0 - np.abs(2 * p - 1))
                m2 = make_pipe(gam, C, l1); m2.fit(D.iloc[a], y[a], **{"clf__sample_weight": w})
                o[b] = m2.predict_proba(D.iloc[b])[:, 1]
                continue
            o[b] = m.predict_proba(D.iloc[b])[:, 1]
        outs.append(o)
    return np.array(outs)


print("=== reference (no GAM, lbfgs C=0.02) ===")
D = base_frame(tr, False)
outs = []
for s in SEEDS:
    o = np.zeros(len(D))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(D, y):
        spec = make_spec(False)
        m = Pipeline([("sp", spec), ("clf", LogisticRegression(max_iter=10000, C=0.02))])
        m.fit(D.iloc[a], y[a]); o[b] = m.predict_proba(D.iloc[b])[:, 1]
    outs.append(o)
REF = np.array(outs)
f1, auc, comb, sm = met(REF)
print(f"  f1={f1:.4f} auc={auc:.4f} comb={comb:.4f} sm={sm:.4f}  <-- reference")

print("\n=== 1) GAM with band-specific year splines ===")
for C in [0.005, 0.01, 0.02, 0.05]:
    o = run(True, C)
    f1, auc, comb, sm = met(o)
    print(f"  GAM C={C:<6}: f1={f1:.4f} auc={auc:.4f} comb={comb:.4f} sm={sm:.4f} pos={(o.mean(0)>=.5).mean():.3f}", flush=True)

print("\n=== 2) F1-focused boundary sample weighting (on reference model) ===")
for w in [0.5, 1.0, 2.0]:
    o = run(False, 0.02, weight=w)
    f1, auc, comb, sm = met(o)
    print(f"  weight +{w:<4}: f1={f1:.4f} auc={auc:.4f} comb={comb:.4f} sm={sm:.4f} pos={(o.mean(0)>=.5).mean():.3f}", flush=True)

print("\n=== 3) elastic-net sweep (saga) ===")
for l1 in [0.0, 0.3, 0.6, 0.9]:
    for C in [0.02, 0.05]:
        try:
            o = run(False, C, l1)
            f1, auc, comb, sm = met(o)
            print(f"  l1={l1:<4} C={C:<5}: f1={f1:.4f} auc={auc:.4f} comb={comb:.4f} sm={sm:.4f}", flush=True)
        except Exception as e:
            print(f"  l1={l1} C={C} FAIL {repr(e)[:70]}", flush=True)

print("\n=== 4) per-age-band isotonic recalibration (F1 only, AUC unchanged) ===")
for nb in [3, 5]:
    b = pd.cut(tr["age"], np.linspace(-0.01, 90, nb + 1), labels=False).fillna(nb - 1).astype(int).values
    cal = np.zeros_like(REF)
    for i in range(REF.shape[0]):
        p = REF[i]
        for k in range(nb):
            msk = (b == k)
            if msk.sum() < 80 or len(np.unique(y[msk])) < 2:
                cal[i][msk] = p[msk]; continue
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[msk], y[msk])
            cal[i][msk] = ir.predict(p[msk])
    f1, auc, comb, sm = met(cal)
    print(f"  bands={nb}: f1={f1:.4f} auc={auc:.4f} comb={comb:.4f} sm={sm:.4f} pos={(cal.mean(0)>=.5).mean():.3f}")
