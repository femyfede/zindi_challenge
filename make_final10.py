"""FINAL: bagged-over-binnings spline-logistic ensemble -> exactly 10 submissions
(competition closes today, limit is 10/day)."""
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from bench import load
from src.submission import build_submission, verify_submission

BREAK = pd.Timestamp("2017-07-01")
SEEDS = (42, 7, 123, 2024, 99)
tr, te = load()
y = tr["is_climate_sensitive"].values

B20 = [-0.01, 0, 1, 2, 3, 4, 5, 8, 12, 18, 25, 35, 45, 55, 62, 68, 73, 78, 85, 200]


def pool(n_extra, seed=0):
    rng = np.random.default_rng(seed)
    out = [B20]
    for _ in range(n_extra):
        inner = sorted(set([0, 1, 2, 3, 4, 5] + [int(v) for v in rng.integers(6, 20, 8)] +
                          [int(v) for v in rng.integers(20, 95, 5)]))
        out.append([-0.01] + inner + [200])
    return out


def frames(df, bins):
    d = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["deathdate"])
    post = (dt >= BREAK).astype(float)
    d["gender"] = df["gender"]; d["zone"] = df["zone"]
    d["age"] = df["age"].astype(float)
    d["year_cont"] = (dt.dt.year + (dt.dt.dayofyear - 1) / 365.25).astype(float)
    d["doy"] = dt.dt.dayofyear.astype(float)
    d["age_x_post"] = df["age"].astype(float) * post
    cell = pd.cut(df["age"], bins).astype(str) + "|" + np.where(dt >= BREAK, "post", "pre")
    d["cell"] = cell
    d["cell_g"] = cell + "|" + df["gender"].astype(str)
    return d


CATS = ["gender", "zone", "cell", "cell_g"]


def pipe(C):
    spec = ColumnTransformer([
        ("sp_age", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age"]),
        ("sp_agepost", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age_x_post"]),
        ("sp_year", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["year_cont"]),
        ("sp_doy", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["doy"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATS)])
    return Pipeline([("sp", spec), ("clf", LogisticRegression(max_iter=10000, C=C))])


def bag_oof(C, bins_list):
    outs = []
    for s in SEEDS:
        o = np.zeros(len(tr))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(tr, y):
            ps = []
            for bn in bins_list:
                m = pipe(C); m.fit(frames(tr.iloc[a], bn), y[a])
                ps.append(m.predict_proba(frames(tr.iloc[b], bn))[:, 1])
            o[b] = np.mean(ps, axis=0)
        outs.append(o)
    return np.array(outs)


def bag_test(C, bins_list):
    ps = []
    for bn in bins_list:
        m = pipe(C); m.fit(frames(tr, bn), y)
        ps.append(m.predict_proba(frames(te, bn))[:, 1])
    return np.mean(ps, axis=0)


def met(o):
    per = [(f1_score(y, (x >= .5).astype(int)), roc_auc_score(y, x)) for x in o]
    m = o.mean(axis=0)
    f1 = f1_score(y, (m >= .5).astype(int)); auc = roc_auc_score(y, m)
    return f1, auc, 0.6 * f1 + 0.4 * auc, float(np.mean([0.6 * a + 0.4 * b for a, b in per]))


P12, P24 = pool(11, 0), pool(23, 1)
PLAN = [
    ("F1_BAG12_C020", 0.02, P12),
    ("F2_BAG12_C030", 0.03, P12),
    ("F3_BAG12_C050", 0.05, P12),
    ("F4_BAG12_C010", 0.01, P12),
    ("F5_BAG12_C015", 0.015, P12),
    ("F6_BAG24_C020", 0.02, P24),
    ("F7_BAG24_C030", 0.03, P24),
    ("F8_SINGLE_B20_C020", 0.02, [B20]),
    ("F9_BAG12B_C020", 0.02, pool(11, 7)),
    ("F10_BAG24_C015", 0.015, P24),
]
rows = []
for i, (tag, C, bl) in enumerate(PLAN, 1):
    t0 = time.time()
    o = bag_oof(C, bl)
    f1, auc, comb, sm = met(o)
    p = np.clip(bag_test(C, bl), 1e-6, 1 - 1e-6)
    sub = build_submission(te["ID"], p)
    errs = verify_submission(sub, te, sample_path="SampleSubmission.csv")
    assert not errs, errs
    fn = f"F{i:02d}_{tag}.csv"
    sub.to_csv(fn, index=False)
    rows.append(dict(file=fn, C=C, nbins=len(bl), cvF1=f1, cvAUC=auc, cvComb=comb, cvSeedMean=sm,
                     posrate=float(sub.TargetF1.mean()), meanP=float(sub.TargetRAUC.mean())))
    print(f"{fn:26s} nb={len(bl):2d} C={C:<5} cvF1={f1:.4f} cvAUC={auc:.4f} cvComb={comb:.4f} "
          f"sm={sm:.4f} pos={sub.TargetF1.mean():.4f} [{time.time()-t0:.0f}s]", flush=True)

df = pd.DataFrame(rows)
df.to_csv("experiments/final10.csv", index=False)
pd.set_option("display.width", 220)
print("\n" + df.sort_values("cvSeedMean", ascending=False).to_string(index=False))
