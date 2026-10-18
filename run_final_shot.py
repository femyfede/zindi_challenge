"""Final long shot: self-training on the test features, plus registration-artifact features."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from bench import load

BREAK = pd.Timestamp("2017-07-01")
SEEDS = (42, 7, 123, 2024, 99)
tr, te = load()
y = tr["is_climate_sensitive"].values
B20 = [-0.01, 0, 1, 2, 3, 4, 5, 8, 12, 18, 25, 35, 45, 55, 62, 68, 73, 78, 85, 200]
CATS = ["gender", "zone", "cell", "cell_g"]


def frames(df, bins, extra=False):
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
    if extra:
        d["dow"] = dt.dt.dayofweek.astype(str)
        d["day"] = dt.dt.day.astype(str)
        d["weekofyr"] = (dt.dt.isocalendar().week.astype(int)).astype(str)
    return d


def pipe(C, extra=False):
    sp = [("sp_age", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age"]),
          ("sp_agepost", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age_x_post"]),
          ("sp_year", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["year_cont"]),
          ("sp_doy", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["doy"])]
    cats = list(CATS)
    if extra:
        cats += ["dow", "day", "weekofyr"]
    sp.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
    return Pipeline([("sp", ColumnTransformer(sp)),
                     ("clf", LogisticRegression(max_iter=10000, C=C))])


def met(o):
    per = [(f1_score(y, (x >= .5).astype(int)), roc_auc_score(y, x)) for x in o]
    m = o.mean(axis=0)
    f1 = f1_score(y, (m >= .5).astype(int)); auc = roc_auc_score(y, m)
    return f1, auc, 0.6 * f1 + 0.4 * auc, float(np.mean([0.6 * a + 0.4 * b for a, b in per]))


def oof_plain(C, extra=False, seeds=SEEDS):
    outs = []
    for s in seeds:
        o = np.zeros(len(tr))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(tr, y):
            m = pipe(C, extra); m.fit(frames(tr.iloc[a], B20, extra), y[a])
            o[b] = m.predict_proba(frames(tr.iloc[b], B20, extra))[:, 1]
        outs.append(o)
    return np.array(outs)


base = oof_plain(0.02)
f1, auc, comb, sm = met(base)
print(f"reference (B20, C=0.02): f1={f1:.4f} auc={auc:.4f} comb={comb:.4f} sm={sm:.4f}")

print("\n=== 1) registration-artifact features (day-of-week, day-of-month, ISO week) ===")
o = oof_plain(0.02, extra=True)
f1, auc, comb, sm = met(o)
print(f"  +dow/+day/+week: f1={f1:.4f} auc={auc:.4f} comb={comb:.4f} sm={sm:.4f}")

print("\n=== 2) self-training on the test features (transductive) ===")
# teacher fit on all train; pseudo-label test; retrain with test rows at several weights
teacher = pipe(0.02); teacher.fit(frames(tr, B20), y)
pte = teacher.predict_proba(frames(te, B20))[:, 1]
print(f"  teacher test posrate={(pte>=0.5).mean():.4f} meanP={pte.mean():.4f}")

for conf in [0.90, 0.95, 0.98]:
    for w in [0.3, 1.0]:
        keep = np.abs(pte - 0.5) > (1 - conf)      # confident pseudo-labels only
        pl = (pte[keep] >= 0.5).astype(int)
        outs = []
        for s in SEEDS[:3]:
            o = np.zeros(len(tr))
            for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(tr, y):
                Xa = pd.concat([frames(tr.iloc[a], B20),
                                frames(te[keep].reset_index(drop=True), B20)], ignore_index=True)
                ya = np.concatenate([y[a], pl.astype(float) * 0 + pl])
                sw = np.concatenate([np.ones(len(a)), np.full(len(pl), w)])
                mdl = pipe(0.02)
                mdl.fit(Xa, ya, **{"clf__sample_weight": sw})
                o[b] = mdl.predict_proba(frames(tr.iloc[b], B20))[:, 1]
            outs.append(o)
        f1, auc, comb, sm = met(np.array(outs))
        print(f"  conf>{conf} w={w} n_pl={keep.sum():4d}: f1={f1:.4f} auc={auc:.4f} "
              f"comb={comb:.4f} sm={sm:.4f}", flush=True)

print("\n=== 3) target-mean shift check: does test look like train? ===")
import numpy as np
print(f"  train mean p={base.mean():.4f} | test mean p={pte.mean():.4f}")
print(f"  train posrate={(base.mean(0)>=0.5).mean():.4f} | test posrate={(pte>=0.5).mean():.4f}")
dte = pd.to_datetime(te["deathdate"])
print(f"  test years {dte.dt.year.min()}-{dte.dt.year.max()}, "
      f"pre-break {int((dte<BREAK).sum())} / post {int((dte>=BREAK).sum())}")
print(f"  test infant(age<=1) share={(te.age<=1).mean():.4f} vs train {(tr.age<=1).mean():.4f}")
