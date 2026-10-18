"""Build the final submission: probability-averaged spline-logistic ensemble.

Model: age x period spline logistic regression.
Key structure exploited: a structural break at 2017-07 in which the
climate-sensitive rate for infants collapses from ~0.97 to ~0.55.
Validation: 3 seeds x 5-fold stratified CV (matches the random train/test split).
"""
import numpy as np, pandas as pd, warnings, time, json
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from bench import load
from features2 import build
from src.submission import build_submission, verify_submission

BREAK = pd.Timestamp("2017-07-01")
SEEDS = (42, 7, 123)
BINS = {
    20: [-0.01, 0, 1, 2, 3, 4, 5, 8, 12, 18, 25, 35, 45, 55, 62, 68, 73, 78, 85, 200],
    13: [-0.01, 0, 1, 3, 6, 12, 18, 30, 45, 60, 70, 75, 85, 200],
}
tr, te = load()
y = tr["is_climate_sensitive"].values


def frame(df, bins, with_gender):
    d = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["deathdate"])
    post = (dt >= BREAK).astype(float)
    d["gender"] = df["gender"]; d["zone"] = df["zone"]
    d["age"] = df["age"].astype(float)
    d["year_cont"] = (dt.dt.year + (dt.dt.dayofyear - 1) / 365.25).astype(float)
    d["doy"] = dt.dt.dayofyear.astype(float)
    d["age_x_post"] = df["age"].astype(float) * post
    ab = pd.cut(df["age"], BINS[bins]).astype(str)
    cell = ab + "|" + np.where(dt >= BREAK, "post", "pre")
    d["cell"] = cell
    cats = ["gender", "zone", "cell"]
    if with_gender:
        d["cell_g"] = cell + "|" + df["gender"].astype(str)
        cats.append("cell_g")
    return d, cats


def make_pipe(cats, C):
    spec = ColumnTransformer([
        ("sp_age", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age"]),
        ("sp_agepost", SplineTransformer(n_knots=10, degree=3, include_bias=False), ["age_x_post"]),
        ("sp_year", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["year_cont"]),
        ("sp_doy", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["doy"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
    return Pipeline([("sp", spec), ("clf", LogisticRegression(max_iter=10000, C=C))])


MEMBERS = [("A_bins20_g_C02", 20, True, 0.02),
           ("B_bins20_C03", 20, False, 0.03),
           ("C_bins13_g_C03", 13, True, 0.03),
           ("D_bins13_C02", 13, False, 0.02)]

print("=== repeated 5-fold CV (3 seeds) ===")
oofs = {}
for name, bins, g, C in MEMBERS:
    D, cats = frame(tr, bins, g)
    outs = []
    for s in SEEDS:
        o = np.zeros(len(D))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(D, y):
            m = make_pipe(cats, C); m.fit(D.iloc[a], y[a]); o[b] = m.predict_proba(D.iloc[b])[:, 1]
        outs.append(o)
    oofs[name] = np.array(outs)
    per = [(f1_score(y, (o >= .5).astype(int)), roc_auc_score(y, o)) for o in outs]
    mm = oofs[name].mean(axis=0)
    f1 = f1_score(y, (mm >= .5).astype(int)); auc = roc_auc_score(y, mm)
    print(f"  {name:16s} f1={f1:.4f} auc={auc:.4f} comb={0.6*f1+0.4*auc:.4f} "
          f"(seedmean {np.mean([0.6*a+0.4*b for a,b in per]):.4f})", flush=True)

print("\n=== ensemble candidates ===")
CAND = {"ENS_all4": [m[0] for m in MEMBERS],
        "ENS_ABC": ["A_bins20_g_C02", "B_bins20_C03", "C_bins13_g_C03"],
        "ENS_AB": ["A_bins20_g_C02", "B_bins20_C03"],
        "ENS_singleA": ["A_bins20_g_C02"]}
best = None
for cname, mem in CAND.items():
    p = np.mean([oofs[m] for m in mem], axis=0)
    per = [(f1_score(y, (o >= .5).astype(int)), roc_auc_score(y, o)) for o in p]
    f1 = f1_score(y, (p.mean(0) >= .5).astype(int)); auc = roc_auc_score(y, p.mean(0))
    sm = np.mean([0.6 * a + 0.4 * b for a, b in per])
    print(f"  {cname:14s} f1={f1:.4f} auc={auc:.4f} comb={0.6*f1+0.4*auc:.4f} seedmean={sm:.4f}")
    if best is None or sm > best[1]:
        best = (cname, sm, mem)
print(f"\nSELECTED: {best[0]} (seedmean {best[1]:.4f})")

# ---- fit on ALL train, predict test ----
test_proba = {}
for name, bins, g, C in MEMBERS:
    Dtr, cats = frame(tr, bins, g)
    Dte, _ = frame(te, bins, g)
    m = make_pipe(cats, C)
    m.fit(Dtr, y)
    test_proba[name] = m.predict_proba(Dte)[:, 1]

p_test = np.mean([test_proba[m] for m in best[2]], axis=0)
p_test = np.clip(p_test, 1e-6, 1 - 1e-6)

sub = build_submission(te["ID"], p_test)
errs = verify_submission(sub, te, sample_path="SampleSubmission.csv")
assert not errs, errs
sub.to_csv("submission_v2.csv", index=False)
print(f"\nwrote submission_v2.csv  n={len(sub)}  posrate={(sub.TargetF1==1).mean():.4f}  "
      f"meanP={sub.TargetRAUC.mean():.4f}  members={best[2]}")

# alternative candidate: the single strongest member
alt = np.clip(test_proba["A_bins20_g_C02"], 1e-6, 1 - 1e-6)
sub2 = build_submission(te["ID"], alt)
assert not verify_submission(sub2, te, sample_path="SampleSubmission.csv")
sub2.to_csv("submission_v2_single.csv", index=False)
print(f"wrote submission_v2_single.csv  posrate={(sub2.TargetF1==1).mean():.4f}")

json.dump({"selected": best[0], "members": best[2], "cv_seedmean_combined": best[1]},
          open("experiments/final_choice.json", "w"), indent=1)
