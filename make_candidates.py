"""Build a spread of strong, differentiated candidates.

Public LB is ~309 rows => SE on the score is ~0.04. Submitting several
*genuinely* good models at different operating points lets the public board
reveal which point suits this test split. All candidates have CV >= 0.818.
"""
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from make_submission import frame, make_pipe, y, tr, te
from src.submission import build_submission, verify_submission

SEEDS = (42, 7, 123, 2024, 99)
BINS20, BINS13 = 20, 13

# (name, bins, gender-cross, C)
MEMBERS = [
    ("C02", 20, True, 0.02), ("C03", 20, True, 0.03), ("C05", 20, True, 0.05),
    ("C01", 20, True, 0.01),
    ("B_C02", 20, False, 0.02), ("B_C03", 20, False, 0.03),
    ("N_C02", 13, True, 0.02), ("N_C03", 13, True, 0.03),
]

oof, testp = {}, {}
for name, b, g, C in MEMBERS:
    D, cats = frame(tr, b, g)
    Dt, _ = frame(te, b, g)
    outs = []
    for s in SEEDS:
        o = np.zeros(len(D))
        for a, bb in StratifiedKFold(5, shuffle=True, random_state=s).split(D, y):
            m = make_pipe(cats, C); m.fit(D.iloc[a], y[a]); o[bb] = m.predict_proba(D.iloc[bb])[:, 1]
        outs.append(o)
    oof[name] = np.array(outs)
    mdl = make_pipe(cats, C); mdl.fit(D, y)
    testp[name] = mdl.predict_proba(Dt)[:, 1]

# candidate submissions: singles + probability-averaged blends
CAND = {
    "s1_C02":        ["C02"],
    "s2_C05":        ["C05"],
    "s3_C03":        ["C03"],
    "s4_C01_maxPos": ["C01"],
    "s5_blend4":     ["C02", "C03", "C05", "B_C02"],
    "s6_blend6":     ["C02", "C03", "C05", "C01", "B_C02", "N_C02"],
    "s7_blend_bins": ["C02", "B_C02", "N_C02"],
}
rows = []
for tag, mem in CAND.items():
    o = np.mean([oof[m] for m in mem], axis=0)
    per = [(f1_score(y, (x >= .5).astype(int)), roc_auc_score(y, x)) for x in o]
    mm = o.mean(axis=0)
    f1 = f1_score(y, (mm >= .5).astype(int)); auc = roc_auc_score(y, mm)
    sm = float(np.mean([0.6 * a + 0.4 * b for a, b in per]))
    p = np.clip(np.mean([testp[m] for m in mem], axis=0), 1e-6, 1 - 1e-6)
    sub = build_submission(te["ID"], p)
    assert not verify_submission(sub, te, sample_path="SampleSubmission.csv")
    fn = f"SUB_{tag}.csv"
    sub.to_csv(fn, index=False)
    rows.append(dict(cand=tag, file=fn, members="+".join(mem), cvF1=f1, cvAUC=auc,
                     cvComb=0.6 * f1 + 0.4 * auc, cvSeedMean=sm,
                     posrate=float(sub.TargetF1.mean()), meanP=float(sub.TargetRAUC.mean())))

r = pd.DataFrame(rows).sort_values("cvSeedMean", ascending=False)
r.to_csv("experiments/candidates.csv", index=False)
pd.set_option("display.width", 200)
print(r.to_string(index=False))
print("\nSUBMIT ORDER: start with the top 3 by cvSeedMean.")
print("For the PRIVATE board pick the 2 with best CV - NOT the best public score,")
print("since public is only ~309 rows (SE ~0.04) and that luck washes out on private.")
