"""Produce 3 differentiated submissions to spread risk across the 30% public-LB sample."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from make_submission import frame, make_pipe, y, tr, te
from src.submission import build_submission, verify_submission

SEEDS = (42, 7, 123, 2024, 99)
VARIANTS = [
    ("v1_C02_maxF1",  0.02, "F1-optimal, high positive rate"),
    ("v2_C05_maxAUC", 0.05, "AUC-optimal, lower positive rate"),
    ("v3_C03_mid",    0.03, "middle ground"),
]

for name, C, desc in VARIANTS:
    D, cats = frame(tr, 20, True)
    Dt, _ = frame(te, 20, True)
    outs = []
    for s in SEEDS:
        o = np.zeros(len(D))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(D, y):
            m = make_pipe(cats, C); m.fit(D.iloc[a], y[a]); o[b] = m.predict_proba(D.iloc[b])[:, 1]
        outs.append(o)
    outs = np.array(outs)
    m = outs.mean(axis=0)
    f1 = f1_score(y, (m >= .5).astype(int)); auc = roc_auc_score(y, m)
    per = [(f1_score(y, (o >= .5).astype(int)), roc_auc_score(y, o)) for o in outs]
    sm = float(np.mean([0.6 * a + 0.4 * b for a, b in per]))

    mdl = make_pipe(cats, C); mdl.fit(D, y)
    p = np.clip(mdl.predict_proba(Dt)[:, 1], 1e-6, 1 - 1e-6)
    sub = build_submission(te["ID"], p)
    errs = verify_submission(sub, te, sample_path="SampleSubmission.csv")
    assert not errs, errs
    fn = f"submission_v2_{name}.csv"
    sub.to_csv(fn, index=False)
    print(f"{fn:34s} C={C:<5} cvF1={f1:.4f} cvAUC={auc:.4f} cvComb={0.6*f1+0.4*auc:.4f} "
          f"seedmean={sm:.4f} | test posrate={sub.TargetF1.mean():.4f} meanP={sub.TargetRAUC.mean():.4f}")
    print(f"{'':34s} {desc}")
