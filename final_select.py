"""Final selection: 5-seed repeated CV tie-break between the top candidates, then write submissions."""
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from make_submission import frame, make_pipe, MEMBERS, load, tr, te, y

SEEDS = (42, 7, 123, 2024, 99)
frames = {}
for name, bins, g, C in MEMBERS:
    frames[name] = (frame(tr, bins, g), frame(te, bins, g), C)

OOF = {}
for name, (D, _, C) in frames.items():
    cats = D[1]
    outs = []
    for s in SEEDS:
        o = np.zeros(len(D[0]))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(D[0], y):
            m = make_pipe(cats, C); m.fit(D[0].iloc[a], y[a]); o[b] = m.predict_proba(D[0].iloc[b])[:, 1]
        outs.append(o)
    OOF[name] = np.array(outs)
    print(f"oof done: {name}", flush=True)


def report(tag, p):
    per = [(f1_score(y, (o >= .5).astype(int)), roc_auc_score(y, o)) for o in p]
    m = p.mean(axis=0)
    f1 = f1_score(y, (m >= .5).astype(int)); auc = roc_auc_score(y, m)
    comb = [0.6 * a + 0.4 * b for a, b in per]
    print(f"  {tag:18s} f1={f1:.4f} auc={auc:.4f} comb={0.6*f1+0.4*auc:.4f} "
          f"seedmean={np.mean(comb):.4f} seedmin={np.min(comb):.4f} seedmax={np.max(comb):.4f}")
    return float(np.mean(comb)), float(np.min(comb))


print("\n=== candidates (5 seeds) ===")
names = list(OOF)
CAND = {
    "singleA": ["A_bins20_g_C02"],
    "singleB": ["B_bins20_C03"],
    "singleC": ["C_bins13_g_C03"],
    "AB": ["A_bins20_g_C02", "B_bins20_C03"],
    "ABC": ["A_bins20_g_C02", "B_bins20_C03", "C_bins13_g_C03"],
    "ABCD": names,
}
res = {}
for tag, mem in CAND.items():
    res[tag] = report(tag, np.mean([OOF[m] for m in mem], axis=0))

best = max(res, key=lambda k: (res[k][0], res[k][1]))
print(f"\nBEST: {best}")

# refit members on full train, predict test
tp = {}
for name, (D, Dt, C) in frames.items():
    m = make_pipe(D[1], C); m.fit(D[0], y)
    tp[name] = m.predict_proba(Dt[0])[:, 1]

from src.submission import build_submission, verify_submission

for tag in [best, "ABC" if best != "ABC" else "AB"]:
    p = np.clip(np.mean([tp[m] for m in CAND[tag]], axis=0), 1e-6, 1 - 1e-6)
    sub = build_submission(te["ID"], p)
    errs = verify_submission(sub, te, sample_path="SampleSubmission.csv")
    assert not errs, errs
    fn = f"submission_v2_{tag}.csv"
    sub.to_csv(fn, index=False)
    print(f"wrote {fn}: n={len(sub)} posrate={(sub.TargetF1==1).mean():.4f} meanP={sub.TargetRAUC.mean():.4f}")

json.dump({k: {"cv_seedmean_comb": v[0], "cv_seedmin_comb": v[1]} for k, v in res.items()},
          open("experiments/final_selection.json", "w"), indent=1)
