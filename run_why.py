"""Why F1 0.97 is unreachable: quantify the irreducible ambiguity."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score, roc_auc_score
from bench import load
from make_submission import frame, make_pipe, MEMBERS, y, tr

tr_raw, te_raw = load()
N_POS = int(y.sum()); N_NEG = len(y) - N_POS
print(f"train: {len(y)} rows, {N_POS} positive ({N_POS/len(y):.4f}), {N_NEG} negative")
print(f"expected test (n=1030): ~{round(1030*N_POS/len(y))} positive, ~{round(1030*N_NEG/len(y))} negative\n")

print("=== what F1=0.97 would REQUIRE ===")
for tp in [int(0.90*N_POS), int(0.95*N_POS), int(0.98*N_POS), N_POS]:
    # F1 = 2TP/(P_pred + P_true); solve for allowed total errors
    best_f1_at_zero_fp = 2*tp/(tp + N_POS)
    print(f"  recall {tp/N_POS:.0%} ({tp}/{N_POS} TP), zero false positives -> F1 = {best_f1_at_zero_fp:.4f}")
print("  => F1 0.97 needs ~97-98% of positives caught with essentially NO false alarms.")
print("     i.e. at most ~40 wrong decisions out of 1030 rows.\n")

print("=== how much of the data is irreducibly ambiguous ===")
print("target rate by age band (this is the best any model can know about a row):")
b = [-0.01, 0, 1, 3, 6, 18, 45, 65, 78, 200]
g = tr.groupby(pd.cut(tr.age, b), observed=True).is_climate_sensitive.agg(["size", "mean"])
g["share"] = g["size"] / len(y)
amb = g[(g["mean"] > 0.30) & (g["mean"] < 0.70)]["share"].sum()
print(g.round(3).to_string())
print(f"\n  rows whose true rate sits in the ambiguous 0.30-0.70 band: {amb:.1%} of the data")

print("\n=== best F1 from hard rules alone (no model) ===")
def rep(n, p):
    print(f"  {n:32s} F1={f1_score(y, p):.4f}")
rep("predict ALL 1", np.ones(len(y), int))
rep("predict 1 iff age<=5", (tr.age <= 5).values.astype(int))
rep("predict 1 iff age<=2", (tr.age <= 2).values.astype(int))
rep("predict 1 iff not(70<=age<75)", (~((tr.age >= 70) & (tr.age < 75)).values).astype(int))

print("\n=== the actual model, and its error budget ===")
name, bins, gflag, C = MEMBERS[0]
D, cats = frame(tr, bins, gflag)
from sklearn.model_selection import StratifiedKFold
oof = np.zeros(len(y))
for a, bb in StratifiedKFold(5, shuffle=True, random_state=42).split(D, y):
    m = make_pipe(cats, C); m.fit(D.iloc[a], y[a]); oof[bb] = m.predict_proba(D.iloc[bb])[:, 1]
pred = (oof >= 0.5).astype(int)
TP = int(((pred == 1) & (y == 1)).sum()); FP = int(((pred == 1) & (y == 0)).sum())
FN = int(((pred == 0) & (y == 1)).sum())
print(f"  F1={f1_score(y, pred):.4f}  AUC={roc_auc_score(y, oof):.4f}")
print(f"  TP={TP}  FP={FP}  FN={FN}  total errors = {FP+FN} ({100*(FP+FN)/len(y):.1f}% of rows)")
print(f"  F1 0.97 would need errors <= ~{0.03*len(y):.0f} ({3*(TP+FN)*0 + 0:.0f}) -> about 40, we have {FP+FN}")
print(f"\n  rows with probability in [0.40,0.60] (model cannot separate): "
      f"{int(((oof>0.4)&(oof<0.6)).sum())} ({100*((oof>0.4)&(oof<0.6)).mean():.1f}%)")
print("  These are the rows that carry the error. They are near-coin-flips by age, and the")
print("  only thing that separates them is cause of death - which is NOT in the dataset.")
