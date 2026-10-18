"""Benchmark harness: repeated stratified CV with the exact competition metric."""
import os, sys, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42
TGT = "is_climate_sensitive"


def load():
    tr = pd.read_csv("Train.csv")
    te = pd.read_csv("Test.csv")
    cl = pd.read_csv("climate_features.csv").drop(columns=["deathdate"])
    tr = tr.merge(cl, on="ID", how="left", validate="one_to_one")
    te = te.merge(cl, on="ID", how="left", validate="one_to_one")
    return tr, te


def score(y, p):
    f1 = f1_score(y, (p >= 0.5).astype(int), zero_division=0)
    auc = roc_auc_score(y, p)
    return f1, auc, 0.6 * f1 + 0.4 * auc


def repeated_cv(fit_predict, X, y, n_splits=5, repeats=3, seeds=(42, 7, 123)):
    """fit_predict(Xtr, ytr, Xva) -> proba. Returns per-repeat oof metrics + pooled oof."""
    rows, oofs = [], []
    for s in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        oof = np.zeros(len(X))
        for a, b in skf.split(X, y):
            oof[b] = fit_predict(X.iloc[a], y.iloc[a], X.iloc[b])
        f1, auc, comb = score(y, oof)
        rows.append(dict(seed=s, f1=f1, auc=auc, combined=comb))
        oofs.append(oof)
    df = pd.DataFrame(rows)
    m = dict(model="", f1=df.f1.mean(), auc=df.auc.mean(), combined=df.combined.mean(),
             f1_sd=df.f1.std(), auc_sd=df.auc.std(), comb_sd=df.combined.std())
    return m, df, np.array(oofs)


def show(df, title=""):
    d = df.copy()
    d["f1"] = d.f1.map("{:.4f}".format); d["auc"] = d.auc.map("{:.4f}".format)
    d["combined"] = d.combined.map("{:.4f}".format)
    if "f1_sd" in d.columns:
        d["f1_sd"] = d.f1_sd.map("{:.4f}".format); d["auc_sd"] = d.auc_sd.map("{:.4f}".format)
        d["comb_sd"] = d.comb_sd.map("{:.4f}".format)
    print("\n=== " + title + " ===")
    print(d.sort_values("combined", ascending=False).to_string(index=False))
