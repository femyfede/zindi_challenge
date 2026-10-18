"""Sweep 4: age x year-bin cells (finer than the pre/post split) + probability-averaged ensemble."""
import numpy as np, pandas as pd, warnings, time, itertools, json
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from bench import load
from features2 import build

BREAK = pd.Timestamp("2017-07-01")
SEEDS = (42, 7, 123)
tr, te = load()
y = tr["is_climate_sensitive"].values
B20 = [-0.01, 0, 1, 2, 3, 4, 5, 8, 12, 18, 25, 35, 45, 55, 62, 68, 73, 78, 85, 200]


def frame(df, cfg):
    d = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["deathdate"])
    post = (dt >= BREAK).astype(float)
    d["gender"] = df["gender"]; d["zone"] = df["zone"]
    d["age"] = df["age"].astype(float)
    d["year_cont"] = (dt.dt.year + (dt.dt.dayofyear - 1) / 365.25).astype(float)
    d["doy"] = dt.dt.dayofyear.astype(float)
    d["age_x_post"] = df["age"].astype(float) * post
    ab = pd.cut(df["age"], B20).astype(str)
    per = np.where(dt >= BREAK, "post", "pre")
    d["cell"] = ab + "|" + per
    if cfg.get("cell_g"):
        d["cell_g"] = d["cell"] + "|" + df["gender"].astype(str)
    w = cfg.get("ywin", 0)
    if w:
        yb = (dt.dt.year // w).astype(str)
        d["cell_y"] = ab + "|" + yb
        if cfg.get("cell_y_g"):
            d["cell_yg"] = d["cell_y"] + "|" + df["gender"].astype(str)
    if cfg.get("cell_pg"):          # age x period x gender
        d["cell_pg"] = d["cell"] + "|" + df["gender"].astype(str)
    if cfg.get("cell_pz"):
        d["cell_pz"] = d["cell"] + "|" + df["zone"].astype(str)
    return d


def spec(cfg):
    t = [("sp_age", SplineTransformer(n_knots=cfg.get("k_age", 10), degree=3, include_bias=False), ["age"]),
         ("sp_agepost", SplineTransformer(n_knots=cfg.get("k_age", 10), degree=3, include_bias=False), ["age_x_post"]),
         ("sp_year", SplineTransformer(n_knots=cfg.get("k_year", 6), degree=3, include_bias=False), ["year_cont"]),
         ("sp_doy", SplineTransformer(n_knots=6, degree=3, include_bias=False), ["doy"])]
    cats = ["gender", "zone", "cell"]
    for k in ("cell_g", "cell_y", "cell_yg", "cell_pg", "cell_pz"):
        if cfg.get(k):
            cats.append(k)
    t.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
    return ColumnTransformer(t)


def pipe(cfg):
    return Pipeline([("sp", spec(cfg)), ("clf", LogisticRegression(max_iter=10000, C=cfg["C"]))])


SPLITS = {s: list(StratifiedKFold(5, shuffle=True, random_state=s).split(tr, y)) for s in SEEDS}


def oof(cfg):
    D = frame(tr, cfg)
    outs = []
    for s in SEEDS:
        o = np.zeros(len(D))
        for a, b in SPLITS[s]:
            m = pipe(cfg); m.fit(D.iloc[a], y[a]); o[b] = m.predict_proba(D.iloc[b])[:, 1]
        outs.append(o)
    return np.array(outs)


def met(p):
    per = []
    for s in SEEDS:
        o = p[list(p).index(s)] if isinstance(p, dict) else None
    return per


def score_mean(oofs):
    per = [(f1_score(y, (o >= 0.5).astype(int), zero_division=0), roc_auc_score(y, o)) for o in oofs]
    m = oofs.mean(axis=0)
    f1 = f1_score(y, (m >= 0.5).astype(int), zero_division=0); auc = roc_auc_score(y, m)
    return dict(f1=f1, auc=auc, comb=0.6 * f1 + 0.4 * auc,
                sm=float(np.mean([0.6 * a + 0.4 * b for a, b in per])), pos=float((m >= 0.5).mean()))


CFGS = []
for ywin in [0, 3, 4, 5]:
    for extra in [{}, {"cell_g": 1}, {"cell_pg": 1}, {"cell_pz": 1}]:
        for C in [0.02, 0.03, 0.05]:
            c = {"C": C, "ywin": ywin, **extra}
            if ywin and "cell_y_g" in extra:
                pass
            CFGS.append(c)

rows, O = [], {}
t0 = time.time()
for cfg in CFGS:
    tag = ",".join(f"{k}={v}" for k, v in cfg.items() if k != "C" and v)
    tag = tag or "base"
    try:
        oo = oof(cfg)
    except Exception as e:
        print("FAIL", tag, cfg["C"], repr(e)[:80], flush=True); continue
    r = score_mean(oo); r.update(cfg); r["tag"] = tag
    rows.append(r); O[(tag, cfg["C"])] = oo
    print(f"{tag:28s} C={cfg['C']:<5} f1={r['f1']:.4f} auc={r['auc']:.4f} comb={r['comb']:.4f} "
          f"sm={r['sm']:.4f} pos={r['pos']:.3f}", flush=True)
    pd.DataFrame(rows).drop(columns=[k for k in CFGS[0] if k not in ("C", "cell_g", "ywin")], errors="ignore") \
        .to_csv("experiments/lin4.csv", index=False)

res = pd.DataFrame(rows).sort_values("sm", ascending=False)
res.to_csv("experiments/lin4.csv", index=False)
print(f"\n({time.time()-t0:.0f}s)\nTOP 15:")
print(res.head(15).to_string(index=False))

# ---------- probability-averaged ensembles ----------
print("\n=== PROBABILITY-AVERAGED ENSEMBLES (top configs, uniform) ===")
top = list(O)[-0:] 
keys = list(O)
ranked = sorted(rows, key=lambda r: -r["sm"])
topk = [((r["tag"], r["C"])) for r in ranked[:12]]
best = []
for k in range(1, 7):
    for combo in itertools.combinations(topk, k):
        if k == 1:
            continue
        p = np.mean([O[c] for c in combo], axis=0)
        r = score_mean(p)
        best.append((r["sm"], r["comb"], r["f1"], r["auc"], r["pos"], combo))
best.sort(reverse=True)
for sm, cb, f1, auc, pos, combo in best[:15]:
    print(f"  sm={sm:.4f} comb={cb:.4f} f1={f1:.4f} auc={auc:.4f} pos={pos:.3f}  "
          f"[{' | '.join(f'{a}@{c}' for a, c in combo)}]")
json.dump([[str(c), sm, cb] for sm, cb, f1, auc, pos, c in best[:25]],
          open("experiments/lin4_ens.json", "w"), indent=1)
