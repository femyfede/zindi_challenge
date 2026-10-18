"""DATE-LEVEL TARGET ENCODING (leakage-safe).

Discovery: avg_temperature / max_temperature / precipitation are pure functions of
deathdate in this dataset, and 51.9% of test rows share an exact deathdate with
train rows. Dates with >=5 train records show label base rates from 0.20 to 1.00,
i.e. the date carries genuine label information.

We therefore build date-level aggregates from TRAINING ROWS ONLY:
  - smoothed target mean per date   (computed inside the CV fold)
  - date record count
  - age-adjusted date target rate  (residual after removing the age effect)
  - for exact (date, location) matches, the local cluster rate

No test label is ever used. For a validation row the encoding is built from the
training fold only, so the row's own label cannot leak.
"""
import numpy as np
import pandas as pd

GLOBAL_PRIOR = 0.6507  # train base rate, only used for smoothing


def date_rate_table(df_tr, y_tr, smooth=8.0, prior=GLOBAL_PRIOR):
    """Smoothed target rate per deathdate from training rows only."""
    g = pd.DataFrame({"d": df_tr["deathdate"], "y": y_tr}).groupby("d")["y"].agg(["sum", "count"])
    g["rate"] = (g["sum"] + smooth * prior) / (g["count"] + smooth)
    return g


def age_residual_rate_table(df_tr, y_tr, smooth=8.0, prior=GLOBAL_PRIOR):
    """Date-level rate AFTER removing each row's expected rate from age+gender+zone.

    Uses the fitted age-band prior from the same training fold, so the residual
    isolates date-specific signal rather than re-learning the age effect.
    """
    b = pd.cut(df_tr["age"], [-1, 1, 5, 18, 40, 60, 500],
               labels=["infant", "u5", "c5_18", "a18_40", "a40_60", "a60"]).astype(str)
    key = b + "|" + df_tr["gender"] + "|" + df_tr["zone"]
    exp = pd.DataFrame({"k": key, "y": y_tr}).groupby("k")["y"].mean()  # expected rate
    resid = pd.Series(y_tr - key.map(exp).values, index=df_tr.index)
    t = pd.DataFrame({"d": df_tr["deathdate"].values, "r": resid.values}).groupby("d")["r"].agg(["mean", "count"])
    t["res_rate"] = (t["mean"] + smooth * 0.0) / (t["count"] + smooth)
    t["res_sum"] = t["mean"] * t["count"]
    return t, exp


def add_date_features(df_target, df_fit, y_fit, smooth=8.0, use_age_resid=True):
    """Attach date features to df_target, computing statistics ONLY from df_fit/y_fit.

    df_target may be validation rows or test rows; df_fit is always training rows
    from the current fold.
    """
    out = pd.DataFrame(index=df_target.index)
    tab = date_rate_table(df_fit, y_fit, smooth=smooth)
    out["date_rate"] = df_target["deathdate"].map(tab["rate"]).astype(float).fillna(GLOBAL_PRIOR)
    out["date_cnt"] = df_target["deathdate"].map(tab["count"]).astype(float).fillna(0.0)
    out["date_seen"] = out["date_cnt"] > 0
    out["date_cnt"] = np.log1p(out["date_cnt"])

    if use_age_resid:
        rt, _ = age_residual_rate_table(df_fit, y_fit, smooth=smooth)
        out["date_resid"] = df_target["deathdate"].map(rt["res_rate"]).astype(float).fillna(0.0)
        out["date_resid_cnt"] = df_target["deathdate"].map(rt["count"]).astype(float).fillna(0.0)
        out["date_resid_cnt"] = np.log1p(out["date_resid_cnt"])

    # exact (date, location) cluster rate from training rows only
    key_fit = df_fit["deathdate"].astype(str) + "|" + df_fit["location"].astype(str)
    g = pd.DataFrame({"k": key_fit, "y": y_fit}).groupby("k")["y"].agg(["sum", "count"])
    g["rate"] = (g["sum"] + 3.0 * GLOBAL_PRIOR) / (g["count"] + 3.0)
    key_t = df_target["deathdate"].astype(str) + "|" + df_target["location"].astype(str)
    out["dl_rate"] = key_t.map(g["rate"]).astype(float)
    out["dl_cnt"] = key_t.map(g["count"]).astype(float).fillna(0.0)
    out.loc[out["dl_rate"].isna(), "dl_rate"] = out.loc[out["dl_rate"].isna(), "date_rate"]
    out["dl_cnt"] = np.log1p(out["dl_cnt"])
    return out


DATE_FEATS = ["date_rate", "date_cnt", "date_seen", "date_resid", "date_resid_cnt",
              "dl_rate", "dl_cnt"]