"""Feature set v2: adds defensible row-wise features to v1. No target use."""
import numpy as np
import pandas as pd


def add_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # cyclic month (seasonality without arbitrary jumps Dec->Jan)
    d["month_sin"] = np.sin(2 * np.pi * d["death_month"] / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * d["death_month"] / 12.0)
    # age nonlinearities (risk is U-shaped: high infant, low middle, higher elderly non-climate)
    d["age_log1p"] = np.log1p(d["age"])
    d["age_sq"] = d["age"] ** 2
    # gender x climate (differential vulnerability, association only)
    is_f = (d["gender"] == "Female").astype(int)
    d["female_x_tmax"] = is_f * d["tmax_30d"]
    d["female_x_rain30"] = is_f * np.log1p(d["rain_sum_30d"])
    # zone x climate
    is_r = (d["zone"] == "Rural").astype(int)
    d["rural_x_rain30"] = is_r * np.log1p(d["rain_sum_30d"])
    d["peri_x_tmax"] = (1 - is_r) * d["tmax_30d"]
    # vegetation x water stress combo
    d["ndvi_x_rain"] = d["ndvi_30d"] * np.log1p(d["rain_sum_30d"])
    # dry-spell severity + rainfall persistence
    d["dry_severity"] = np.clip(5 - d["rain_days_30d"], 0, 5)
    d["rain_persist"] = d["rain_sum_30d"] - d["rain_sum_7d"] * (30.0 / 7.0)
    # cold stress flag (low minimum temperature exposure)
    d["cold_flag"] = (d["tmin_30d"] < 15.0).astype(int)
    # elderly x dry (older-age vulnerability during dry periods)
    d["elderly_x_dry"] = d["is_elderly"] * d["dry_severity"]
    return d


V2_EXTRA = ["month_sin", "month_cos", "age_log1p", "age_sq",
            "female_x_tmax", "female_x_rain30", "rural_x_rain30", "peri_x_tmax",
            "ndvi_x_rain", "dry_severity", "rain_persist", "cold_flag",
            "elderly_x_dry"]
