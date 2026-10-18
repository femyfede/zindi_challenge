"""Feature engineering — only scientifically/statistically defensible transforms of
available columns. No target use, no cross-row aggregation (leakage-safe row-wise ops)."""
import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # --- temporal (from deathdate; parsed row-wise) ---
    dt = pd.to_datetime(d["deathdate"], errors="coerce")
    d["death_year"] = dt.dt.year
    d["death_month"] = dt.dt.month
    # Uganda bimodal rains: long rains Mar-May, short rains Sep-Nov
    d["is_long_rains"] = d["death_month"].isin([3, 4, 5]).astype(int)
    d["is_short_rains"] = d["death_month"].isin([9, 10, 11]).astype(int)
    d["is_wet_season"] = d["is_long_rains"] | d["is_short_rains"]

    # --- temperature group ---
    d["temp_range"] = d["max_temperature"] - d["min_temperature"]  # diurnal range, heat stress
    d["t_anom_7_30"] = d["tavg_7d"] - d["tavg_30d"]     # short-term warm/cool spell
    d["t_anom_30_90"] = d["tavg_30d"] - d["tavg_90d"]   # medium-term anomaly
    d["heat_30d"] = d["tmax_30d"] - d["tavg_30d"]       # peak intensity above monthly mean

    # --- rainfall group ---
    d["rain_intensity_30d"] = d["rain_sum_30d"] / (d["rain_days_30d"] + 1.0)
    d["rain_7d_share"] = d["rain_sum_7d"] / (d["rain_sum_30d"] + 1.0)  # recent vs monthly
    d["rain_30_90_diff"] = d["rain_sum_30d"] - d["rain_sum_90d"] / 3.0  # monthly vs 90d avg pace
    d["is_dry_30d"] = (d["rain_days_30d"] <= 2).astype(int)             # dry-period indicator
    d["heavy_rain_flag"] = (d["max_daily_rain_30d"] >= 25.0).astype(int)  # intense storm

    # --- vegetation group ---
    d["ndvi_change"] = d["ndvi_30d"] - d["ndvi_90d"]      # greening/browning
    d["low_veg"] = (d["ndvi_30d"] < 0.5).astype(int)      # stressed vegetation

    # --- demographic group ---
    bins = [-1, 1, 5, 18, 40, 60, 500]
    labels = ["infant", "under5", "child", "adult", "middle", "elderly"]
    d["age_group"] = pd.cut(d["age"], bins=bins, labels=labels)
    d["is_under5"] = (d["age"] <= 5).astype(int)
    d["is_elderly"] = (d["age"] >= 60).astype(int)
    d["is_infant"] = (d["age"] <= 1).astype(int)

    # --- defensible interactions (vulnerability x exposure) ---
    d["age_x_tmax"] = d["age"] * d["tmax_30d"]
    d["under5_x_rain30"] = d["is_under5"] * np.log1p(d["rain_sum_30d"])
    d["elderly_x_tmax"] = d["is_elderly"] * d["tmax_30d"]
    d["precip_x_temp"] = d["precipitation"] * d["avg_temperature"]

    return d


# Feature sets for ablation (geo excluded vs included)
BASE_NUMERIC = [
    "age", "avg_temperature", "max_temperature", "min_temperature", "precipitation",
    "rain_sum_7d", "rain_sum_30d", "rain_sum_90d", "rain_days_30d", "max_daily_rain_30d",
    "tavg_7d", "tavg_30d", "tavg_90d", "tmax_30d", "tmin_30d", "temp_range_mean_30d",
    "ndvi_30d", "ndvi_90d",
    "death_year", "death_month", "is_long_rains", "is_short_rains", "is_wet_season",
    "temp_range", "t_anom_7_30", "t_anom_30_90", "heat_30d",
    "rain_intensity_30d", "rain_7d_share", "rain_30_90_diff", "is_dry_30d", "heavy_rain_flag",
    "ndvi_change", "low_veg",
    "is_under5", "is_elderly", "is_infant",
    "age_x_tmax", "under5_x_rain30", "elderly_x_tmax", "precip_x_temp",
]
BASE_CATEGORICAL = ["gender", "zone", "age_group"]
GEO_NUMERIC = ["latitude", "longitude", "elevation", "slope"]

FEATURES_NO_GEO_NUM = BASE_NUMERIC
FEATURES_NO_GEO_CAT = BASE_CATEGORICAL
FEATURES_WITH_GEO_NUM = BASE_NUMERIC + GEO_NUMERIC
FEATURES_WITH_GEO_CAT = BASE_CATEGORICAL

# No-climate baseline (demographic + geo-numeric + time only) for interpretability section
NOCLIMATE_NUM = ["age", "death_year", "death_month", "is_under5", "is_elderly", "is_infant"]
NOCLIMATE_CAT = ["gender", "zone", "age_group"]
