"""Aggressive feature set v3: location encoding, spatial, richer climate, interactions."""
import numpy as np
import pandas as pd


def add_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # ---- temporal ----
    dt = pd.to_datetime(d["deathdate"], errors="coerce")
    d["death_year"] = dt.dt.year
    d["death_month"] = dt.dt.month
    d["month_sin"] = np.sin(2 * np.pi * d["death_month"] / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * d["death_month"] / 12.0)
    d["is_long_rains"] = d["death_month"].isin([3, 4, 5]).astype(int)
    d["is_short_rains"] = d["death_month"].isin([9, 10, 11]).astype(int)
    d["is_wet"] = d["is_long_rains"] | d["is_short_rains"]

    # ---- age transforms ----
    d["age_log1p"] = np.log1p(d["age"])
    d["age_sqrt"] = np.sqrt(d["age"])
    d["age_sq"] = d["age"] ** 2
    d["is_infant"] = (d["age"] <= 1).astype(int)
    d["is_under5"] = (d["age"] <= 5).astype(int)
    d["is_child"] = ((d["age"] > 5) & (d["age"] <= 18)).astype(int)
    d["is_elderly"] = (d["age"] >= 60).astype(int)
    # age bins as numeric
    d["age_bin"] = pd.cut(d["age"], bins=[-1, 1, 5, 18, 40, 60, 200],
                          labels=[0, 1, 2, 3, 4, 5]).astype(float)

    # ---- temperature group ----
    d["temp_range"] = d["max_temperature"] - d["min_temperature"]
    d["t_anom_7_30"] = d["tavg_7d"] - d["tavg_30d"]
    d["t_anom_30_90"] = d["tavg_30d"] - d["tavg_90d"]
    d["heat_30d"] = d["tmax_30d"] - d["tavg_30d"]
    d["cold_30d"] = d["tavg_30d"] - d["tmin_30d"]
    d["extreme_heat"] = (d["tmax_30d"] >= 32).astype(int)
    d["extreme_cold"] = (d["tmin_30d"] <= 14).astype(int)

    # ---- rainfall group ----
    d["rain_intensity_30d"] = d["rain_sum_30d"] / (d["rain_days_30d"] + 1.0)
    d["rain_7d_share"] = d["rain_sum_7d"] / (d["rain_sum_30d"] + 1.0)
    d["rain_30_90_diff"] = d["rain_sum_30d"] - d["rain_sum_90d"] / 3.0
    d["dry_days_30d"] = 30 - d["rain_days_30d"]
    d["is_dry_30d"] = (d["rain_days_30d"] <= 2).astype(int)
    d["heavy_rain"] = (d["max_daily_rain_30d"] >= 25).astype(int)
    d["very_heavy_rain"] = (d["max_daily_rain_30d"] >= 50).astype(int)
    d["rain_sum_7d_log"] = np.log1p(d["rain_sum_7d"])
    d["rain_sum_30d_log"] = np.log1p(d["rain_sum_30d"])
    d["rain_sum_90d_log"] = np.log1p(d["rain_sum_90d"])

    # ---- vegetation ----
    d["ndvi_change"] = d["ndvi_30d"] - d["ndvi_90d"]
    d["ndvi_ratio"] = d["ndvi_30d"] / (d["ndvi_90d"] + 1e-6)
    d["low_veg"] = (d["ndvi_30d"] < 0.5).astype(int)
    d["veg_stress"] = (d["ndvi_30d"] < d["ndvi_90d"] - 0.05).astype(int)

    # ---- climate stress composites ----
    d["heat_drought"] = d["extreme_heat"] * d["is_dry_30d"]
    d["cold_wet"] = d["extreme_cold"] * (d["rain_sum_30d"] > d["rain_sum_30d"].median()).astype(int)
    d["heat_rain"] = d["tmax_30d"] * np.log1p(d["rain_sum_30d"])
    d["temp_range_x_rain"] = d["temp_range_mean_30d"] * np.log1p(d["rain_sum_30d"])

    # ---- demographic interactions ----
    is_f = (d["gender"] == "Female").astype(int)
    is_r = (d["zone"] == "Rural").astype(int)
    d["female_x_tmax"] = is_f * d["tmax_30d"]
    d["female_x_rain30"] = is_f * np.log1p(d["rain_sum_30d"])
    d["female_x_ndvi"] = is_f * d["ndvi_30d"]
    d["rural_x_rain30"] = is_r * np.log1p(d["rain_sum_30d"])
    d["rural_x_tmax"] = is_r * d["tmax_30d"]
    d["rural_x_ndvi"] = is_r * d["ndvi_30d"]
    d["under5_x_tmax"] = d["is_under5"] * d["tmax_30d"]
    d["under5_x_rain30"] = d["is_under5"] * np.log1p(d["rain_sum_30d"])
    d["elderly_x_tmax"] = d["is_elderly"] * d["tmax_30d"]
    d["elderly_x_dry"] = d["is_elderly"] * d["is_dry_30d"]
    d["elderly_x_cold"] = d["is_elderly"] * d["extreme_cold"]

    # ---- geographic (raw) ----
    # These will be encoded separately with target encoding
    return d


# Column groups for reference
V3_NUM = [
    "age", "avg_temperature", "max_temperature", "min_temperature", "precipitation",
    "rain_sum_7d", "rain_sum_30d", "rain_sum_90d", "rain_days_30d", "max_daily_rain_30d",
    "tavg_7d", "tavg_30d", "tavg_90d", "tmax_30d", "tmin_30d", "temp_range_mean_30d",
    "ndvi_30d", "ndvi_90d", "elevation", "slope",
    "death_year", "death_month", "month_sin", "month_cos", "is_long_rains",
    "is_short_rains", "is_wet",
    "age_log1p", "age_sqrt", "age_sq", "age_bin", "is_infant", "is_under5",
    "is_child", "is_elderly",
    "temp_range", "t_anom_7_30", "t_anom_30_90", "heat_30d", "cold_30d",
    "extreme_heat", "extreme_cold",
    "rain_intensity_30d", "rain_7d_share", "rain_30_90_diff", "dry_days_30d",
    "is_dry_30d", "heavy_rain", "very_heavy_rain",
    "rain_sum_7d_log", "rain_sum_30d_log", "rain_sum_90d_log",
    "ndvi_change", "ndvi_ratio", "low_veg", "veg_stress",
    "heat_drought", "cold_wet", "heat_rain", "temp_range_x_rain",
    "female_x_tmax", "female_x_rain30", "female_x_ndvi",
    "rural_x_rain30", "rural_x_tmax", "rural_x_ndvi",
    "under5_x_tmax", "under5_x_rain30", "elderly_x_tmax",
    "elderly_x_dry", "elderly_x_cold",
]
V3_CAT = ["gender", "zone"]
V3_GEO_RAW = ["location", "latitude", "longitude", "elevation", "slope"]