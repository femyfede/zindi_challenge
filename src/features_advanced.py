"""Advanced feature engineering: location-specific climate normals, anomalies, richer interactions."""
import numpy as np
import pandas as pd


def add_features_advanced(df: pd.DataFrame, location_climatology: dict = None) -> pd.DataFrame:
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
    # year trend
    d["year_norm"] = (d["death_year"] - 2007) / (2022 - 2007)

    # ---- age ----
    d["age_log1p"] = np.log1p(d["age"])
    d["age_sqrt"] = np.sqrt(d["age"])
    d["age_sq"] = d["age"] ** 2
    d["is_infant"] = (d["age"] <= 1).astype(int)
    d["is_under5"] = (d["age"] <= 5).astype(int)
    d["is_child"] = ((d["age"] > 5) & (d["age"] <= 18)).astype(int)
    d["is_young_adult"] = ((d["age"] > 18) & (d["age"] <= 40)).astype(int)
    d["is_middle"] = ((d["age"] > 40) & (d["age"] <= 60)).astype(int)
    d["is_elderly"] = (d["age"] >= 60).astype(int)

    # ---- temperature ----
    d["temp_range"] = d["max_temperature"] - d["min_temperature"]
    d["t_anom_7_30"] = d["tavg_7d"] - d["tavg_30d"]
    d["t_anom_30_90"] = d["tavg_30d"] - d["tavg_90d"]
    d["heat_30d"] = d["tmax_30d"] - d["tavg_30d"]
    d["cold_30d"] = d["tavg_30d"] - d["tmin_30d"]
    d["extreme_heat"] = (d["tmax_30d"] >= 32).astype(int)
    d["extreme_cold"] = (d["tmin_30d"] <= 14).astype(int)
    d["temp_var_30d"] = d["temp_range_mean_30d"]  # proxy for variability

    # ---- rainfall ----
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
    d["rain_days_log"] = np.log1p(d["rain_days_30d"])

    # ---- vegetation ----
    d["ndvi_change"] = d["ndvi_30d"] - d["ndvi_90d"]
    d["ndvi_ratio"] = d["ndvi_30d"] / (d["ndvi_90d"] + 1e-6)
    d["low_veg"] = (d["ndvi_30d"] < 0.5).astype(int)
    d["veg_stress"] = (d["ndvi_30d"] < d["ndvi_90d"] - 0.05).astype(int)
    d["ndvi_anom_30_90"] = d["ndvi_30d"] - d["ndvi_90d"]

    # ---- location-specific climate anomalies (if climatology provided) ----
    if location_climatology is not None:
        loc = d["location"]
        for var in ["tavg_30d", "tavg_7d", "rain_sum_30d", "rain_sum_7d", "ndvi_30d", "tmax_30d", "tmin_30d"]:
            mean_key = f"{var}_loc_mean"
            std_key = f"{var}_loc_std"
            if mean_key in location_climatology and std_key in location_climatology:
                means = loc.map(location_climatology[mean_key])
                stds = loc.map(location_climatology[std_key]).clip(lower=1e-3)
                d[f"{var}_loc_anom"] = (d[var] - means) / stds
    
    # ---- composite stress indices ----
    d["heat_drought"] = d["extreme_heat"] * d["is_dry_30d"]
    d["cold_wet"] = d["extreme_cold"] * (d["rain_sum_30d"] > d["rain_sum_30d"].median()).astype(int)
    d["heat_rain"] = d["tmax_30d"] * np.log1p(d["rain_sum_30d"])
    d["temp_range_x_rain"] = d["temp_range_mean_30d"] * np.log1p(d["rain_sum_30d"])
    d["veg_drought"] = d["low_veg"] * d["is_dry_30d"]
    d["humid_heat"] = (d["rain_sum_30d"] > d["rain_sum_30d"].median()).astype(int) * d["extreme_heat"]

    # ---- demographic interactions ----
    is_f = (d["gender"] == "Female").astype(int)
    is_r = (d["zone"] == "Rural").astype(int)
    d["female_x_tmax"] = is_f * d["tmax_30d"]
    d["female_x_rain30"] = is_f * np.log1p(d["rain_sum_30d"])
    d["female_x_ndvi"] = is_f * d["ndvi_30d"]
    d["female_x_heat"] = is_f * d["extreme_heat"]
    d["rural_x_rain30"] = is_r * np.log1p(d["rain_sum_30d"])
    d["rural_x_tmax"] = is_r * d["tmax_30d"]
    d["rural_x_ndvi"] = is_r * d["ndvi_30d"]
    d["rural_x_dry"] = is_r * d["is_dry_30d"]
    d["under5_x_tmax"] = d["is_under5"] * d["tmax_30d"]
    d["under5_x_rain30"] = d["is_under5"] * np.log1p(d["rain_sum_30d"])
    d["under5_x_heat"] = d["is_under5"] * d["extreme_heat"]
    d["elderly_x_tmax"] = d["is_elderly"] * d["tmax_30d"]
    d["elderly_x_dry"] = d["is_elderly"] * d["is_dry_30d"]
    d["elderly_x_cold"] = d["is_elderly"] * d["extreme_cold"]
    d["elderly_x_heat"] = d["is_elderly"] * d["extreme_heat"]
    d["child_x_rain"] = d["is_child"] * np.log1p(d["rain_sum_30d"])
    
    # ---- three-way interactions ----
    d["rural_female_x_heat"] = is_r * is_f * d["extreme_heat"]
    d["rural_under5_x_rain"] = is_r * d["is_under5"] * np.log1p(d["rain_sum_30d"])
    
    return d


def compute_location_climatology(df: pd.DataFrame) -> dict:
    """Compute per-location mean/std for climate variables from training data."""
    clim_vars = ["tavg_30d", "tavg_7d", "rain_sum_30d", "rain_sum_7d", "ndvi_30d", "tmax_30d", "tmin_30d"]
    result = {}
    for var in clim_vars:
        means = df.groupby("location")[var].mean()
        stds = df.groupby("location")[var].std().clip(lower=1e-3)
        result[f"{var}_loc_mean"] = means.to_dict()
        result[f"{var}_loc_std"] = stds.to_dict()
    return result


ADV_NUM = [
    "age", "avg_temperature", "max_temperature", "min_temperature", "precipitation",
    "rain_sum_7d", "rain_sum_30d", "rain_sum_90d", "rain_days_30d", "max_daily_rain_30d",
    "tavg_7d", "tavg_30d", "tavg_90d", "tmax_30d", "tmin_30d", "temp_range_mean_30d",
    "ndvi_30d", "ndvi_90d", "elevation", "slope",
    "death_year", "death_month", "month_sin", "month_cos", "is_long_rains",
    "is_short_rains", "is_wet", "year_norm",
    "age_log1p", "age_sqrt", "age_sq", "is_infant", "is_under5", "is_child",
    "is_young_adult", "is_middle", "is_elderly",
    "temp_range", "t_anom_7_30", "t_anom_30_90", "heat_30d", "cold_30d",
    "extreme_heat", "extreme_cold", "temp_var_30d",
    "rain_intensity_30d", "rain_7d_share", "rain_30_90_diff", "dry_days_30d",
    "is_dry_30d", "heavy_rain", "very_heavy_rain",
    "rain_sum_7d_log", "rain_sum_30d_log", "rain_sum_90d_log", "rain_days_log",
    "ndvi_change", "ndvi_ratio", "low_veg", "veg_stress", "ndvi_anom_30_90",
    "heat_drought", "cold_wet", "heat_rain", "temp_range_x_rain", "veg_drought", "humid_heat",
    "female_x_tmax", "female_x_rain30", "female_x_ndvi", "female_x_heat",
    "rural_x_rain30", "rural_x_tmax", "rural_x_ndvi", "rural_x_dry",
    "under5_x_tmax", "under5_x_rain30", "under5_x_heat",
    "elderly_x_tmax", "elderly_x_dry", "elderly_x_cold", "elderly_x_heat",
    "child_x_rain",
    "rural_female_x_heat", "rural_under5_x_rain",
]
ADV_CAT = ["gender", "zone"]