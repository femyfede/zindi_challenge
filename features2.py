"""Feature builder v2 — built around the age x period structural break."""
import numpy as np
import pandas as pd

BREAK = pd.Timestamp("2017-07-01")

AGE_BINS = [-0.01, 0, 1, 3, 6, 12, 18, 30, 45, 60, 70, 75, 85, 200]
AGE_LABELS = ["age_0", "age_1", "age_2_3", "age_4_6", "age_7_12", "age_13_18",
              "age_19_30", "age_31_45", "age_46_60", "age_61_70", "age_71_75",
              "age_76_85", "age_85p"]

CLIM = ["avg_temperature", "max_temperature", "min_temperature", "precipitation",
        "rain_sum_7d", "rain_sum_30d", "rain_sum_90d", "rain_days_30d",
        "max_daily_rain_30d", "tavg_7d", "tavg_30d", "tavg_90d", "tmax_30d",
        "tmin_30d", "temp_range_mean_30d", "ndvi_30d", "ndvi_90d"]
GEO = ["latitude", "longitude", "elevation", "slope"]


def build(df, use=("demo", "time", "clim")):
    d = df.copy()
    dt = pd.to_datetime(d["deathdate"])
    d["d"] = dt

    # ---------- demographic ----------
    if "demo" in use:
        d["age_log"] = np.log1p(d["age"])
        d["age_sqrt"] = np.sqrt(d["age"])
        d["age_band"] = pd.cut(d["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
        d["is_age0"] = (d["age"] == 0).astype(int)
        d["is_u1"] = (d["age"] <= 1).astype(int)
        d["is_u5"] = (d["age"] <= 5).astype(int)
        d["is_child"] = ((d["age"] > 5) & (d["age"] <= 18)).astype(int)
        d["is_adult"] = ((d["age"] > 18) & (d["age"] <= 60)).astype(int)
        d["is_old"] = (d["age"] > 60).astype(int)
        d["is_female"] = (d["gender"] == "Female").astype(int)
        d["is_rural"] = (d["zone"] == "Rural").astype(int)

    # ---------- time / period ----------
    if "time" in use:
        d["year"] = dt.dt.year
        d["month"] = dt.dt.month
        d["doy"] = dt.dt.dayofyear
        d["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        d["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
        d["doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
        d["doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
        d["is_post"] = (dt >= BREAK).astype(int)
        d["yrs_since_break"] = ((dt - BREAK).dt.days / 365.25).clip(lower=-10)
        d["year_cont"] = dt.dt.year + (dt.dt.dayofyear - 1) / 365.25

    # ---------- climate ----------
    if "clim" in use:
        d["temp_range"] = d["max_temperature"] - d["min_temperature"]
        d["t_anom"] = d["tavg_7d"] - d["tavg_90d"]
        d["rain_30_90"] = d["rain_sum_30d"] - d["rain_sum_90d"] / 3.0
        d["ndvi_chg"] = d["ndvi_30d"] - d["ndvi_90d"]
        d["dry_days"] = 30 - d["rain_days_30d"]
        d["rain_log"] = np.log1p(d["rain_sum_30d"])
        d["is_dry"] = (d["rain_days_30d"] <= 2).astype(int)

    # ---------- age x period interaction (the dominant structure) ----------
    if "demo" in use and "time" in use:
        d["cell_age_period"] = d["age_band"] + "_" + np.where(dt >= BREAK, "post", "pre")
        d["u5_x_post"] = ((d["age"] <= 5).astype(int) * (dt >= BREAK).astype(int))
        d["age_x_year"] = d["age"] * dt.dt.year
        d["age_x_post"] = d["age"] * (dt >= BREAK).astype(int)

    drop = ["ID", "deathdate", "d", "location", "hot_days_30d", "is_climate_sensitive"]
    if "clim" not in use:
        drop += CLIM
    if "geo" not in use:
        drop += GEO
    cat = ["gender", "zone"]
    if "demo" in use:
        cat += ["age_band", "cell_age_period"]
    drop = [c for c in drop if c in d.columns and c not in cat]
    d = d.drop(columns=drop)
    num = [c for c in d.columns if c not in cat]
    return d[num + cat], num, cat
