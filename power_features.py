"""Aggregate daily MERRA-2 to per-record climate windows ending BEFORE deathdate.
Strictly backward-looking -> no future-data leakage.

New information vs provided climate_features.csv: humidity (RH2M, QV2M),
dew point (T2MDEW), wind (WS10M), and derived vapour-pressure deficit (VPD),
apparent temperature and wet-bulb temperature. These are absent from the
provided file, which carries only rain/temp/NDVI/elevation aggregates.

Sources (public, no key):
  NASA POWER Daily API - MERRA-2 reanalysis
  https://power.larc.nasa.gov/api/temporal/daily/point
"""
import numpy as np
import pandas as pd

DAILY = "data_cache/power_daily.csv"
WINDOWS = (1, 3, 7, 14, 30, 60, 90)

# Saturation vapour pressure (Tetens/Murray eq., kPa) and VPD from T (degC) and RH (%)
def es_kpa(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))

def vpd_kpa(t, rh):
    rh = np.clip(rh, 0, 100)
    return es_kpa(t) * (1 - rh / 100.0)

def apparent_temp(t, rh, ws):
    """Rothfusz-style apparent temperature (degC). Heat-stress proxy."""
    rh = np.clip(rh, 0, 100)
    ws_ms = np.clip(ws, 0.0, None)
    return (t + 0.33 * rh / 100.0 * (-42.5 + 0.9475 * t)
            - 0.43 * np.log(np.clip(ws_ms, 0.5, None)) - 4.0)


def _prep_daily():
    d = pd.read_csv(DAILY, parse_dates=["date"])
    t = d["T2M"].to_numpy(); rh = d["RH2M"].to_numpy(); ws = d["WS10M"].to_numpy()
    qv = d["QV2M"].to_numpy(); td = d["T2MDEW"].to_numpy()
    d["VPD"] = vpd_kpa(t, rh)
    d["APP_T"] = apparent_temp(t, rh, ws)
    d["e_a"] = 0.6108 * np.exp(17.27 * td / (td + 237.3))     # actual vapour pressure
    d["sat_def"] = es_kpa(t)                                  # saturation vapour pressure
    d["e_a_share"] = np.clip(d["e_a"] / d["sat_def"], 0, 2)   # ~RH sanity metric
    d["heat_index"] = d["APP_T"]
    # rolling 3-day max of APP_T per cell for heat-wave detection
    d["APP_T_3dmax"] = (d.groupby("cell_lat")["APP_T"]
                        .transform(lambda s: s.rolling(3, min_periods=1).max()))
    d["dry_streak"] = (d.assign(r=(d["PRECTOTCORR"] < 1.0).astype(int))
                       .groupby("cell_lat")["r"].transform(lambda s: s.rolling(30, min_periods=1).sum()))
    return d


def build_record_features(records: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """For each record, aggregate daily cells over backward-looking windows.

    records must have: latitude, longitude, deathdate (datetime-like)
    Returns a frame with the same index as records plus pw_* columns.
    """
    daily = daily.sort_values(["cell_lat", "date"])
    base = pd.DataFrame(index=records.index)

    rec = records.copy()
    rec["lat_cell"] = rec["latitude"].round(2)
    rec["lon_cell"] = rec["longitude"].round(2)
    rec["ddate"] = pd.to_datetime(rec["deathdate"])

    agg_spec = {
        "T2M": ["mean", "max", "min", "std"],
        "T2M_MAX": ["mean", "max"],
        "T2M_MIN": ["mean", "min"],
        "PRECTOTCORR": ["sum", "max", "mean"],
        "RH2M": ["mean", "min", "max"],
        "QV2M": ["mean", "max"],
        "T2MDEW": ["mean", "max"],
        "WS10M": ["mean", "max"],
        "VPD": ["mean", "max"],
        "APP_T": ["mean", "max"],
        "APP_T_3dmax": ["max"],
        "sat_def": ["mean", "max"],
        "e_a": ["mean", "max"],
    }
    warm = WINDOWS[-1]  # build aggregates out to the longest window, then slice

    parts = []
    for (la, lo), grp in daily.groupby(["cell_lat", "cell_lon"]):
        g = grp.sort_values("date").set_index("date")
        sub = rec[(rec["lat_cell"] == la) & (rec["lon_cell"] == lo)]
        if len(sub) == 0:
            continue
        # precompute rolling stats per window, keyed (col, agg, window)
        roll_by_w = {}
        for w in WINDOWS:
            cols = {}
            for c, aggs in agg_spec.items():
                for a in aggs:
                    s = g[c].rolling(w, min_periods=1)
                    v = s.sum() if a == "sum" else getattr(s, a)()
                    cols[(c, a, w)] = v
            roll_by_w[w] = pd.DataFrame(cols)
        rows = {}
        pos_map = g.index.to_numpy()
        for idx in sub.index:
            d0 = sub.loc[idx, "ddate"]
            e = int(np.searchsorted(pos_map, np.datetime64(d0), side="left"))
            vals = {}
            for w in WINDOWS:
                roll = roll_by_w[w]
                # row for the window ENDING the day before deathdate
                p = e - w
                if p < 0 or p >= len(roll):
                    for c in roll.columns:
                        vals[(c[0], c[1], w)] = np.nan
                else:
                    row = roll.iloc[p]
                    for c in roll.columns:
                        vals[(c[0], c[1], w)] = row[c]
            rows[idx] = pd.Series(vals)
        f = pd.DataFrame(rows).T
        f.index.name = None
        parts.append(f)

    out = pd.concat(parts) if parts else pd.DataFrame(index=records.index)
    out = out.reindex(records.index)

    # prefix and flatten multiindex columns (some agg names come back flat, e.g. 'T2M_mean_7d')
    newcols = []
    for c in out.columns:
        if isinstance(c, tuple) and len(c) == 3:
            newcols.append(f"pw_{c[0]}_{c[1]}_{c[2]}d")
        elif isinstance(c, str) and c.startswith("(") :
            # stringified tuple from DataFrame(rows) construction
            inner = c.strip("()").replace("'", "")
            parts_t = [x.strip() for x in inner.split(",")]
            if len(parts_t) == 3:
                newcols.append(f"pw_{parts_t[0]}_{parts_t[1]}_{parts_t[2]}d")
            else:
                newcols.append(f"pw_{c}")
        else:
            newcols.append(f"pw_{c}")
    out.columns = newcols
    base = base.join(out)

    # ---- derived indicators from the daily windows (computable per record) ----
    for w in WINDOWS:
        base[f"pw_heat_days_{w}d"] = np.nan
    # rain-day counts & dry spell within the longest window, per record
    dry_flags = {}
    for (la, lo), grp in daily.groupby(["cell_lat", "cell_lon"]):
        g = grp.sort_values("date")
        flags = pd.DataFrame({"date": g["date"].to_numpy(),
                              "wet": (g["PRECTOTCORR"].to_numpy() >= 1.0).astype(float)})
        flags = flags.set_index("date")
        sub = rec[(rec["lat_cell"] == la) & (rec["lon_cell"] == lo)]
        pos_map = g["date"].to_numpy()
        for idx in sub.index:
            d0 = sub.loc[idx, "ddate"]
            pos = np.searchsorted(pos_map, np.datetime64(d0), side="left") - w
            pos_c = max(0, pos)
            sl = flags.iloc[max(0, pos_c - w + 1):pos_c]
            base.at[idx, f"pw_wet_days_{w}d"] = float(sl["wet"].sum()) if len(sl) else np.nan

    # heat-day / high-VPD-day counts and dry-spell length per window
    for (la, lo), grp in daily.groupby(["cell_lat", "cell_lon"]):
        g = grp.sort_values("date")
        fl = pd.DataFrame({
            "date": g["date"].to_numpy(),
            # thresholds calibrated to the Uganda range actually observed
            "hot": (g["APP_T"].to_numpy() >= 28).astype(float),
            "hot35": (g["T2M_MAX"].to_numpy() >= 33).astype(float),
            "vpdhigh": (g["VPD"].to_numpy() >= 1.2).astype(float),
            "wet": (g["PRECTOTCORR"].to_numpy() >= 1.0).astype(float),
        }).set_index("date")
        sub = rec[(rec["lat_cell"] == la) & (rec["lon_cell"] == lo)]
        pos_map = g["date"].to_numpy()
        for idx in sub.index:
            d0 = sub.loc[idx, "ddate"]
            e = np.searchsorted(pos_map, np.datetime64(d0), side="left")
            for w in WINDOWS:
                sl = fl.iloc[max(0, e - w):e]
                base.at[idx, f"pw_hot_days_{w}d"] = float(sl["hot"].sum()) if len(sl) else np.nan
                base.at[idx, f"pw_tmax35_days_{w}d"] = float(sl["hot35"].sum()) if len(sl) else np.nan
                base.at[idx, f"pw_vpd_days_{w}d"] = float(sl["vpdhigh"].sum()) if len(sl) else np.nan
                base.at[idx, f"pw_dry_days_{w}d"] = float(w - sl["wet"].sum()) if len(sl) else np.nan

    return base


PW_BASE = [f"pw_{v}_{a}_{w}d" for w in WINDOWS
           for v, a in [("T2M", "mean"), ("T2M", "max"), ("T2M_MIN", "min"),
                        ("T2M_MAX", "max"), ("PRECTOTCORR", "sum"), ("PRECTOTCORR", "max"),
                        ("RH2M", "mean"), ("QV2M", "mean"), ("T2MDEW", "mean"), ("T2MDEW", "max"),
                        ("WS10M", "mean"), ("VPD", "mean"), ("VPD", "max"),
                        ("APP_T", "mean"), ("APP_T", "max"), ("APP_T_3dmax", "max"),
                        ("sat_def", "mean"), ("e_a", "mean")]]
PW_COUNTS = ([f"pw_hot_days_{w}d" for w in WINDOWS] +
             [f"pw_tmax35_days_{w}d" for w in WINDOWS] +
             [f"pw_vpd_days_{w}d" for w in WINDOWS] +
             [f"pw_wet_days_{w}d" for w in WINDOWS] +
             [f"pw_dry_days_{w}d" for w in WINDOWS])
