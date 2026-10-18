"""Download daily climate from NASA POWER (MERRA-2 reanalysis, public, no key).
Caches to disk so re-runs are instant. 46 unique 2-dp coordinate cells x 1 request each.

Source: https://power.larc.nasa.gov/api/temporal/daily/point
Docs:   https://power.larc.nasa.gov/docs/services/api
MERRA-2 = Modern-Era Retrospective Analysis for Research and Applications, version 2.
NOTE: T2M/T2M_MAX/T2M_MIN in degC, PRECTOTCORR in mm/day, QV2M in g/kg (specific
humidity), RH2M in %, T2MDEW in degC, WS10M in m/s. Fill value is -999.0.
"""
import os, json, time, urllib.request, socket
import numpy as np
import pandas as pd

CACHE_DIR = "data_cache/power"
os.makedirs(CACHE_DIR, exist_ok=True)
PARAMS = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,QV2M,T2MDEW,WS10M"
START, END = "20070101", "20221231"
socket.setdefaulttimeout(120)


def cell_path(lat, lon):
    return os.path.join(CACHE_DIR, f"cell_{lat:.2f}_{lon:.2f}.json")


def fetch_cell(lat, lon, retries=4):
    """Return {param: {YYYYMMDD: value}} for one coordinate cell. Cached."""
    p = cell_path(lat, lon)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass  # corrupt cache -> refetch
    url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters={PARAMS}"
           f"&community=AG&longitude={lon:.4f}&latitude={lat:.4f}"
           f"&start={START}&end={END}&format=JSON")
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url) as r:
                d = json.load(r)
            out = {}
            for k, v in d["properties"]["parameter"].items():
                out[k] = {dte: (None if val is None or val <= -900 else float(val))
                          for dte, val in v.items()}
            with open(p, "w") as f:
                json.dump(out, f)
            return out
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed {lat},{lon}: {last}")


def build_daily_frame(workers=8):
    """Concatenate all cached cells into one long daily DataFrame. Parallel fetch."""
    from concurrent.futures import ThreadPoolExecutor
    tr = pd.read_csv("Train.csv"); te = pd.read_csv("Test.csv")
    allrec = pd.concat([tr, te], ignore_index=True)
    cells = sorted(allrec.groupby([allrec.latitude.round(2),
                                  allrec.longitude.round(2)]).groups.items())
    todo = [(float(la), float(lo)) for (la, lo), _ in cells]

    def job(ll):
        lat, lon = ll
        try:
            fetch_cell(lat, lon)
            return f"ok   ({lat:.2f},{lon:.2f})"
        except Exception as e:
            return f"FAIL ({lat:.2f},{lon:.2f}): {e}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, msg in enumerate(ex.map(job, todo), 1):
            print(f"  [{i}/{len(todo)}] {msg}", flush=True)

    rows = []
    for (la, lo), idx in cells:
        data = fetch_cell(float(la), float(lo))
        # data: {param: {YYYYMMDD: value}} -> build date-indexed frame explicitly.
        # Do NOT rely on .T: the outer keys become the index, so build from items.
        per_date = {}
        for param, dmap in data.items():
            for dte, val in dmap.items():
                per_date.setdefault(dte, {})[param] = val
        df = pd.DataFrame.from_dict(per_date, orient="index")
        df.index.name = "date"
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m%d")
        for c in ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "QV2M", "T2MDEW", "WS10M"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["cell_lat"] = float(la); df["cell_lon"] = float(lo)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["date", "cell_lat", "cell_lon"]).reset_index(drop=True)
    out.to_csv("data_cache/power_daily.csv", index=False)
    print(f"saved data_cache/power_daily.csv shape={out.shape}", flush=True)
    return out


if __name__ == "__main__":
    df = build_daily_frame()
    print(df.head())
    print(df.describe().to_string())
