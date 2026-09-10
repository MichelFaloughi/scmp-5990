"""Fetch AIFS-Single surface fields from the ECMWF Open Data AWS mirror via byte ranges.

For each init (00/12 UTC) and step, reads the .index file, pulls only the requested GRIB messages
with HTTP Range requests, decodes them, crops to the NA+EP box and stores compact arrays.

Output: data/aifs/{YYYYMMDDHH}.npz with msl[step,lat,lon] (Pa, f32), u10, v10 (m/s, f16),
        lats, lons, steps.  ~1 MB per step in the box.

Usage:  python tc/fetch_aifs.py 2025090100 [2025090112 ...]
        python tc/fetch_aifs.py --inits data/tc/inits.csv
"""
import json, os, sys, time
import numpy as np, requests, pandas as pd
import earthkit.data as ekd

BASE = "https://ecmwf-forecasts.s3.amazonaws.com"
STEPS = list(range(0, 169, 6))           # 0..168 h, the AIFS-TC lead range
PARAMS = ["msl", "10u", "10v"]
LAT = (0.0, 60.0); LON = (-180.0, 0.0)   # NA + EP box; open-data files store lon -180..179.75
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "aifs")
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
S = requests.Session()
S.mount("https://", HTTPAdapter(max_retries=Retry(total=8, backoff_factor=1.0,
                                                 status_forcelist=[429, 500, 502, 503, 504],
                                                 allowed_methods=["GET"])))
PAUSE = float(os.environ.get("AIFS_FETCH_PAUSE", "0.2"))  # seconds between requests, be polite to S3

def url(init, step, ext):
    d, h = init.strftime("%Y%m%d"), init.strftime("%H")
    return f"{BASE}/{d}/{h}z/aifs-single/0p25/oper/{d}{h}0000-{step}h-oper-fc.{ext}"

def fetch_step(init, step):
    idx = S.get(url(init, step, "index"), timeout=60); idx.raise_for_status()
    rows = [json.loads(l) for l in idx.text.splitlines() if l.strip()]
    want = {r["param"]: r for r in rows if r.get("levtype") == "sfc" and r["param"] in PARAMS}
    if len(want) != len(PARAMS):
        raise RuntimeError(f"missing {set(PARAMS) - set(want)} at {init} +{step}h")
    out = {}
    for p, r in want.items():
        rng = f"bytes={r['_offset']}-{r['_offset'] + r['_length'] - 1}"
        g = S.get(url(init, step, "grib2"), headers={"Range": rng}, timeout=120); g.raise_for_status()
        time.sleep(PAUSE)
        f = ekd.from_source("memory", g.content)[0]
        arr = f.to_numpy()                                    # (721, 1440): lat 90..-90, lon 0..359.75
        if "lats" not in out:
            ll = f.to_latlon(flatten=False); lats, lons = ll["lat"][:, 0], ll["lon"][0, :]
            ila = np.where((lats >= LAT[0]) & (lats <= LAT[1]))[0]
            ilo = np.where((lons >= LON[0]) & (lons <= LON[1]))[0]
            out["lats"], out["lons"], out["_ix"] = lats[ila], lons[ilo], (ila, ilo)
        ila, ilo = out["_ix"]
        out[p] = arr[np.ix_(ila, ilo)].astype(np.float32 if p == "msl" else np.float16)
    return out

def fetch_init(init, force=False):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{init.strftime('%Y%m%d%H')}.npz"
    if os.path.exists(path) and not force:
        return path, "cached"
    t0 = time.time(); acc = {p: [] for p in PARAMS}; meta = None
    for s in STEPS:
        d = fetch_step(init, s)
        for p in PARAMS: acc[p].append(d[p])
        meta = meta or {"lats": d["lats"], "lons": d["lons"]}
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, msl=np.stack(acc["msl"]), u10=np.stack(acc["10u"]), v10=np.stack(acc["10v"]),
                        lats=meta["lats"], lons=meta["lons"], steps=np.array(STEPS))
    os.replace(tmp, path)
    return path, f"{time.time() - t0:.0f}s"

if __name__ == "__main__":
    a = sys.argv[1:]
    inits = pd.to_datetime(pd.read_csv(a[1])["init"]) if a and a[0] == "--inits" \
        else [pd.Timestamp(x[:8]) + pd.Timedelta(hours=int(x[8:10])) for x in a]
    for i, init in enumerate(inits):
        try:
            path, how = fetch_init(init)
            print(f"[{i + 1}/{len(inits)}] {init:%Y-%m-%d %HZ} -> {os.path.basename(path)} ({how})", flush=True)
        except Exception as e:
            print(f"[{i + 1}/{len(inits)}] {init:%Y-%m-%d %HZ} FAILED: {e}", flush=True)
