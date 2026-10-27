"""Fetch the AIFS-Single fields needed for SHIPS-style predictors, via byte ranges.

Same approach as fetch_aifs.py (index file + HTTP Range per GRIB message, crop to the NA+EP box),
extended to pressure levels. Uses the Google mirror of ECMWF open data by default, which does not
rate-limit the way the S3 bucket does; set AIFS_BASE to override.

Fields per step (10): u, v at 850 and 200 hPa (shear, vorticity, divergence), t at 200, 500, 700 hPa,
q at 500, 700 hPa (relative humidity), skt (sea surface temperature stand-in over ocean).

Output: data/aifs_pl/{YYYYMMDDHH}.npz with arrays named like u850, v200, t500, q700, skt,
        each [step, lat, lon] as float32, plus lats, lons, steps. About 150 MB per init compressed.

Usage:  python pipeline/fetch_aifs_pl.py 2025090100 [2025090112 ...]
        python pipeline/fetch_aifs_pl.py --inits data/tc/inits.csv [--workers 4]
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, requests, pandas as pd
import earthkit.data as ekd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = os.environ.get("AIFS_BASE", "https://storage.googleapis.com/ecmwf-open-data")
STEPS = list(range(0, 169, 6))
PL = {"u": [850, 200], "v": [850, 200], "t": [200, 500, 700], "q": [500, 700]}
SFC = ["skt"]
LAT = (0.0, 60.0); LON = (-180.0, 0.0)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "aifs_pl")
PAUSE = float(os.environ.get("AIFS_FETCH_PAUSE", "0.05"))

def session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=8, backoff_factor=1.0,
                                                     status_forcelist=[429, 500, 502, 503, 504],
                                                     allowed_methods=["GET"])))
    return s

def url(init, step, ext):
    d, h = init.strftime("%Y%m%d"), init.strftime("%H")
    return f"{BASE}/{d}/{h}z/aifs-single/0p25/oper/{d}{h}0000-{step}h-oper-fc.{ext}"

def key(r):
    return f"{r['param']}{r['levelist']}" if r.get("levtype") == "pl" else r["param"]

def wanted():
    return {f"{p}{l}" for p, ls in PL.items() for l in ls} | set(SFC)

def fetch_step(S, init, step):
    idx = S.get(url(init, step, "index"), timeout=60); idx.raise_for_status()
    rows = [json.loads(l) for l in idx.text.splitlines() if l.strip()]
    want = wanted()
    sel = {key(r): r for r in rows if key(r) in want and
           (r.get("levtype") == "pl" or (r.get("levtype") == "sfc" and r["param"] in SFC))}
    if set(sel) != want:
        raise RuntimeError(f"missing {want - set(sel)} at {init} +{step}h")
    out, ix = {}, None
    for k, r in sel.items():
        rng = f"bytes={r['_offset']}-{r['_offset'] + r['_length'] - 1}"
        g = S.get(url(init, step, "grib2"), headers={"Range": rng}, timeout=120); g.raise_for_status()
        time.sleep(PAUSE)
        f = ekd.from_source("memory", g.content)[0]
        arr = f.to_numpy()
        if ix is None:
            ll = f.to_latlon(flatten=False); lats, lons = ll["lat"][:, 0], ll["lon"][0, :]
            ila = np.where((lats >= LAT[0]) & (lats <= LAT[1]))[0]
            ilo = np.where((lons >= LON[0]) & (lons <= LON[1]))[0]
            out["lats"], out["lons"], ix = lats[ila], lons[ilo], (ila, ilo)
        out[k] = arr[np.ix_(*ix)].astype(np.float32)
    return out

def fetch_init(init, force=False):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{init.strftime('%Y%m%d%H')}.npz"
    if os.path.exists(path) and not force:
        return path, "cached"
    S = session(); t0 = time.time(); acc = {k: [] for k in wanted()}; meta = None
    for s in STEPS:
        d = fetch_step(S, init, s)
        for k in acc: acc[k].append(d[k])
        meta = meta or {"lats": d["lats"], "lons": d["lons"]}
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **{k: np.stack(v) for k, v in acc.items()},
                        lats=meta["lats"], lons=meta["lons"], steps=np.array(STEPS))
    os.replace(tmp, path)
    return path, f"{time.time() - t0:.0f}s"

if __name__ == "__main__":
    a = sys.argv[1:]; workers = 1
    if "--workers" in a:
        i = a.index("--workers"); workers = int(a[i + 1]); a = a[:i] + a[i + 2:]
    inits = list(pd.to_datetime(pd.read_csv(a[1])["init"])) if a and a[0] == "--inits" \
        else [pd.Timestamp(x[:8]) + pd.to_timedelta(int(x[8:10]), unit="h") for x in a]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_init, init): init for init in inits}
        for fut in as_completed(futs):
            init = futs[fut]; done += 1
            try:
                path, how = fut.result()
                print(f"[{done}/{len(inits)}] {init:%Y-%m-%d %HZ} -> {os.path.basename(path)} ({how})", flush=True)
            except Exception as e:
                print(f"[{done}/{len(inits)}] {init:%Y-%m-%d %HZ} FAILED: {e}", flush=True)
