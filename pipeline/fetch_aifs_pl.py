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
        python pipeline/fetch_aifs_pl.py --inits data/tc_global/inits.csv --boxes wp,sh [--workers 4]
Boxes as in fetch_aifs.py (na_ep writes to data/aifs_pl, wp to data/aifs_pl_wp, sh to data/aifs_pl_sh).
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
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from fetch_aifs import BOXES, crop_index
OUTDIR = {"na_ep": "aifs_pl", "wp": "aifs_pl_wp", "sh": "aifs_pl_sh"}
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

def fetch_step(S, init, step, boxes=("na_ep",)):
    idx = S.get(url(init, step, "index"), timeout=60); idx.raise_for_status()
    rows = [json.loads(l) for l in idx.text.splitlines() if l.strip()]
    want = wanted()
    sel = {key(r): r for r in rows if key(r) in want and
           (r.get("levtype") == "pl" or (r.get("levtype") == "sfc" and r["param"] in SFC))}
    if set(sel) != want:
        raise RuntimeError(f"missing {want - set(sel)} at {init} +{step}h")
    out = {b: {} for b in boxes}; ix = None
    for k, r in sel.items():
        rng = f"bytes={r['_offset']}-{r['_offset'] + r['_length'] - 1}"
        g = S.get(url(init, step, "grib2"), headers={"Range": rng}, timeout=120); g.raise_for_status()
        time.sleep(PAUSE)
        f = ekd.from_source("memory", g.content)[0]
        arr = f.to_numpy()
        if ix is None:
            ll = f.to_latlon(flatten=False); lats, lons = ll["lat"][:, 0], ll["lon"][0, :]
            ix = {}
            for b in boxes:
                ila, ilo, la, lo = crop_index(lats, lons, b)
                ix[b] = (ila, ilo); out[b]["lats"], out[b]["lons"] = la, lo
        for b in boxes:
            out[b][k] = arr[np.ix_(*ix[b])].astype(np.float32)
    return out

def out_path(init, box):
    return os.path.join(ROOT, "data", OUTDIR[box], f"{init.strftime('%Y%m%d%H')}.npz")

def fetch_init(init, boxes=("na_ep",), force=False):
    todo = [b for b in boxes if force or not os.path.exists(out_path(init, b))]
    if not todo:
        return [out_path(init, b) for b in boxes], "cached"
    S = session(); t0 = time.time(); acc = {b: {k: [] for k in wanted()} for b in todo}; meta = {}
    for s in STEPS:
        d = fetch_step(S, init, s, todo)
        for b in todo:
            for k in acc[b]: acc[b][k].append(d[b][k])
            meta.setdefault(b, {"lats": d[b]["lats"], "lons": d[b]["lons"]})
    paths = []
    for b in todo:
        path = out_path(init, b); os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **{k: np.stack(v) for k, v in acc[b].items()},
                            lats=meta[b]["lats"], lons=meta[b]["lons"], steps=np.array(STEPS))
        os.replace(tmp, path); paths.append(path)
    return paths, f"{time.time() - t0:.0f}s"

if __name__ == "__main__":
    a = sys.argv[1:]; workers = 1; boxes = None
    if "--workers" in a:
        i = a.index("--workers"); workers = int(a[i + 1]); a = a[:i] + a[i + 2:]
    if "--boxes" in a:
        i = a.index("--boxes"); boxes = a[i + 1].split(","); a = a[:i] + a[i + 2:]
    if a and a[0] == "--inits":
        tab = pd.read_csv(a[1]); tab["init"] = pd.to_datetime(tab["init"])
        if "boxes" in tab.columns:
            jobs = [(t, [b for b in bx.split() if boxes is None or b in boxes]) for t, bx in zip(tab.init, tab.boxes)]
            jobs = [(t, bx) for t, bx in jobs if bx]
        else:
            jobs = [(t, boxes or ["na_ep"]) for t in tab.init]
    else:
        jobs = [(pd.Timestamp(x[:8]) + pd.to_timedelta(int(x[8:10]), unit="h"), boxes or ["na_ep"]) for x in a]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_init, t, bx): (t, bx) for t, bx in jobs}
        for fut in as_completed(futs):
            t, bx = futs[fut]; done += 1
            try:
                paths, how = fut.result()
                print(f"[{done}/{len(jobs)}] {t:%Y-%m-%d %HZ} {'+'.join(bx)} -> {', '.join(os.path.basename(p) for p in paths)} ({how})", flush=True)
            except Exception as e:
                print(f"[{done}/{len(jobs)}] {t:%Y-%m-%d %HZ} {'+'.join(bx)} FAILED: {e}", flush=True)
