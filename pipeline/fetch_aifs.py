"""Fetch AIFS-Single surface fields from the ECMWF Open Data AWS mirror via byte ranges.

For each init (00/12 UTC) and step, reads the .index file, pulls only the requested GRIB messages
with HTTP Range requests, decodes them, crops to the NA+EP box and stores compact arrays.

Output: data/aifs/{YYYYMMDDHH}.npz with msl[step,lat,lon] (Pa, f32), u10, v10 (m/s, f16),
        lats, lons, steps.  ~1 MB per step in the box.

Boxes (added 2026-11-23 for the other basins): na_ep is the original crop and writes to data/aifs;
wp (West Pacific and North Indian) and sh (South Indian and South Pacific, longitudes stored
0..360 so the box is contiguous across the dateline) write to data/aifs_wp and data/aifs_sh. One
global message is downloaded per (init, step, field) and cropped to every box the init needs.

Usage:  python pipeline/fetch_aifs.py 2025090100 [2025090112 ...]            (na_ep box)
        python pipeline/fetch_aifs.py --inits data/tc/inits.csv
        python pipeline/fetch_aifs.py --inits data/tc_global/inits.csv --boxes wp,sh [--workers 4]
        (an inits file may carry a "boxes" column, e.g. "wp sh"; --boxes then filters it)
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, requests, pandas as pd
import earthkit.data as ekd

BASE = os.environ.get("AIFS_BASE", "https://ecmwf-forecasts.s3.amazonaws.com")
STEPS = list(range(0, 169, 6))           # 0..168 h, the AIFS-TC lead range
PARAMS = ["msl", "10u", "10v"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# lat range, lon range (in the file's -180..179.75 convention unless wrap is set, then 0..360), output dir
BOXES = {"na_ep": dict(lat=(0.0, 60.0), lon=(-180.0, 0.0), wrap=False, out="aifs"),
         "wp":    dict(lat=(0.0, 60.0), lon=(40.0, 180.0), wrap=False, out="aifs_wp"),
         "sh":    dict(lat=(-50.0, 0.0), lon=(30.0, 240.0), wrap=True, out="aifs_sh")}
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

def crop_index(lats, lons, box):
    """Row and column indices of a box, and the stored lat/lon vectors. Longitudes of a wrapped box are
    returned in 0..360 and sorted so the crop is contiguous across the dateline."""
    b = BOXES[box]
    ila = np.where((lats >= b["lat"][0]) & (lats <= b["lat"][1]))[0]
    if b["wrap"]:
        l360 = np.where(lons < 0, lons + 360.0, lons)
        ilo = np.where((l360 >= b["lon"][0]) & (l360 <= b["lon"][1]))[0]
        ilo = ilo[np.argsort(l360[ilo])]
        return ila, ilo, lats[ila], l360[ilo]
    ilo = np.where((lons >= b["lon"][0]) & (lons <= b["lon"][1]))[0]
    return ila, ilo, lats[ila], lons[ilo]

def fetch_step(init, step, boxes=("na_ep",), S=S):
    idx = S.get(url(init, step, "index"), timeout=60); idx.raise_for_status()
    rows = [json.loads(l) for l in idx.text.splitlines() if l.strip()]
    want = {r["param"]: r for r in rows if r.get("levtype") == "sfc" and r["param"] in PARAMS}
    if len(want) != len(PARAMS):
        raise RuntimeError(f"missing {set(PARAMS) - set(want)} at {init} +{step}h")
    out = {b: {} for b in boxes}; ix = None
    for p, r in want.items():
        rng = f"bytes={r['_offset']}-{r['_offset'] + r['_length'] - 1}"
        g = S.get(url(init, step, "grib2"), headers={"Range": rng}, timeout=120); g.raise_for_status()
        time.sleep(PAUSE)
        f = ekd.from_source("memory", g.content)[0]
        arr = f.to_numpy()                                    # (721, 1440): lat 90..-90, lon -180..179.75
        if ix is None:
            ll = f.to_latlon(flatten=False); lats, lons = ll["lat"][:, 0], ll["lon"][0, :]
            ix = {}
            for b in boxes:
                ila, ilo, la, lo = crop_index(lats, lons, b)
                ix[b] = (ila, ilo); out[b]["lats"], out[b]["lons"] = la, lo
        for b in boxes:
            ila, ilo = ix[b]
            out[b][p] = arr[np.ix_(ila, ilo)].astype(np.float32 if p == "msl" else np.float16)
    return out

def out_path(init, box):
    return os.path.join(ROOT, "data", BOXES[box]["out"], f"{init.strftime('%Y%m%d%H')}.npz")

def fetch_init(init, boxes=("na_ep",), force=False):
    todo = [b for b in boxes if force or not os.path.exists(out_path(init, b))]
    if not todo:
        return [out_path(init, b) for b in boxes], "cached"
    S = requests.Session()
    S.mount("https://", HTTPAdapter(max_retries=Retry(total=8, backoff_factor=1.0,
                                                     status_forcelist=[429, 500, 502, 503, 504],
                                                     allowed_methods=["GET"])))
    t0 = time.time(); acc = {b: {p: [] for p in PARAMS} for b in todo}; meta = {}
    for s in STEPS:
        d = fetch_step(init, s, todo, S)
        for b in todo:
            for p in PARAMS: acc[b][p].append(d[b][p])
            meta.setdefault(b, {"lats": d[b]["lats"], "lons": d[b]["lons"]})
    paths = []
    for b in todo:
        path = out_path(init, b); os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, msl=np.stack(acc[b]["msl"]), u10=np.stack(acc[b]["10u"]), v10=np.stack(acc[b]["10v"]),
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
        jobs = [(pd.Timestamp(x[:8]) + pd.Timedelta(hours=int(x[8:10])), boxes or ["na_ep"]) for x in a]
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
