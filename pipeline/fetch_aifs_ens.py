"""Fetch AIFS-ENS member intensities along the AIFS-Single track, via byte ranges.

For each init in data/tc/inits_ens.csv and each step in STEPS, reads the control (cf) and
perturbed (pf, 50 members) index files, pulls msl, 10u, 10v for every member with HTTP Range
requests, and for every storm case of that init and step (position from the AIFS-Single track in
data/tc/aifs_tracks_eta.csv) records the member's maximum 10 m wind and minimum MSLP within
250 km of the deterministic position. Nothing gridded is stored.

Output: data/ens/{YYYYMMDDHH}.csv with init, sid, step, member (0 = control), vmax_kt, pmin_hpa.
        --merge concatenates them into data/tc/ens_members.csv.

Usage:  python pipeline/fetch_aifs_ens.py --inits data/tc/inits_ens.csv [--workers 4]
        python pipeline/fetch_aifs_ens.py --merge
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, requests, pandas as pd
import earthkit.data as ekd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = os.environ.get("AIFS_BASE", "https://storage.googleapis.com/ecmwf-open-data")
STEPS = [6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
PARAMS = ["msl", "10u", "10v"]
RADIUS_KM = 250.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "ens")
TRACKS = os.path.join(ROOT, "data", "tc", "aifs_tracks_eta.csv")
PAUSE = float(os.environ.get("AIFS_FETCH_PAUSE", "0.05"))
MS2KT = 1.943844

def session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=8, backoff_factor=1.0,
                                                     status_forcelist=[429, 500, 502, 503, 504],
                                                     allowed_methods=["GET"])))
    return s

def url(init, step, kind, ext):
    d, h = init.strftime("%Y%m%d"), init.strftime("%H")
    return f"{BASE}/{d}/{h}z/aifs-ens/0p25/enfo/{d}{h}0000-{step}h-enfo-{kind}.{ext}"

def haversine_km(lat0, lon0, lat, lon):
    p0, p = np.radians(lat0), np.radians(lat); dl = np.radians(lon - lon0)
    a = np.sin((p - p0) / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

def decode(content):
    f = ekd.from_source("memory", content)[0]
    ll = f.to_latlon(flatten=False)
    return f.to_numpy(), ll["lat"][:, 0], ll["lon"][0, :]

def member_fields(S, init, step, kind, rows, number):
    out = {}
    for p in PARAMS:
        r = [x for x in rows if x["param"] == p and (kind == "cf" or x.get("number") == number)]
        if len(r) != 1:
            raise RuntimeError(f"{init} +{step}h {kind} {number} {p}: {len(r)} rows")
        r = r[0]
        g = S.get(url(init, step, kind, "grib2"),
                  headers={"Range": f"bytes={r['_offset']}-{r['_offset'] + r['_length'] - 1}"}, timeout=120)
        g.raise_for_status(); time.sleep(PAUSE)
        out[p] = decode(g.content)
    return out

def case_stats(fields, lat0, lon0):
    msl, lats, lons = fields["msl"]
    lons = np.where(lons > 180, lons - 360, lons)
    ia = np.where(np.abs(lats - lat0) <= 3.0)[0]; io = np.where(np.abs(lons - lon0) <= 3.5)[0]
    LA, LO = np.meshgrid(lats[ia], lons[io], indexing="ij")
    m = haversine_km(lat0, lon0, LA, LO) <= RADIUS_KM
    sub = lambda a: a[np.ix_(ia, io)][m]
    spd = np.hypot(sub(fields["10u"][0]), sub(fields["10v"][0]))
    return float(spd.max() * MS2KT), float(sub(msl).min() / 100.0)

def fetch_init(init, cases, force=False):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{init.strftime('%Y%m%d%H')}.csv"
    if os.path.exists(path) and not force:
        return path, "cached"
    S = session(); t0 = time.time(); rows_out = []
    for step in STEPS:
        c = cases[cases.step == step]
        if c.empty:
            continue
        idx = {}
        for kind in ("cf", "pf"):
            r = S.get(url(init, step, kind, "index"), timeout=60); r.raise_for_status()
            idx[kind] = [json.loads(l) for l in r.text.splitlines() if l.strip()]
        members = [("cf", None)] + [("pf", str(n)) for n in range(1, 51)]
        for k, (kind, num) in enumerate(members):
            f = member_fields(S, init, step, kind, idx[kind], num)
            for _, cs in c.iterrows():
                v, p = case_stats(f, float(cs.lat), float(cs.lon))
                rows_out.append(dict(init=str(init), sid=cs.sid, step=step, member=k, vmax_kt=v, pmin_hpa=p))
    tmp = path + ".tmp"
    pd.DataFrame(rows_out).to_csv(tmp, index=False); os.replace(tmp, path)
    return path, f"{time.time() - t0:.0f}s, {len(rows_out)} rows"

def merge():
    files = sorted(f for f in os.listdir(OUT) if f.endswith(".csv"))
    df = pd.concat([pd.read_csv(f"{OUT}/{f}") for f in files], ignore_index=True)
    df.to_csv(os.path.join(ROOT, "data", "tc", "ens_members.csv"), index=False)
    print(f"merged {len(files)} inits, {len(df)} rows -> data/tc/ens_members.csv")

if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--merge":
        merge(); sys.exit()
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 1
    inits = pd.to_datetime(pd.read_csv(a[a.index("--inits") + 1])["init"])
    tracks = pd.read_csv(TRACKS, keep_default_na=False, na_values=[""])
    tracks["init"] = pd.to_datetime(tracks.init)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_init, init, tracks[tracks.init == init]): init for init in inits}
        for fut in as_completed(futs):
            init = futs[fut]; done += 1
            try:
                path, how = fut.result()
                print(f"[{done}/{len(inits)}] {init:%Y-%m-%d %HZ} -> {os.path.basename(path)} ({how})", flush=True)
            except Exception as e:
                print(f"[{done}/{len(inits)}] {init:%Y-%m-%d %HZ} FAILED: {e}", flush=True)
