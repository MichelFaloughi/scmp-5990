"""Parse NHC ATCF a-decks for the official forecast (OFCL) and CARQ (operational initial estimate).

Output: data/tc/ofcl.csv  columns: atcf_id, init, tech, lead, lat, lon, vmax_kt, pmin_hpa
ATCF format: comma-separated, lat like '173N' (tenths of degree), lon like '273W'.
"""
import glob, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = {"OFCL", "CARQ", "EAIO", "GDMN", "FNV3", "HAFS", "HFSA", "HWRF", "AVNO", "AEMN"}

def parse_latlon(s):
    s = s.strip()
    if not s or s in ("0", "-"):
        return np.nan
    v = float(s[:-1]) / 10.0
    return -v if s[-1] in "SW" else v

rows = []
for f in sorted(glob.glob(f"{ROOT}/data/atcf/a[ae][lp]??20??.dat")):
    for line in open(f, errors="ignore"):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 10 or p[4] not in KEEP:
            continue
        try:
            rows.append(dict(atcf_id=f"{p[0]}{int(p[1]):02d}{p[2][:4]}", init=pd.to_datetime(p[2][:10], format="%Y%m%d%H"), tech=p[4],
                             lead=int(p[5]), lat=parse_latlon(p[6]), lon=parse_latlon(p[7]),
                             vmax_kt=float(p[8]) if p[8] else np.nan,
                             pmin_hpa=float(p[9]) if p[9] and float(p[9]) > 0 else np.nan))
        except (ValueError, IndexError):
            continue
df = pd.DataFrame(rows)
# a-decks repeat rows per wind radius; keep one per (id, init, tech, lead)
df = df.sort_values(["atcf_id", "init", "tech", "lead"]).drop_duplicates(["atcf_id", "init", "tech", "lead"])
df = df[df.vmax_kt > 0]
df.to_csv(f"{ROOT}/data/tc/ofcl.csv", index=False)
o = df[df.tech == "OFCL"]
print(f"OFCL rows: {len(o)}  storms: {o.atcf_id.nunique()}  inits: {o.groupby('atcf_id').init.nunique().sum()}  leads: {sorted(o.lead.unique())}")
print(o[(o.atcf_id == 'EP132025') & (o.init == '2025-09-01')].to_string(index=False))
