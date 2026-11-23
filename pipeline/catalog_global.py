"""Storm catalogue and best track for the other basins (West Pacific, North Indian, Southern
Hemisphere), the advisor's item 3 of 2026-11-11. Same logic as catalog.py, written to
data/tc_global/ so the North Atlantic and East Pacific files in data/tc/ are untouched.

Outputs
  data/tc_global/besttrack.csv : 6-hourly best-track rows, all basins, 2025-02-25 onward
  data/tc_global/storms.csv    : one row per storm (peak >= 34 kt) with its fetch box
  data/tc_global/inits.csv     : init times to fetch, with the boxes needed at each init

Boxes (see fetch_aifs.py): na_ep is the existing Western Hemisphere crop; wp covers the West
Pacific and North Indian basins; sh covers the South Indian and South Pacific basins.
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "tc_global")
AIFS_START = pd.Timestamp("2025-02-25")
BOX = {"NA": "na_ep", "EP": "na_ep", "WP": "wp", "NI": "wp", "SI": "sh", "SP": "sh", "SA": "sh"}

raw = pd.read_csv(f"{ROOT}/data/ibtracs/ibtracs.last3years.csv", skiprows=[1], low_memory=False,
                  keep_default_na=False, na_values=[" ", ""])
raw["ISO_TIME"] = pd.to_datetime(raw.ISO_TIME)
for c in ["USA_LAT", "USA_LON", "USA_WIND", "USA_PRES", "USA_RMW", "USA_ROCI", "USA_POCI", "DIST2LAND"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw = raw[raw.ISO_TIME >= AIFS_START]

bt = raw[["SID", "NAME", "BASIN", "SEASON", "USA_ATCF_ID", "ISO_TIME", "NATURE", "USA_STATUS",
          "USA_LAT", "USA_LON", "USA_WIND", "USA_PRES", "USA_RMW", "USA_ROCI", "USA_POCI", "DIST2LAND"]].copy()
bt = bt[bt.ISO_TIME.dt.hour.isin([0, 6, 12, 18]) & (bt.ISO_TIME.dt.minute == 0)]
bt = bt.dropna(subset=["USA_LAT", "USA_LON"])
# a storm that crosses basins (EP to WP, or SI to SP) keeps the basin of its first fix
first_basin = bt.sort_values("ISO_TIME").groupby("SID").BASIN.first()
bt["BASIN"] = bt.SID.map(first_basin)
bt["box"] = bt.BASIN.map(BOX)
os.makedirs(OUT, exist_ok=True)
bt.to_csv(f"{OUT}/besttrack.csv", index=False)

st = bt.groupby(["SID", "NAME", "BASIN", "SEASON", "USA_ATCF_ID", "box"]).agg(
    start=("ISO_TIME", "min"), end=("ISO_TIME", "max"), peak_kt=("USA_WIND", "max"),
    min_pres=("USA_PRES", "min"), n=("ISO_TIME", "size")).reset_index()
st = st[st.peak_kt >= 34].sort_values("start")
st["days"] = (st.end - st.start).dt.total_seconds() / 86400
st.to_csv(f"{OUT}/storms.csv", index=False)

inits = {}
for _, s in st.iterrows():
    t = (s.start - pd.Timedelta(hours=12)).floor("12h")
    while t <= s.end:
        inits.setdefault(t, set()).add(s.box); t += pd.Timedelta(hours=12)
ini = pd.DataFrame({"init": sorted(inits), "boxes": [" ".join(sorted(inits[t])) for t in sorted(inits)]})
ini.to_csv(f"{OUT}/inits.csv", index=False)

print(st.groupby("BASIN").agg(storms=("SID", "size"), ge96=("peak_kt", lambda x: (x >= 96).sum()),
                              days=("days", "sum")).to_string())
for b in ["na_ep", "wp", "sh"]:
    print(f"inits needing box {b}: {ini.boxes.str.contains(b).sum()}")
print(f"storms: {len(st)}   best-track rows: {len(bt)}   inits: {len(ini)}")
