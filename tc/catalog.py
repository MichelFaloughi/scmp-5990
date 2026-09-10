"""Build the storm catalogue and best-track table from IBTrACS.

Outputs
  data/tc/besttrack.csv : 6-hourly best-track rows (NA + EP, 2025+), USA agency values
  data/tc/storms.csv    : one row per storm with start/end/peak
  data/tc/inits.csv     : unique AIFS init times (00/12 UTC) covering each storm's lifetime

IBTrACS gotcha: basin code "NA" (North Atlantic) is read as NaN by pandas unless keep_default_na=False.
"""
import pandas as pd

AIFS_START = pd.Timestamp("2025-02-25")  # first AIFS-Single open-data run on the AWS mirror

raw = pd.read_csv("data/ibtracs/ibtracs.last3years.csv", skiprows=[1], low_memory=False,
                  keep_default_na=False, na_values=[" ", ""])
raw = raw[raw.BASIN.isin(["NA", "EP"])].copy()
raw["ISO_TIME"] = pd.to_datetime(raw.ISO_TIME)
for c in ["USA_LAT", "USA_LON", "USA_WIND", "USA_PRES", "USA_RMW", "USA_ROCI", "USA_POCI", "DIST2LAND"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
raw = raw[raw.ISO_TIME >= AIFS_START]

bt = raw[["SID", "NAME", "BASIN", "SEASON", "USA_ATCF_ID", "ISO_TIME", "NATURE", "USA_STATUS",
          "USA_LAT", "USA_LON", "USA_WIND", "USA_PRES", "USA_RMW", "USA_ROCI", "USA_POCI", "DIST2LAND"]].copy()
bt = bt[bt.ISO_TIME.dt.hour.isin([0, 6, 12, 18]) & (bt.ISO_TIME.dt.minute == 0)]  # synoptic times only
bt = bt.dropna(subset=["USA_LAT", "USA_LON"])
bt.to_csv("data/tc/besttrack.csv", index=False)

st = bt.groupby(["SID", "NAME", "BASIN", "SEASON", "USA_ATCF_ID"]).agg(
    start=("ISO_TIME", "min"), end=("ISO_TIME", "max"), peak_kt=("USA_WIND", "max"),
    min_pres=("USA_PRES", "min"), n=("ISO_TIME", "size")).reset_index()
st = st[st.peak_kt >= 34].sort_values("start")
st["days"] = (st.end - st.start).dt.total_seconds() / 86400
st.to_csv("data/tc/storms.csv", index=False)

# AIFS inits: every 00/12 UTC from 12 h before genesis to the last best-track time
inits = set()
for _, s in st.iterrows():
    t = (s.start - pd.Timedelta(hours=12)).floor("12h")
    while t <= s.end:
        inits.add(t); t += pd.Timedelta(hours=12)
inits = pd.Series(sorted(inits), name="init")
inits.to_csv("data/tc/inits.csv", index=False)

print(st[["NAME", "BASIN", "SEASON", "start", "end", "peak_kt", "days"]].to_string(index=False))
print(f"\nstorms: {len(st)}  (NA {sum(st.BASIN=='NA')}, EP {sum(st.BASIN=='EP')})   storm-days: {st.days.sum():.0f}")
print(f"best-track rows: {len(bt)}   unique inits: {len(inits)}")
