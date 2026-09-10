#!/bin/zsh
# Progress bar for the AIFS archive download. Usage: tc/progress.sh   (or: watch -n 30 tc/progress.sh)
cd "$(dirname "$0")/.." || exit 1
total=$(( $(wc -l < data/tc/inits.csv) - 1 ))
done_=$(ls data/aifs/*.npz 2>/dev/null | wc -l | tr -d ' ')
fail=$(cat data/aifs/fetch_shard*.log 2>/dev/null | grep -c FAILED)
pct=$(( 100 * done_ / total ))
width=40; filled=$(( width * done_ / total ))
bar=$(printf '%*s' $filled '' | tr ' ' '#')$(printf '%*s' $((width - filled)) '' | tr ' ' '.')
running=$(pgrep -f "fetch_aifs.py --inits" | wc -l | tr -d ' ')
printf '[%s] %3d%%  %d/%d runs  %s  failed:%s  workers:%s\n' "$bar" "$pct" "$done_" "$total" "$(du -sh data/aifs | cut -f1)" "$fail" "$running"
