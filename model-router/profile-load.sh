#!/bin/bash
# Sample disk read MB/s, top CPU threads, and memory every 2 s while a model loads.
# usage: profile-load.sh <outfile> <seconds>
OUT=$1; DUR=${2:-900}
dev=nvme0n1
rd() { awk -v d=$dev '$3==d {print $6}' /proc/diskstats; }   # sectors read
prev=$(rd); t0=$(date +%s)
echo "t  disk_MBps  MemAvail_GB  Cached_GB  Swap_used_GB  top_threads(%cpu:comm)" > "$OUT"
while [ $(( $(date +%s) - t0 )) -lt $DUR ]; do
  sleep 2
  cur=$(rd); mbps=$(( (cur - prev) * 512 / 2 / 1000000 )); prev=$cur
  mem=$(awk '/MemAvailable/{a=$2}/^Cached/{c=$2}/SwapTotal/{st=$2}/SwapFree/{sf=$2}END{printf "%.1f %.1f %.1f", a/1e6, c/1e6, (st-sf)/1e6}' /proc/meminfo)
  thr=$(top -H -b -n1 -w 200 | awk 'NR>7 && $9+0>20 {printf "%s:%s ", int($9), $12}' | head -c 150)
  echo "$(( $(date +%s) - t0 )) $mbps $mem $thr" >> "$OUT"
done
