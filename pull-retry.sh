#!/bin/bash
# usage: pull-retry.sh IMAGE LOG  — retries docker pull until it succeeds
img="$1"; log="$2"
for i in $(seq 1 50); do
  echo "=== attempt $i $(date) ===" >> "$log"
  docker pull "$img" >> "$log" 2>&1 && { echo "=== PULL OK $(date) ===" >> "$log"; exit 0; }
  sleep 30
done
echo "=== PULL GAVE UP $(date) ===" >> "$log"; exit 1
