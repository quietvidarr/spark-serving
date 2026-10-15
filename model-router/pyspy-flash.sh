#!/bin/bash
# Profile Flash Next's weight loading with py-spy during a real switch.
C=Qwen3.8-Flash-Next-NVFP4
OUT=~/recipes/model-router/prof/pyspy
mkdir -p $OUT
curl -s --max-time 3600 -X POST http://127.0.0.1:8888/router/switch -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next-NVFP4"}' > $OUT/switch.json &
# wait for the new container, then install py-spy inside it
until docker ps --format '{{.Names}} {{.CreatedAt}}' | grep -q "^$C .*$(date +%Y-%m-%d)" && \
      [ "$(docker inspect -f '{{.State.Running}}' $C 2>/dev/null)" = true ] && \
      docker inspect -f '{{.HostConfig.CapAdd}}' $C | grep -q SYS_PTRACE; do sleep 2; done
docker exec $C pip install -q py-spy 2>&1 | tail -1
# wait until main weights are loading (non-offload progress bar)
until docker logs $C 2>&1 | grep -aE "Loading safetensors checkpoint shards( \(eager\))?: +[0-9]+% Completed \| [1-9][0-9]*/" | grep -qv PLE; do sleep 2; done
W=$(docker exec $C bash -c 'for p in /proc/[0-9]*; do c=$(tr "\0" " " < $p/cmdline 2>/dev/null); n=$(cat $p/comm 2>/dev/null); echo "${p#/proc/} $n $c"; done' | awk '$2 ~ /VLLM::Worker|python/ {print}' )
echo "$W" > $OUT/procs.txt
PID=$(docker exec $C bash -c 'for p in /proc/[0-9]*; do [ "$(cat $p/comm 2>/dev/null)" = "VLLM::Worker" ] && echo ${p#/proc/}; done' | head -1)
echo "worker pid=$PID" >> $OUT/procs.txt
docker exec $C py-spy record -p $PID -d 60 -r 50 -f raw -o /tmp/load.raw --nonblocking 2>&1 | tail -2
docker cp $C:/tmp/load.raw $OUT/load.raw
wait
echo done
