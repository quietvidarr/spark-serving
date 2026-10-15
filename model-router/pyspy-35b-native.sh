#!/bin/bash
# Profile the 35B's torch.compile cache loading ("Directly load the compiled graph(s)")
# during a real switch. Needs --cap-add=SYS_PTRACE in the 35B's EXTRA_DOCKER_ARGS.
C=Qwen3.6-35B-A3B-NVFP4
OUT=~/recipes/model-router/prof/pyspy35native
mkdir -p $OUT
curl -s --max-time 3600 -X POST http://127.0.0.1:8888/router/switch -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B-NVFP4"}' > $OUT/switch.json &
until [ "$(docker inspect -f '{{.State.Running}}' $C 2>/dev/null)" = true ] && \
      docker inspect -f '{{.HostConfig.CapAdd}}' $C | grep -q SYS_PTRACE && \
      [ "$(docker inspect -f '{{.State.StartedAt}}' $C | cut -c1-16)" \> "$(date -u -d '-3 min' +%Y-%m-%dT%H:%M)" ]; do sleep 1; done
docker exec $C pip install -q py-spy 2>&1 | grep -v "Running pip as" | tail -1
# the compile-cache load starts right after the 2nd "Loading weights took" (MTP) line
until [ "$(docker logs $C 2>&1 | grep -c 'Loading weights took')" -ge 2 ]; do sleep 0.5; done
PID=$(docker exec $C bash -c 'for p in /proc/[0-9]*; do n=$(cat $p/comm 2>/dev/null); case "$n" in VLLM::EngineCor*|VLLM::Worker*) echo ${p#/proc/};; esac; done' | head -1)
echo "profiling pid $PID ($(docker exec $C cat /proc/$PID/comm))"
docker exec $C py-spy record --native -p $PID -d 25 -r 100 -f raw -o /tmp/compile.raw 2>&1 | grep -v "flamegraph" | tail -1
docker cp $C:/tmp/compile.raw $OUT/compile.raw
wait
echo done
