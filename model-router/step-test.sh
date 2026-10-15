#!/bin/bash
# usage: step-test.sh <label> <model> [<model> ...]  — switch to each model in turn, timing each
L=$1; shift
D=~/recipes/model-router/prof; mkdir -p $D
for m in "$@"; do
  ~/recipes/model-router/profile-load.sh $D/$L-$m.txt 900 & P=$!
  t0=$(date +%s)
  r=$(curl -s --max-time 3600 -X POST http://127.0.0.1:8888/router/switch -H 'Content-Type: application/json' -d "{\"model\":\"$m\"}" | jq -c '{loaded,state,e: (.last_error // "" | .[0:120])}')
  t=$(( $(date +%s)-t0 ))
  kill $P 2>/dev/null
  # first real request after the switch (captures JIT-on-first-use spikes)
  f=$(curl -s -o /dev/null -w '%{time_total}' http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$m\",\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"Write one sentence about the sea.\"}]}")
  s=$(curl -s -o /dev/null -w '%{time_total}' http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$m\",\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"Write one sentence about a forest.\"}]}")
  c=$(docker ps --format '{{.Names}}' | grep -E 'NVFP4' | head -1)
  ph=$(docker logs $c 2>&1 | grep -E "Loading weights took|Available KV|init engine|Multi-modal warmup completed" | sed -E 's/.*\] //' | cut -c1-60 | paste -sd'|')
  echo "$(date +%T) [$L] -> $m: switch ${t}s | 1st req ${f}s, 2nd ${s}s | $r | $ph"
done
