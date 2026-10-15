#!/bin/bash
# Where does a TensorFold start spend its time?  (no py-spy needed)
#  1. phase split from the timestamps of the server's own "[tensorfold]" log lines
#  2. optional sampling profile: TensorFold prints every thread's Python stack on SIGUSR1
#     (tensorfold/server/stacks.py), so `SAMPLE=1` sends USR1 once a second while it loads
#     and counts the innermost tensorfold frames of the main thread.
# usage: tf-load-profile.sh            # analyse the current/last start of the container
#        SAMPLE=1 tf-load-profile.sh   # run it right after the start was triggered; samples until ready
C=${CONTAINER:-Qwen3.8-Flash-Next-MLX-4bit-TensorFold}
URL=${URL:-http://127.0.0.1:8001}

if [ "${SAMPLE:-0}" = 1 ]; then
  old=$(docker inspect -f '{{.Id}}' "$C" 2>/dev/null)          # a kept, stopped container from the last run
  until [ "$(docker inspect -f '{{.Id}} {{.State.Running}}' "$C" 2>/dev/null)" != "$old true" ] &&
        [ "$(docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null)" = true ] &&
        docker logs "$C" 2>&1 | grep -q '\[tensorfold\] loading'; do sleep 0.5; done   # USR1 is armed from here on
  n=0
  until curl -sf --max-time 2 "$URL/v1/models" >/dev/null 2>&1; do
    docker kill -s USR1 "$C" >/dev/null 2>&1 || break
    n=$((n+1)); sleep 1
  done
  echo "sent $n stack requests"
fi

echo "== phases (container clock, seconds since container start)"
start=$(docker inspect -f '{{.State.StartedAt}}' "$C")
docker logs -t "$C" 2>&1 | python3 -c '
import sys, re, datetime
def t(s): return datetime.datetime.fromisoformat(s[:26].rstrip("Z")).timestamp()
t0 = t(sys.argv[1]); prev = t0
for line in sys.stdin:
    m = re.match(r"(\S+) (.*)", line)
    if not m or "[tensorfold]" not in m.group(2): continue
    if "request error" in m.group(2): continue
    now = t(m.group(1))
    print(f"  +{now-t0:7.1f}s  (step {now-prev:6.1f}s)  {m.group(2)[:150]}")
    prev = now
' "$start"

if [ "${SAMPLE:-0}" = 1 ]; then
  echo "== sampled stacks: main thread, innermost tensorfold frame (1 sample ~ 1 s)"
  docker logs "$C" 2>&1 | python3 "$(dirname "$0")/tf-stacks.py"
fi
