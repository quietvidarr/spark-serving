#!/bin/bash
# Control the model router (http://<spark>:8888). Works from any computer:
#   ROUTER=http://<spark-host>:8888 ./switch-model.sh status
# usage: switch-model.sh 35b | flash | tensorfold | stop | status   (any model id or alias)
# Switching waits for in-flight requests to finish, then swaps (35b ~6 min, flash ~13 min).
R="${ROUTER:-http://127.0.0.1:8888}"
case "$1" in
  status) curl -s "$R/router/status" | jq '{state, loaded, switching_to, in_flight, waiting, last_used, last_error}' ;;
  stop)   curl -s -X POST "$R/router/stop" | jq '{state, loaded}' ;;
  "")     echo "usage: $0 35b | flash | tensorfold | stop | status"; exit 1 ;;
  *)      curl -s -X POST "$R/router/switch" -H 'Content-Type: application/json' \
            -d "{\"model\":\"$1\"}" | jq '{loaded, state, last_error, error}' ;;
esac
