#!/bin/bash
# Times model switches through the router and checks keep-alive while waiting.
U=${ROUTER:-http://localhost:8888}
req() {  # model stream(true/false)
  local t0=$(date +%s.%N)
  curl -sN --max-time 3600 "$U/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$1\",\"stream\":$2,\"max_tokens\":20,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in 5 words.\"}]}" \
    > ${TMPDIR:-/tmp}/switch-$$.out
  local t=$(python3 -c "print(round($(date +%s.%N)-$t0))")
  local ka=$(grep -c '^: waiting' ${TMPDIR:-/tmp}/switch-$$.out)
  local sp=$(head -c 200 ${TMPDIR:-/tmp}/switch-$$.out | tr -cd ' ' | wc -c)
  local ans
  if [ "$2" = true ]; then ans=$(grep '^data: {' ${TMPDIR:-/tmp}/switch-$$.out | sed 's/^data: //' | jq -rj '.choices[0].delta.content // empty' 2>/dev/null)
  else ans=$(jq -r '.choices[0].message.content // .error.message' ${TMPDIR:-/tmp}/switch-$$.out); fi
  echo "$(date +%T) model=$1 stream=$2 -> ${t}s  keepalives=$ka leading_spaces=$sp  answer: $(echo $ans | head -c 60)"
}
mkdir -p ${TMPDIR:-/tmp}
req Qwen3.6-35B-A3B-NVFP4 true     # switch Flash -> 35B, first start fills caches
req Qwen3.8-Flash-Next-NVFP4 false # switch 35B -> Flash (eager load)
req Qwen3.6-35B-A3B-NVFP4 true     # switch Flash -> 35B, caches warm
rm -f ${TMPDIR:-/tmp}/switch-$$.out
docker logs model-router 2>&1 | grep -E "ready after|failed|disconnected" | tail -6
