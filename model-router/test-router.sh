#!/bin/bash
# End-to-end test of the model router from the LAN address.
U=${1:-http://localhost:8888}
ts() { date +%H:%M:%S; }
ask() {  # model maxtokens prompt
  curl -s "$U/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$1\",\"max_tokens\":$2,\"messages\":[{\"role\":\"user\",\"content\":\"$3\"}],\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    -D - -o /tmp/router-test-$$.json | grep -i x-router-model | tr -d '\r'
  jq -r '"   answer: " + (.choices[0].message.content // .error.message | tostring | .[0:80] | gsub("\n";" ")) + "  | tokens=" + (.usage.completion_tokens|tostring)' /tmp/router-test-$$.json
}
echo "$(ts) 1) model=any with nothing loaded (expect: starts last-used = qwen3.6-35b)"
ask any 20 "Say hello in 5 words."
echo "$(ts) done 1"

echo "$(ts) 2) long 35b request in background, then a flash request (expect: 35b finishes first, then switch)"
( ask 35b 1500 "Write a detailed 1000-word essay about the history of the bicycle."; echo "$(ts) long 35b request FINISHED" ) &
sleep 5
curl -s "$U/router/status" | jq -c '{state,loaded,in_flight}'
ask flash 400 "What is 19*23? Answer with just the number."
echo "$(ts) done 2 (flash answered)"
wait
curl -s "$U/router/status" | jq -c '{state,loaded}'

echo "$(ts) 3) model=any while flash is loaded (expect: flash, no switch)"
ask any 20 "Say hello in 5 words."
echo "$(ts) done 3"

echo "$(ts) 4) model=qwen3.6-35b (expect: switch back)"
ask qwen3.6-35b 20 "Say hello in 5 words."
echo "$(ts) done 4"
curl -s "$U/v1/models" | jq -c '[.data[] | {id,loaded}]'
rm -f /tmp/router-test-$$.json
