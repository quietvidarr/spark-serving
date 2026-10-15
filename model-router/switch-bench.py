#!/usr/bin/env python3
"""Time model switches through the router, the way a client experiences them (stdlib only).

usage: switch-bench.py MODEL [MODEL ...]      e.g. switch-bench.py tensorfold 35b tensorfold flash
       ROUTER=http://host:8888 switch-bench.py ...

For each model in turn it sends one small chat request naming that model and reports
  total   wall time of the request (drain + stop old + start new + answer)
  stop    time the router spent stopping the previous model   (router log)
  ready   "ready after" = start command launched -> model serves its name (router log)
  answer  total - stop - ready (drain, the request itself, and anything queued ahead of it)
  (srv)   the engine's own compute time for the request, where it reports it (TensorFold)
plus the engine's own load phases when it prints them (TensorFold: "loaded in").
Run on the Spark itself: the phase split comes from `docker logs model-router`.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

R = os.environ.get("ROUTER", "http://127.0.0.1:8888")


def get(path):
    return json.load(urllib.request.urlopen(R + path, timeout=30))


def ask(model):
    body = {"model": model, "max_tokens": 20, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "What is 19*23? Answer with just the number."}]}
    req = urllib.request.Request(R + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    if "error" in d:
        return "ERROR: " + str(d["error"].get("message"))[:300], None
    tf = d.get("tensorfold") or {}           # TensorFold reports its own prefill/decode seconds
    srv = (tf.get("prefill_s") or 0) + (tf.get("decode_s") or 0) if tf else None
    return (d["choices"][0]["message"].get("content") or "").strip()[:40], srv


def router_log(since):
    try:
        out = subprocess.run(["docker", "logs", "--since", since, "model-router"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout + out.stderr
    except Exception:
        return ""


def ts(line):
    m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+)", line)
    return time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) + int(m.group(2)) / 1000 if m else None


def main():
    models = sys.argv[1:]
    if not models:
        sys.exit(__doc__)
    print(f"{'from':<40} {'to':<40} {'total':>7} {'stop':>6} {'ready':>6} {'answer':>6} {'(srv)':>6}  reply")
    for want in models:
        before = get("/router/status").get("loaded")
        since = time.strftime("%Y-%m-%dT%H:%M:%S")
        t0 = time.time()
        reply, srv = ask(want)
        total = time.time() - t0
        after = get("/router/status").get("loaded")
        log = router_log(since)
        stop = ready = None
        t_stop = [ts(l) for l in log.splitlines() if " stopping " in l]
        t_start = [ts(l) for l in log.splitlines() if " starting " in l]
        m = re.search(r"ready after (\d+)s", log)
        if t_stop and t_start and t_stop[0] and t_start[-1]:
            stop = t_start[-1] - t_stop[0]
        if m and t_start:
            t_ready = [ts(l) for l in log.splitlines() if "ready after" in l]
            ready = (t_ready[-1] - t_start[-1]) if t_ready and t_ready[-1] else float(m.group(1))
        f = lambda x: f"{x:6.1f}" if x is not None else "     -"
        answer = total - (stop or 0) - (ready or 0) if ready is not None else total
        print(f"{str(before):<40} {str(after):<40} {total:7.1f} {f(stop)} {f(ready)} {f(answer)} {f(srv)}  {reply!r}",
              flush=True)


if __name__ == "__main__":
    main()
