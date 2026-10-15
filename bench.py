#!/usr/bin/env python3
"""Small speed test for an OpenAI-compatible server: TTFT, prefill tok/s, decode tok/s.
Usage: python3 bench.py [base_url] [label]   (stdlib only)"""
import json, sys, time, urllib.request, concurrent.futures as cf

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8888/v1"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "model"
KEY = None
import os
KEY = os.environ.get("API_KEY")

def hdrs():
    h = {"Content-Type": "application/json"}
    if KEY: h["Authorization"] = f"Bearer {KEY}"
    return h

def get_model():
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/models", headers=hdrs()), timeout=30)
    return json.load(r)["data"][0]["id"]

MODEL = get_model()

FILLER = ("The quick brown fox jumps over the lazy dog while engineers benchmark "
          "large language models on a small desktop supercomputer. ")

def run(prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "min_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(), headers=hdrs())
    t0 = time.perf_counter(); ttft = None; usage = None; last = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    now = time.perf_counter()
                    if ttft is None: ttft = now - t0
                    last = now
    total = time.perf_counter() - t0
    pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
    decode_t = (last - t0 - ttft) if last and ct > 1 else float("nan")
    return {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": ttft, "total_s": total,
            "prefill_tps": pt / ttft, "decode_tps": (ct - 1) / decode_t if decode_t else float("nan")}

def fmt(name, r):
    print(f"  {name:<28} prompt={r['prompt_tokens']:>6} tok  gen={r['completion_tokens']:>4} tok  "
          f"TTFT={r['ttft_s']*1000:8.0f} ms  prefill={r['prefill_tps']:8.0f} tok/s  "
          f"decode={r['decode_tps']:6.1f} tok/s  total={r['total_s']:6.1f} s", flush=True)

print(f"== {LABEL}  ({MODEL}) @ {BASE}")
run("Say hi.", 8)  # warmup
results = {}
for name, words, gen in [("short prompt", 0, 256), ("~2k-token prompt", 2000, 256),
                         ("~8k-token prompt", 8000, 256), ("~32k-token prompt", 32000, 128)]:
    n = max(1, words // 20)
    # unique prefix per test so prefix caching does not hide prefill cost
    prompt = f"[{name} {time.time()}] " + FILLER * n + "\nSummarize the text above in detail."
    if words == 0: prompt = f"[{time.time()}] Write a long story about a robot learning to paint."
    results[name] = run(prompt, gen); fmt(name, results[name])

for conc in (4, 8):
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(conc) as ex:
        rs = list(ex.map(lambda i: run(f"[{i} {time.time()}] Write a long story about a robot learning to paint.", 256), range(conc)))
    wall = time.perf_counter() - t0
    agg = sum(r["completion_tokens"] for r in rs) / wall
    per = sum(r["decode_tps"] for r in rs) / conc
    ttft = sum(r["ttft_s"] for r in rs) / conc
    print(f"  {conc} concurrent streams          aggregate={agg:6.1f} tok/s  per-stream decode={per:6.1f} tok/s  "
          f"avg TTFT={ttft*1000:.0f} ms  wall={wall:.1f} s", flush=True)
