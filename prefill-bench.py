#!/usr/bin/env python3
"""Uncached prefill benchmark with real, varied text (what an agent sends), stdlib only.

Each request is a fresh slice of local text (Python stdlib source + /usr/share/doc), behind a unique nonce, so
no prefix cache helps. Prefill tok/s = prompt tokens / time to first token (one output token, thinking off).
--cold FILE drops FILE from the page cache before each request (e.g. an n-gram table), to measure the table
read from SSD. Usage:
  prefill-bench.py --model NAME [--base http://127.0.0.1:8888/v1] [--sizes 8000,32000] [--reps 2]
                   [--kind code|prose|mix] [--cold FILE ...] [--json out.json]"""
import argparse
import glob
import gzip
import json
import os
import random
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--base", default="http://127.0.0.1:8888/v1")
p.add_argument("--sizes", default="8000,32000")
p.add_argument("--reps", type=int, default=2)
p.add_argument("--kind", default="mix", choices=["code", "prose", "mix"])
p.add_argument("--cold", action="append", default=[])
p.add_argument("--seed", type=int, default=None)
p.add_argument("--json")
a = p.parse_args()


def corpus(kind):
    parts = []
    if kind in ("code", "mix"):
        for f in sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))[:4000]:
            try:
                parts.append(open(f, errors="ignore").read())
            except OSError:
                pass
    if kind in ("prose", "mix"):
        for f in sorted(glob.glob("/usr/share/doc/**/*", recursive=True)):
            try:
                if f.endswith(".gz"):
                    parts.append(gzip.open(f, "rt", errors="ignore").read())
                elif os.path.isfile(f) and os.path.getsize(f) < 2_000_000:
                    t = open(f, errors="ignore").read()
                    if t.isprintable() or "\n" in t:
                        parts.append(t)
            except (OSError, EOFError, ValueError):
                pass
    random.Random(1).shuffle(parts)
    return "\n\n".join(parts)


TEXT = corpus(a.kind)
rng = random.Random(a.seed if a.seed is not None else time.time_ns())


def drop(paths):
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def once(chars):
    start = rng.randrange(0, len(TEXT) - chars)
    prompt = (f"[{time.time_ns()}] Summarize the following material in one sentence.\n\n"
              + TEXT[start:start + chars])
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if ttft is None and any((c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content")
                                    for c in d.get("choices", [])):
                ttft = time.perf_counter() - t0       # the first real token (the role-only chunk comes at once)
    total = time.perf_counter() - t0
    n = (usage or {}).get("prompt_tokens")
    return n, ttft or total, total, len(prompt)


results = []
ratio = 3.2 if a.kind == "code" else 3.6          # chars per token, refined after the first request
for size in [int(s) for s in a.sizes.split(",")]:
    for rep in range(a.reps):
        if a.cold:
            drop(a.cold)
        n, ttft, total, chars = once(int(size * ratio))
        if n:
            ratio = chars / n                       # this text's characters per token, for the next request
        rate = n / ttft if n else None
        results.append(dict(size=size, tokens=n, ttft_s=round(ttft, 2), prefill_tps=round(rate) if rate else None,
                            cold=bool(a.cold)))
        print(f"{a.model} {a.kind} ~{size}: {n} tokens, TTFT {ttft:.2f}s, prefill {rate:.0f} tok/s"
              f"{' (table dropped from cache)' if a.cold else ''}", flush=True)
if a.json:
    json.dump(results, open(a.json, "w"), indent=1)
