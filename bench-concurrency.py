"""Prefill and decode speed by concurrency. [BASE=url] [ROWS=quick] concbench.py MODEL [label]
Every prompt starts with a unique nonce (no prompt-cache hits); replies forced to GEN tokens (ignore_eos).
steady decode = tokens all streams produce while every one of them is decoding (after the last first token, before the
first end), per stream and in total."""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8888/v1")
MODEL = sys.argv[1]
FILLER = ("The quick brown fox jumps over the lazy dog while engineers benchmark "
          "large language models on a small desktop supercomputer. ")


def run(prompt, gen):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": gen, "min_tokens": gen,
            "temperature": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; usage = None; marks = []; chars = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                text = (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content") or ""
                if text:
                    now = time.perf_counter()
                    first = first or now
                    chars += len(text)
                    marks.append((now, chars))
    end = time.perf_counter()
    return {"t0": t0, "first": first, "end": end, "prompt": usage["prompt_tokens"], "gen": usage["completion_tokens"],
            "marks": marks}


def between(r, a, b):
    """Tokens a stream produced in [a, b] (its characters' share of its reply)."""

    total = r["marks"][-1][1]
    ca = max([c for t, c in r["marks"] if t <= a], default=0)
    cb = max([c for t, c in r["marks"] if t <= b], default=0)
    return r["gen"] * (cb - ca) / total


def prompt(words, i):
    return f"[run {time.time_ns()}-{i}] " + FILLER * (words // 20) + "\nWrite a long story about it."


def row(conc, words, gen):
    with cf.ThreadPoolExecutor(conc) as ex:
        rs = list(ex.map(lambda i: run(prompt(words, i), gen), range(conc)))
    start = min(r["t0"] for r in rs)
    ttft = [r["first"] - r["t0"] for r in rs]
    dec = [(r["gen"] - 1) / (r["end"] - r["first"]) for r in rs]
    ptoks = sum(r["prompt"] for r in rs)
    prefill_agg = ptoks / (max(r["first"] for r in rs) - start)
    wall = max(r["end"] for r in rs) - start
    a, b = max(r["first"] for r in rs), min(r["end"] for r in rs)
    steady = sum(between(r, a, b) for r in rs) / (b - a) if b - a > 0.5 else float("nan")
    print(f"  conc {conc}  prompt {rs[0]['prompt']:6d}  gen {gen}  TTFT avg {sum(ttft) / conc * 1000:7.0f} ms "
          f"max {max(ttft) * 1000:7.0f} ms  prefill agg {prefill_agg:7.0f} tok/s  decode/stream "
          f"{sum(dec) / conc:6.1f} tok/s  decode agg {sum(dec):6.1f} tok/s  steady {steady / conc:6.1f}/stream "
          f"{steady:6.1f} total  wall {wall:6.1f} s", flush=True)


print(f"== {sys.argv[2] if len(sys.argv) > 2 else MODEL}  ({time.strftime('%Y-%m-%d %H:%M')})", flush=True)
run("hi", 4)                                                   # switch/warm
if os.environ.get("ROWS") == "decode":
    for conc in (1, 2, 4, 8):
        row(conc, 200, 512)
else:
    for words, gen in ((2000, 512), (8000, 256)):
        for conc in (1, 2, 4, 8):
            row(conc, words, gen)
    row(1, 30000, 128)
