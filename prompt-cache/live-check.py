"""Live checks of the prompt cache (patches 0006/0104) through the router. MODEL=<name> live-check.py [memory|evict|pin|all]
memory: memory-agent turns (recalled memories in the newest user message, left out of history) -> cached per turn.
evict:  a conversation pushed out of GPU memory by 10 others comes back from disk; greedy reply equals the cold one.
pin:    3 conversations with one system prompt and different questions -> its point is pinned."""
import glob
import os
import json
import sqlite3
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8888/v1"
MODEL = os.environ.get("MODEL", "Qwen3.8-Flash-Next-EXL3-4.05bpw-TensorFold")
SRC = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob("/usr/lib/python3*/email/*.py")))
DB = os.environ.get("PROMPT_CACHE_DB", os.path.expanduser("~/recipes/prompt-cache/store/index.sqlite"))


def ask(msgs, max_tokens=64, temperature=0.0):
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter(); first = None; usage = None; text = ""
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=3600)
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:]); usage = d.get("usage") or usage
        for ch in d.get("choices", []):
            c = (ch.get("delta") or {}).get("content") or ""
            if c and first is None:
                first = time.perf_counter() - t0
            text += c
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return text, usage["prompt_tokens"], cached, first or time.perf_counter() - t0


def memory_agent(tag):
    system = f"You are agent {tag}. Tools and rules follow.\n" + SRC[:12000]
    hist = [{"role": "system", "content": system}]
    for t in range(5):
        memory = f"<memory-context>\n{SRC[20000 + t * 3000:23000 + t * 3000]}\n</memory-context>\n"
        q = f"Turn {t}: in one sentence, what does the code above do? " + SRC[40000 + t * 4000:44000 + t * 4000]
        sent = hist + [{"role": "user", "content": memory + q}]
        text, p, c, ttft = ask(sent)
        print(f"memory turn {t}: prompt {p}, cached {c} ({100 * c / p:.0f}%), TTFT {ttft:.2f}s", flush=True)
        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": text}]


def conv_msgs(i):
    return [{"role": "system", "content": f"Coding agent {i}."},
            {"role": "user", "content": (SRC * 2)[i * 17000:i * 17000 + 30000] + "\nSummarize in one sentence."}]


def evict():
    msgs = conv_msgs(20)
    cold, p, c, t = ask(msgs, 48)
    print(f"target cold: prompt {p}, cached {c}, TTFT {t:.2f}s", flush=True)
    ths = [threading.Thread(target=lambda i=i: ask(conv_msgs(i), 8)) for i in range(10)]
    for th in ths:
        th.start(); time.sleep(0.3)
    for th in ths:
        th.join()
    print("10 other conversations done", flush=True)
    again, p, c, t = ask(msgs, 48)
    print(f"target again: prompt {p}, cached {c}, TTFT {t:.2f}s; same greedy reply: {again == cold}", flush=True)
    if again != cold:
        print("cold :", repr(cold)); print("again:", repr(again))


def pin():
    system = "You are a shared-profile agent.\n" + SRC[60000:72000]
    for q in ("What is a MIME type?", "Name one class.", "What does a parser do?", "Explain headers."):
        _, p, c, t = ask([{"role": "system", "content": system}, {"role": "user", "content": q}], 8)
        print(f"pin q: prompt {p}, cached {c}, TTFT {t:.2f}s", flush=True)
    time.sleep(2)
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    print("pinned points:", db.execute("SELECT n, hits, conts FROM points WHERE pinned=1").fetchall())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("memory", "all"):
        memory_agent(str(time.time_ns()))
    if what in ("evict", "all"):
        evict()
    if what in ("pin", "all"):
        pin()
