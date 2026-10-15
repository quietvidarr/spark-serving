"""Live fork / regenerate / edit / memory-agent / restart scenarios for the TensorFold prompt cache, through the router.

  MODEL=<router name> scenarios.py [run|cost] [--no-restart] [--out FILE]

run   (a) a 19-message conversation built turn by turn, (b) a fork at a middle user message (and the branch's next
      turn, then the original's next turn), (c) regenerating a middle reply, (d) editing an earlier user message
      (truncating, chat-UI style, and in place keeping the later messages), (e) memory-agent turns (recalled
      memories in the newest user message only), (f) a restart through the router, then a fork, a regenerate and
      the original's next turn again (their points now come from disk).
cost  the prefill cost of resume points: one cold 44-message prompt (many tiny messages), twice.

Every request is greedy and streams. For each one the table shows prompt tokens, cached tokens, prefilled tokens,
the IDEAL prefill (prompt tokens after the closest message start at or before the first token that differs from any
earlier request of this run; worked out afterwards with the model's own tokenizer and chat template, run in the model's
image) and TTFT. "exact" compares the reply with the same request sent with "draft": false, which the engine never
resumes (a full prefill from token 0, no MTP drafts): greedy replies must be identical.
The run's system prompt starts with a fresh nonce, so nothing of an earlier run is reused.
"""
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

ROUTER = os.environ.get("ROUTER", "http://127.0.0.1:8888")
MODEL = os.environ.get("MODEL", "Qwen3.8-Flash-Next-EXL3-4.05bpw-TensorFold")
RECIPES = {"Qwen3.8-Flash-Next-EXL3-4.05bpw-TensorFold": "qwen3.8-flash-next-exl3-tensorfold",
           "Qwen3.6-35B-A3B-EXL3-6.00bpw-TensorFold": "qwen3.6-35b-a3b-exl3-tensorfold"}
SRC = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob("/usr/lib/python3*/email/*.py")))
LENS = [3000, 80, 1500, 40, 4000, 200, 1200, 30, 2500]     # characters of the 9 user messages (~4 per token)
LOG: list[dict] = []


def post(path, body, timeout=3600):
    req = urllib.request.Request(ROUTER + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def ask(label, msgs, max_tokens=32, check=False, ref=None):
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    text, p, c, ttft = _stream(body)
    row = {"label": label, "msgs": [dict(m) for m in msgs], "prompt": p, "cached": c, "ttft": ttft, "reply": text}
    if check:
        cold, _, c0, _ = _stream({**body, "draft": False})
        row["exact"] = cold == text
        row["cold_cached"] = c0
        if cold != text:
            row["cold_reply"] = cold
    if ref is not None:
        row["same_as"] = ref == text
    LOG.append(row)
    print(f"{label:34s} prompt {p:6d} cached {c:6d} prefilled {p - c:6d} TTFT {ttft:6.2f}s"
          + (f" exact {row['exact']}" if check else "") + (f" same {row['same_as']}" if ref is not None else ""),
          flush=True)
    return text


def _stream(body):
    t0 = time.perf_counter(); first = None; usage = None; text = ""
    for raw in post("/v1/chat/completions", body):
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:]); usage = d.get("usage") or usage
        for ch in d.get("choices", []):
            piece = (ch.get("delta") or {}).get("content") or ""
            if piece and first is None:
                first = time.perf_counter() - t0
            text += piece
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return text, usage["prompt_tokens"], cached, first or time.perf_counter() - t0


def user(t, extra=""):
    a = 5000 + 7000 * t
    return f"Question {t}{extra}: what does this part do, in one sentence?\n" + SRC[a:a + LENS[t % len(LENS)]]


def wait_loaded(timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            st = json.load(urllib.request.urlopen(ROUTER + "/router/status", timeout=10))
            if st.get("loaded") == MODEL and st.get("state") not in ("switching", "starting"):
                return time.time() - t0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    raise TimeoutError("the model did not come back")


def run(restart=True):
    nonce = time.strftime("%Y%m%d-%H%M%S")
    system = f"[{nonce}] You are a careful coding agent. Tools and rules follow.\n" + SRC[:6000]
    hist = [{"role": "system", "content": system}]
    print(f"== (a) build-up, nonce {nonce}", flush=True)
    for t in range(9):
        hist.append({"role": "user", "content": user(t)})
        hist.append({"role": "assistant", "content": ask(f"a turn {t + 1} ({len(hist)} msgs)", hist)})
    k = 9                                                     # user message 5 (system, u1, a1, ...: index 2t-1)
    print("== (b) fork at message %d" % k, flush=True)
    branch = hist[:k] + [{"role": "user", "content": user(20, " (fork)")}]
    branch.append({"role": "assistant", "content": ask("b fork at msg 9", branch, check=True)})
    branch.append({"role": "user", "content": user(21)})
    ask("b fork's next turn", branch)
    orig = hist + [{"role": "user", "content": user(22)}]
    ask("b original's next turn", orig)
    print("== (c) regenerate the reply to message 7", flush=True)
    ask("c regenerate msg 8 (reply to u4)", hist[:8], check=True, ref=hist[8]["content"])
    print("== (d) edit user message 3", flush=True)
    ask("d edit msg 3, truncate", hist[:3] + [{"role": "user", "content": user(1, " (edited)")}], check=True)
    edited = list(hist); edited[3] = {"role": "user", "content": user(1, " (edited in place)")}
    ask("d edit msg 3, keep later msgs", edited + [{"role": "user", "content": user(23)}], check=True)
    print("== (e) memory-agent turns (memories in the newest user message only)", flush=True)
    h = list(hist)
    for t in range(3):
        mem = f"<memory-context>\n{SRC[90000 + t * 2500:92500 + t * 2500]}\n</memory-context>\n"
        q = user(30 + t)
        reply = ask(f"e memory turn {t + 1}", h + [{"role": "user", "content": mem + q}], check=(t == 2))
        h += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]
    if restart:
        print("== (f) restart through the router, then from disk", flush=True)
        time.sleep(20)                                        # the disk writer finishes its queue
        st = json.load(urllib.request.urlopen(ROUTER + "/router/status", timeout=10))
        if st.get("in_flight") or st.get("waiting"):
            print("router busy: restart skipped", flush=True)
        else:
            post("/router/restart", {"model": MODEL}).read()
            print(f"restarted, back after {wait_loaded():.0f} s", flush=True)
            ask("f fork at msg 13 (disk)", hist[:13] + [{"role": "user", "content": user(24, " (fork)")}],
                check=True)
            ask("f regenerate msg 12 (disk)", hist[:12], check=True, ref=hist[12]["content"])
            ask("f original's next turn (disk)", orig)
            ask("f edit msg 5, keep later (disk)", hist[:5] + [{"role": "user", "content": user(2, " (edit2)")}]
                + hist[6:] + [{"role": "user", "content": user(25)}], check=True)


def cost():
    for rep in range(2):
        msgs = [{"role": "system", "content": f"[{time.time_ns()}] Cost probe.\n" + SRC[:4000]}]
        for t in range(22):
            msgs.append({"role": "user", "content": SRC[10000 + t * 3000:10000 + t * 3000 + (2400 if t % 2 else 60)]})
            msgs.append({"role": "assistant", "content": "Noted." if t % 2 else SRC[50000 + t * 500:50000 + t * 500 + 400]})
        msgs.append({"role": "user", "content": "Summarize all of the above in one sentence."})
        ask(f"cost cold 44+1 msgs #{rep + 1}", msgs, max_tokens=1)


def ideal(path):
    """Ideal prefill per request: tokens after the closest message start <= the first token differing from an
    earlier request of the run (the model's tokenizer and template, in its image)."""

    recipe = os.path.expanduser(f"~/recipes/{RECIPES[MODEL]}")
    cfg = subprocess.run(["bash", "-c", f'cd {recipe} && source local/config.sh >/dev/null 2>&1 && echo "$MODEL_DIR" '
                          '&& local_image'], capture_output=True, text=True).stdout.split("\n")
    mdir, image = cfg[0], cfg[1]
    code = ("import json,sys\nfrom tokenizers import Tokenizer\nfrom tensorfold.cuda.chat_template import ChatTemplate\n"
            "from tensorfold.cuda.markers import TemplateTokens\nfrom tensorfold.engine.prefill_plan import message_markers\n"
            "t=TemplateTokens(Tokenizer.from_file('/m/tokenizer.json'),ChatTemplate(__import__('pathlib').Path('/m')))\no,a=message_markers(t)\n"
            "rows=json.load(sys.stdin)\nprint(json.dumps({'openers':[int(x) for x in o],'ids':[t.apply_chat_template("
            "m,add_generation_prompt=True,enable_thinking=False) for m in rows]}))\n")
    code = code.replace("/m", mdir)
    hub = os.path.expanduser("~/.cache")
    r = subprocess.run(["docker", "run", "--rm", "-i", "--entrypoint", "python3", "-v", f"{hub}:{hub}:ro", image, "-c", code],
                       input=json.dumps([row["msgs"] for row in LOG]), capture_output=True, text=True)
    if r.returncode:
        print("tokenize failed:", r.stderr[-500:]); return
    out = json.loads(r.stdout.strip().splitlines()[-1])
    opener = set(out["openers"])
    seen = []
    for row, ids in zip(LOG, out["ids"]):
        starts = [i for i, x in enumerate(ids) if x in opener]
        best = 0
        for prev, pstarts in seen:
            lcp = 0
            for x, y in zip(ids, prev):
                if x != y:
                    break
                lcp += 1
            best = max([best] + [s for s in pstarts if s <= lcp and s < len(ids)])
        row["ideal_prefill"] = len(ids) - best
        row["tok_len"] = len(ids)
        seen.append((ids, starts))
    print(f"\n{'request':34s} {'prompt':>6s} {'cached':>6s} {'prefill':>7s} {'ideal':>6s} {'TTFT s':>7s}  exact")
    for row in LOG:
        print(f"{row['label']:34s} {row['prompt']:6d} {row['cached']:6d} {row['prompt'] - row['cached']:7d} "
              f"{row.get('ideal_prefill', -1):6d} {row['ttft']:7.2f}  "
              f"{row.get('exact', '')}{' same-as-orig ' + str(row['same_as']) if 'same_as' in row else ''}"
              + ("" if row.get("tok_len") in (None, row["prompt"]) else f"  (tokenizer {row['tok_len']})"))
    with open(path, "w") as f:
        json.dump([{k: v for k, v in row.items() if k != "msgs"} for row in LOG], f, indent=1)


if __name__ == "__main__":
    what = next((a for a in sys.argv[1:] if not a.startswith("-")), "run")
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"/tmp/scenarios-{int(time.time())}.json"
    if what == "cost":
        cost()
    else:
        run(restart="--no-restart" not in sys.argv)
    ideal(out)
