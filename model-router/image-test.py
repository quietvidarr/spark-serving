#!/usr/bin/env python3
"""Send one image request (a generated 640x480 PNG) and time it. usage: image-test.py <model>"""
import base64, json, struct, sys, time, urllib.request, zlib

def png(w, h):
    rows = b"".join(b"\x00" + bytes(((x * 255 // w), (y * 255 // h), 128) * 1)
                    for y in range(h) for x in [0] for _ in [0]) if False else None
    raw = b"".join(b"\x00" + b"".join(bytes((x * 255 // w, y * 255 // h, 128)) for x in range(w)) for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))

model = sys.argv[1] if len(sys.argv) > 1 else "any"
url = "data:image/png;base64," + base64.b64encode(png(640, 480)).decode()
body = {"model": model, "max_tokens": 60, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "Describe the colors in this image in one sentence."}]}]}
t0 = time.time()
req = urllib.request.Request("http://127.0.0.1:8888/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
print(f"{model}: image request {time.time() - t0:.2f}s -> {r['choices'][0]['message']['content'][:90]!r}")
