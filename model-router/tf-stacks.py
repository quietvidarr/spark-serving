#!/usr/bin/env python3
"""Aggregate faulthandler all-thread dumps (TensorFold's SIGUSR1 stacks) from stdin:
main-thread samples by innermost tensorfold frame and by leaf frame."""
import collections
import re
import sys

inner, leaf, blocks = collections.Counter(), collections.Counter(), []
cur = None
for line in sys.stdin:
    if line.startswith(("Thread 0x", "Current thread 0x")):
        cur = []
        blocks.append(cur)
        continue
    m = re.match(r'\s+File "([^"]+)", line (\d+) in (\S+)', line)
    if m and cur is not None:
        cur.append(m.groups())
    elif cur is not None and cur:
        cur = None
n = 0
for b in blocks:
    if not any(fn in ("cmd_serve", "_serve_cuda") for _, _, fn in b):
        continue          # main thread only
    n += 1
    tf = [f for f in b if "tensorfold/" in f[0]]
    short = lambda f: f"{f[0].split('tensorfold/')[-1].split('/')[-1]}:{f[2]}:{f[1]}"
    if tf:
        inner[short(tf[0])] += 1
    leaf[f"{b[0][0].split('/')[-1]}:{b[0][2]}  <-  {short(tf[0]) if tf else '?'}"] += 1
print(f"  {n} main-thread samples")
print("  -- by innermost tensorfold frame")
for k, v in inner.most_common(20):
    print(f"  {v:4d}  {k}")
print("  -- by leaf frame")
for k, v in leaf.most_common(20):
    print(f"  {v:4d}  {k}")
