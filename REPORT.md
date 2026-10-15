# Update 2026-10-01 — third model: Qwen3.8 Flash Next on TensorFold

**Status: installed, wired into the router (:8888), tested, switch time halved.**

| | |
|---|---|
| Router id | `Qwen3.8-Flash-Next-MLX-4bit-TensorFold` (aliases `tensorfold`, `tf`, `flash-tf`, `flash-tensorfold`, `flash-next-tensorfold`, `qwen3.8-flash-next-tensorfold`, `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`) |
| Recipe | ~/recipes/qwen3.8-flash-next-tensorfold (third-party TensorFold recipe), branch `local-patches` |
| Engine | TensorFold 0.3.6.3 (not vLLM), image `tensorfold-qwen38:v0.3.6.3` (= GHCR `…:v0.3.6.3-c1f5d72f8d16`, 24.6 GB unpacked) |
| Weights | Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP, 113 GB (MLX 4-bit). The vLLM recipe's NVFP4 weights can't be reused: different checkpoint/quantization; the recipe's patches, `--ple-on-ssd`, vision and memory plan are for the MLX one |
| Serving | 127.0.0.1:8001 only (router owns the LAN side), 5 streams x 262,144 tokens, int8 KV, images/video, MTP, thinking on by default |
| Context in /v1/models | 262144 (learned live from TensorFold's /health, source `tensorfold`) |
| Container | `Qwen3.8-Flash-Next-MLX-4bit-TensorFold`, kept when stopped, restart policy `no` |

Memory: TensorFold budgets MemAvailable minus 12.2 GiB and needs 102.5 GiB → fits only when nothing else
big runs (budget seen 103.5–104.2 GiB). MemAvailable while serving ~11 GiB, minimum during loads 10.0–10.9 GiB.

## Switch times (router, measured 2026-10-01; "ready" = start launched → model serves its name)

| | before (recipe as shipped) | after |
|---|---|---|
| TensorFold load (ready), 3–6 runs | **90–92 s** | **43–45 s** |
| 35B → TensorFold, full request | 111 s | 66 s |
| Flash (vLLM) → TensorFold, full request | 128 s | 74 s |
| TensorFold → 35B (ready) | 117 s | 114 s (unchanged, as before) |
| TensorFold → Flash vLLM (ready) | 177 s | 177 s (unchanged) |
| Very first TensorFold start (one-time CUDA kernel builds) | 188 s | – |

"Full request" also contains the drain and the first answer; with other clients sending ~20k-token prompts,
their held requests are prefilled first after a switch, so the answer part varied 4–33 s. TensorFold reports its
own compute time per request: 0.2 s when nothing else was queued.

Where TensorFold's load went (stack sampling via TensorFold's own SIGUSR1 dump, `model-router/tf-load-profile.sh`):
start ~1 s, memory admission ~1.5 s, weights ~84 s → ~38 s, vision tower ~4 s, kernel warm-up ~3 s.

What helped (all in ~/recipes/localpatch, fingerprint-guarded, each measured with 3 restarts, others on):

| patch | what | effect |
|---|---|---|
| tf-fast-nibbles | expert-weight nibble reorder (runs on ~60 GiB at every start) as 2 int32 delta swaps instead of 8 int64 passes; bit-identical | 81 s → 44 s |
| tf-load-no-flush | skip `torch.cuda.empty_cache()` after each of 48 layers, flush once at the end | 50–53 s → 44 s |
| tf-pinned-reader | read through 2 reused pinned 64 MiB buffers with async GPU copies; page-cache drop on a background thread | 47 s → 44 s |

Tried, no gain, removed: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` (94 vs 92 s), O_DIRECT reads (44–45 vs 43–44 s),
read-ahead of the next layer (fadvise WILLNEED) and of the vision tower. Not applicable: kept/standby container
(start overhead is ~1 s), page-cache warmth (TensorFold drops the checkpoint's pages on purpose), JIT caches
(kernel builds are already cached by the recipe in ~/.cache/tensorfold-qwen38), image warm-up (TensorFold warms the
vision tower at start; first image request = first text request).

Output unchanged: same token hashes (TensorFold's `token_sha`) with patches on and off for greedy text, code,
thinking, seeded sampling and an image prompt; loaded weights compared byte-for-byte for 46 tensors.

Speed (bench.py on :8001, another client active, so decode numbers are noisy): prefill ~2,350 tok/s at 8–35k
tokens, TTFT 0.19 s short / 3.8 s at 8.8k / 15 s at 35k; decode 34–58 tok/s single stream. Raw: bench-qwen3.8-flash-next-tensorfold.txt.

## Router changes (engine-agnostic for real now)
- `engine` per model in models.json (`vllm` default, `tensorfold`, `openai` = only the OpenAI API): drain
  (vLLM metrics / TensorFold /health requests_running), context length (vLLM max_model_len / TensorFold
  /health context_length / generic fields).
- Images: `image_cmd` (recipe tells which registry tag it wants + local name) and `image_update`; locally
  re-tagged images are compared by digest. Weights: `weights_links: "cache"` for recipes that mount the whole HF cache.
- Aliases can no longer silently shadow another model (logged and ignored).

## Notes
- Download: hf_xet stalled at ~2 MB/s (adaptive concurrency stuck at 1) → finished with curl, every file checked
  against its sha256 (`.dl-logs/tf-weights-curl.sh`).
- 35B's weights update would report "links point outside the model folder" (hf 2.x keeps blobs in a shared
  store); the recipe mounts the whole cache, so adding `"weights_links": "cache"` to its models.json entry fixes the
  report. Not changed (existing model).

# DGX Spark setup report — 2026-09-28

## Status: both models installed and tested

| Component | State | Where |
|---|---|---|
| Qwen3.6 35B-A3B NVFP4 | installed, tested OK, stopped | ~/recipes/qwen3.6-35b |
| Qwen3.8 Flash Next NVFP4 | installed, tested OK, **left running** on 127.0.0.1:8888 | ~/recipes/qwen3.8-flash-next |

Both models answered a correctness check (19*23 = 437).

## Speed test (bench.py, thinking off, fp8 KV, single Spark)

| | Qwen3.6 35B | Qwen3.8 Flash Next |
|---|---|---|
| Startup time | 6 min 17 s | 13 min 18 s (first start incl. 27 GB PLE table build) |
| TTFT, short prompt | 0.10 s | ~0.25 s (warm) |
| TTFT, 2.2k-token prompt | 0.40 s | 2.8 s |
| TTFT, 8.8k-token prompt | 1.5 s | 4.1 s |
| TTFT, 35k-token prompt | 10.9 s | 16.3 s |
| Prefill speed | ~5,700–5,900 tok/s (3,200 at 35k) | ~2,150 tok/s |
| Decode, 1 stream | ~77–82 tok/s | ~35 tok/s (creative) / 45–50 tok/s (summaries) |
| 4 streams, total | 137 tok/s | 74 tok/s |
| 8 streams, total | 268 tok/s | 81 tok/s (max 4 parallel; rest queue) |

Raw results: bench-qwen3.6-35b.txt, bench-qwen3.8-flash-next.txt

## Memory (121 GB unified)

| | Qwen3.6 35B | Qwen3.8 Flash Next |
|---|---|---|
| Weights | 24.8 GiB | 76.5 GiB (weights + non-torch) |
| KV cache reserved | 64.3 GiB = 5.7M tokens (21× full 262k context) | 17.4 GiB = 1.16M tokens (4.4× full context) |
| System used while running | ~108 GB | ~105 GB |

The 35B only *needs* ~25 GB; the rest is KV cache it reserves because of
`--gpu-memory-utilization 0.80` in start.sh. Lower it (e.g. 0.40) to free memory.

## Switching models

    ~/recipes/switch-model.sh 35b      # stop whatever runs, start the 35B (~6 min)
    ~/recipes/switch-model.sh flash    # start Flash Next (~10-13 min)
    ~/recipes/switch-model.sh stop
    ~/recipes/switch-model.sh status

API for apps: http://127.0.0.1:8888/v1 (OpenAI-compatible). From another machine:
`ssh -L 8888:localhost:8888 user@<spark-host>`, then http://localhost:8888/v1.

## Changes vs. the recipe defaults
- Both models bind to **127.0.0.1** instead of 0.0.0.0 (the defaults exposed an
  unauthenticated API to the whole LAN). qwen3.6-35b/start.sh line 7 now reads
  `HOST="${HOST:-127.0.0.1}"`; Flash Next is started with `BIND=127.0.0.1`.
  To expose on the LAN, set HOST/BIND=0.0.0.0 and add an API key.
- qwen3.6-35b/.cache/huggingface is a symlink to ~/.cache/huggingface.

## Problems hit (all fixed)
- Wi-Fi only ~8 MB/s; downloads (~99 GB + 25 GB models, ~40 GB images) took ~6 h.
- Flash Next model download stopped once; restarted detached and resumed.
- Docker image pulls failed with "unexpected EOF" and blocked each other; ran them one
  at a time with ~/recipes/pull-retry.sh.

## Not done / optional
- Flash Next suggests kernel tuning `sudo sysctl -p ~/recipes/qwen3.8-flash-next/files/sysctl-spark3.conf` (needs sudo).
- Neither model auto-starts after reboot.
- Ethernet would make future model downloads much faster.
