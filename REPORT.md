# Report

Summary: three models on one DGX Spark behind the router. TensorFold Flash Next load cut 90 → 44 s by three load patches. NVFP4 35B is the fastest; Flash Next NVFP4 the slowest at concurrency.

## 2026-10-01: Qwen3.8 Flash Next on TensorFold

| | |
|---|---|
| Router id | `Qwen3.8-Flash-Next-MLX-4bit-TensorFold` (aliases `tf`, `flash-tf`, ...) |
| Engine | TensorFold 0.3.6.3, image `tensorfold-qwen38:v0.3.6.3` (24.6 GB) |
| Weights | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`, 113 GB. NVFP4 weights not reusable |
| Serving | 127.0.0.1:8001, 5 streams × 262,144, int8 KV, vision, MTP, thinking on |
| Memory | Needs 102.5 GiB of MemAvailable − 12.2 GiB: nothing else big can run |

### Switch times (start → serving)

| | before | after |
|---|---|---|
| TensorFold load | 90-92 s | 43-45 s |
| 35B → TensorFold, full request | 111 s | 66 s |
| Flash vLLM → TensorFold, full request | 128 s | 74 s |
| TensorFold → 35B | 117 s | 114 s |
| TensorFold → Flash vLLM | 177 s | 177 s |
| First start (kernel builds) | 188 s | – |

Load: weights ~84 → ~38 s; rest ~10 s (admission, vision, warm-up).

| Patch (`localpatch`) | What | Effect |
|---|---|---|
| tf-fast-nibbles | Nibble reorder as 2 int32 swaps, bit-identical | 81 → 44 s |
| tf-load-no-flush | One `empty_cache()` at the end, not per layer | 50-53 → 44 s |
| tf-pinned-reader | 2 pinned 64 MiB buffers, async copies | 47 → 44 s |

No gain, removed: `expandable_segments`, O_DIRECT, read-ahead.
Output unchanged: same `token_sha` on/off; 46 tensors byte-identical.
Speed: prefill ~2,350 tok/s (8-35k); TTFT 0.19 / 3.8 / 15 s (short / 8.8k / 35k); decode 34-58 tok/s. Raw: `bench-qwen3.8-flash-next-tensorfold.txt`.

### Router changes

- `engine` per model (`vllm`, `tensorfold`, `openai`): drain and context length per engine.
- `image_cmd`, `image_update`; re-tagged images compared by digest. `weights_links: "cache"`.
- Aliases can't shadow another model.

## 2026-09-28: NVFP4 setup

| | Qwen3.6 35B | Qwen3.8 Flash Next |
|---|---|---|
| Startup | 6 min 17 s | 13 min 18 s (first, incl. 27 GB PLE table) |
| TTFT short / 2.2k / 8.8k / 35k | 0.10 / 0.40 / 1.5 / 10.9 s | 0.25 / 2.8 / 4.1 / 16.3 s |
| Prefill | ~5,700-5,900 tok/s (3,200 at 35k) | ~2,150 tok/s |
| Decode, 1 stream | ~77-82 tok/s | ~35-50 tok/s |
| 4 / 8 streams total | 137 / 268 tok/s | 74 / 81 tok/s (max 4 parallel) |
| Weights | 24.8 GiB | 76.5 GiB |
| KV reserved | 64.3 GiB (5.7M tokens) | 17.4 GiB (1.16M tokens) |

bench.py, thinking off, fp8 KV. Raw: `bench-qwen3.6-35b.txt`, `bench-qwen3.8-flash-next.txt`.

Notes:
- 35B KV size comes from `--gpu-memory-utilization 0.80`; lower it to free memory.
- Both bind 127.0.0.1 (recipe defaults exposed an unauthenticated API).
- Slow link: use `pull-retry.sh`, one pull at a time.
- Switch: `switch-model.sh 35b|flash|stop|status`.
