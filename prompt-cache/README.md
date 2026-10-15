# Prompt cache

SSD prompt cache for TensorFold models: skip re-prefill across turns, conversations, switches and restarts.
30K tokens on Qwen3.8 Flash EXL3: ~15 s prefill vs ~0.2 s load.

| Part | Where | What |
|---|---|---|
| Resume points | patches 0006 + 0010, `cuda/markers.py` | State at every message start (`TF_POINTS_LAST=all`) |
| Never drop | patch 0010 (+ 0107 35B), `cuda/prompt_store.py` | Full queue: point waits in GPU memory; journal counts losses |
| GPU eviction | patch 0010, `cuda/keep_rank.py` | History first, shared starts last |
| Disk store | patch 0006, `prompt_store.py` + `disk_points.py` | Every kept point written to SSD, loaded on demand |
| Request log | router | `state/requests.jsonl`, `GET /router/cache` |
| Data | `PROMPT_CACHE_DIR`, default `~/.cache/tensorfold-prompt-cache/` | `index.sqlite` + one folder per model key |

## How it works

- Resume only from saved states (recurrent layers can't rewind). Fork/regenerate/edit at message k resumes at k's start (≤ 256 tokens before).
- Each point costs ~one extra weights pass (~110 ms Flash, ~40 ms 35B). `TF_POINTS_GAP` thins older starts.
- Fixes clients that rewrite the newest message in history: resume one turn back, not from the system prompt.
- Point = 1,024-token attention blocks (hash-named, shared) + tail + recurrent state (~115 MB Flash).
- Disk load only when `gain > 1.5 × load time + 0.05 s`.
- Crash/OOM/kill loses points still in memory; counted at next start. Files written to `.tmp` then renamed.
- Store key = image tag + weights revision + KV format + layout. Old normal points expire; old pinned points recomputed when idle.

| | Normal | Pinned |
|---|---|---|
| Budget | 100 GiB | 50 GiB |
| Lifetime | 72 h, LRU over budget | none, LRU over budget |
| From | every kept point | 3 different continuations resumed from it |

## Settings (recipe `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `PROMPT_CACHE_DIR` | `~/.cache/tensorfold-prompt-cache` | empty = off |
| `TF_POINTS_LAST` | all | every message start; N = last N; 0 = upstream |
| `TF_POINTS_GAP` | 256 | min gap for older starts |
| `TF_PROMPT_CACHE_QUEUE_GIB` | 4 | host queue; past it points wait in GPU |
| `TF_PROMPT_CACHE_EXIT_S` | 20 | writer wait on stop |
| `TF_PROMPT_CACHE_GIB` | 100 | normal budget |
| `TF_PROMPT_CACHE_TTL_H` | 72 | normal lifetime |
| `TF_PROMPT_CACHE_PINNED_GIB` | 50 | pinned budget |
| `TF_PROMPT_CACHE_PIN_AFTER` | 3 | continuations to pin |
| `TF_PROMPT_CACHE_PREFILL_TPS` | 1800 | load-decision prefill speed |
| `TF_PROMPT_CACHE_READ_GBS` | 2 | initial read-speed guess |
| `TF_LOCAL_IMAGE` | empty | pin older image (rollback) |

Budgets are shared by all models using the folder.

## Check

```bash
curl -s localhost:8888/router/cache | python3 -m json.tool   # ?hours=N
prompt-cache/status.sh
docker logs <container> | grep prompt-cache
MODEL=<name> python3 prompt-cache/scenarios.py run --out FILE   # fork/regenerate/edit/restart, exactness
python3 prompt-cache/scenarios.py cost                          # resume-point cost
```
vLLM: `cached_tokens` needs `--enable-prompt-tokens-details`.

## Limits

- Disk store only with the concurrent decoder (`STREAMS` ≥ 2), one GPU.
- Image prompts never saved.
- Adapters: Flash Next (`qwen4_exp`), 35B (`qwen3_5_moe`). Other families need their own `disk_points.py`.
- A load pauses other streams (~0.2 s for 30K).
- Local folder only. Remote copy: mirror with `rsync --ignore-existing`, `sqlite3 .backup` for the index. Break-even read speed ≈ bytes/token × prefill tok/s (~50-65 MB/s Flash). `prompt_store.Tier` is the extension point.
