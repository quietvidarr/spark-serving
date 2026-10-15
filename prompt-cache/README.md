# Prompt cache (TensorFold models + the router)

Reprocessing a long prompt is the slowest part of an agent turn: about 15 s for 30K tokens on Qwen3.8 Flash EXL3.
This cache lets TensorFold models skip most of that work, across turns, conversations, model switches and restarts.

| Part | Where | What it does |
|---|---|---|
| More resume points | TensorFold patches 0006 + 0010, `cuda/markers.py` | keeps the state at EVERY message start (`TF_POINTS_LAST=all`), not only after the system prompt and at the prompt end |
| Never drop a point | patch 0010 (+ 0107 on the 35B), `cuda/prompt_store.py` | a full write queue makes a point wait in GPU memory, never skips it; a journal counts what a kill lost |
| GPU eviction order | patch 0010, `cuda/keep_rank.py` | history points leave GPU memory first, shared starting points last |
| Disk store | TensorFold patch 0006, `cuda/prompt_store.py` + `families/qwen4_exp/cuda/disk_points.py` | every kept point is also written to the SSD and read back when a request needs it |
| Request log | router (`~/recipes/model-router/router.py`) | one line per completion in `state/requests.jsonl`; summary at `GET /router/cache` |
| Data | `~/recipes/prompt-cache/store/` | `index.sqlite` + one folder per model key |

## How it works

**Resume points.** TensorFold can only resume a prompt from a saved state, not from any token: most of Qwen's layers
keep a running state that can't be rewound. Upstream saves 2 points per prompt: after the system prompt and at the
prompt end. Patch 0006 added the last 3 message starts; patch 0010 (2026-10-14) saves EVERY message start.

So a fork, a regenerate or an edit at message k (any chat UI or agent) resumes at message k's start, in GPU memory
or on disk, and recomputes only from there; the original conversation's points are never changed (a resume copies
the point into a slot). Two starts closer than 256 tokens (`MIN_GAP`) keep only the earlier one, so the resume point
is at most 256 tokens before the changed message (one tiny message, e.g. "ok").

Cost: each kept point splits the prefill, about one more pass over the weights (measured ~110 ms on Flash, ~40 ms
on the 35B). A chat built turn by turn meets its new starts as its last ones and pays the same as before; a long
history arriving at once (imported, or an early message edited with the later ones kept) pays per message.
`TF_POINTS_GAP=1024` thins only those older starts (a fork there then recomputes up to 1,024 tokens more).

This also fixes memory-injecting agents. Some agents add recalled memories to the newest user message and leaves them out when it sends the
same message as history on the next turn. Before, the next turn's prompt differed inside that message, the prompt-end
point no longer matched, and the engine went back to the end of the system prompt and recomputed everything. Now it
resumes at that message's start and recomputes only the last turn, a few thousand tokens. Nothing is added to the
context, so the client's context budget is exactly what it expects. It works the same for any client.

**Disk store.** Each saved point is:

- the attention rows below it, in blocks of 1,024 tokens. A block is named by a hash of every token up to its end, so
  the points of one conversation, and conversations that share a head, write each block once;
- its own last partial block plus the recurrent (DeltaNet) state, about 115 MB for Flash Next.

When a request arrives, the engine takes the longest saved point its prompt extends, first in GPU memory, then on disk.
A disk point is loaded only when reading it is faster than recomputing the tokens it covers: an estimated
`gain > 1.5 × load time + 0.05 s`, using the measured read speed (`TF_PROMPT_CACHE_READ_GBS` is the starting guess).
Loading copies the point into a free GPU slot and the file stays as it was.

**Never losing a point (patch 0010).** Each point is copied to host memory as soon as it is made (a side stream, no
stall) and written by a background thread right away, not when the server is idle. If the writer falls behind and
the copies waiting for it reach `TF_PROMPT_CACHE_QUEUE_GIB` (4 GiB), a new point is no longer skipped (the old
behaviour): it waits in its GPU slot, which is not reused or evicted until the point is copied (rows below the point
are never rewritten; the Flash slot is copied first if it must be reused, the 35B's state is held). Waiting points are
copied as soon as the queue has room, and all of them the moment the server goes idle. A stopping server (`stop.sh`:
SIGTERM, 30 s) waits up to `TF_PROMPT_CACHE_EXIT_S` (20 s) for the writer. Every queued point is noted in
`store/<key>/journal.log`; the next start reports the points queued but never written as `LOST ... (previous run N)`.
Counters (log every 10 min and `store/<key>/stats.json`): `deferred` (waited on the GPU), `forced` (copied past the
cap), `lost`, `lost_before`, `queue_peak`. What this cannot cover: a crash, OOM kill, `docker kill` or power loss
loses the points still in memory (in practice the last seconds of prefill; they are counted at the next start), and
a file being written at that moment is a torn `.tmp` that is deleted at the next start (files are written to `.tmp`
and renamed, so the index never points at a half file).

**GPU memory (kept points).** Over the GPU cap (`TF_FLASH_KEEP` / `TF_KEEP`), `cuda/keep_rank.py` drops: first a point
inside a conversation's history (a newer point of it is kept, and it is on disk), then a conversation's newest point
or a one-off prompt's end, and last a shared starting point: a system-prompt point a request resumed from, a point
resumed with 2+ different continuations (forks, regenerates, other conversations), or the base of 2+ kept points that
went different ways. Oldest first within each; a point not yet copied for the disk goes last of all.

**Normal and pinned points.**

| | Normal | Pinned |
|---|---|---|
| Budget | `TF_PROMPT_CACHE_GIB` = 100 GiB | `TF_PROMPT_CACHE_PINNED_GIB` = 50 GiB |
| Lifetime | deleted `TF_PROMPT_CACHE_TTL_H` = 72 h after it was saved; least recently used first when over budget | no time limit; least recently used first only when over budget |
| Becomes one | every point the engine keeps | a point that `TF_PROMPT_CACHE_PIN_AFTER` = 3 different continuations resumed from, in practice a system prompt plus tools (agent profiles) |

A pinned point is a normal point under different keep rules. It is never loaded ahead of time and never changed. A
request that matches it gets a copy in a GPU slot, and the points created after it are normal ones. Blocks used by a
pinned point count toward the pinned budget.

**Model updates.** The store key covers the image tag (TensorFold version + our patches), the weights revision, the KV
format and the cache layout. After any change the old points are never read. Old normal points expire on their own.
Old pinned points are recomputed under the new key: after start, while the server has had no request for 30 s, the
engine sends each one to itself (`/v1/completions`, 1 token), one at a time. Then the old copy is deleted.

## Settings

All of these go in the recipe's `.env` (see `local/config.sh`). Changing one recreates the container on its next start.

| Variable | Default | Meaning |
|---|---|---|
| `PROMPT_CACHE_DIR` | `~/recipes/prompt-cache/store` | host folder; empty = disk store off |
| `TF_POINTS_LAST` | all | resume points at every message start (patch 0010); N = the last N; 0 = upstream |
| `TF_POINTS_GAP` | 256 | with `all`: starts before a prompt's last 3 kept at least this far apart (256 = every one); see the cost note |
| `TF_PROMPT_CACHE_QUEUE_GIB` | 4 | host copies waiting for the writer; past it new points wait in GPU memory |
| `TF_PROMPT_CACHE_EXIT_S` | 20 | a stopping server waits this long for the writer |
| `TF_LOCAL_IMAGE` (recipe `.env`) | empty | pin an earlier image tag (rollback), e.g. the one before 0010 |
| `TF_PROMPT_CACHE_GIB` | 100 | normal points budget |
| `TF_PROMPT_CACHE_TTL_H` | 72 | normal point lifetime, from when it was saved |
| `TF_PROMPT_CACHE_PINNED_GIB` | 50 | pinned points budget |
| `TF_PROMPT_CACHE_PIN_AFTER` | 3 | different continuations that pin a point |
| `TF_PROMPT_CACHE_QUEUE_GIB` | 2 | host memory for points waiting to be written; past it a point is skipped, never waited for |
| `TF_PROMPT_CACHE_PREFILL_TPS` | 1800 | prefill speed the load decision compares against |
| `TF_PROMPT_CACHE_READ_GBS` | 2 | first guess of the read speed; replaced by measurements |

The budgets are shared by every model that uses the same folder.

## Checking it

- `curl -s localhost:8888/router/cache | python3 -m json.tool` shows, for the last 24 h (`?hours=N`): cached ÷ prompt
  tokens per model and per conversation, the median time to first token, the last 20 requests, and the disk store's
  size per model and pin state.
- `~/recipes/prompt-cache/status.sh` shows the same disk-store figures from the host.
- The engine log (`docker logs <model container> | grep prompt-cache`) has one line per disk load (tokens, seconds,
  GB/s), each new pin, and a summary every 10 minutes.

- `MODEL=<router name> python3 scenarios.py run --out FILE` (hold `flock ~/recipes/tensorfold-decode-lab/gpu.lock`
  when other GPU jobs run): a 19-message conversation turn by turn, a fork, a regenerate, edits, memory-agent turns and a
  restart through the router (points from disk), with cached / prefilled / ideal tokens, TTFT and an exactness check
  against `"draft": false` (a full prefill). `scenarios.py cost` times a cold 44-message prompt (resume-point cost).
  `live-check.py` is the older memory-agent / eviction / pin check.

`cached_tokens` comes from the engine. vLLM leaves it empty unless started with `--enable-prompt-tokens-details`.

## Limits

- Only TensorFold's concurrent decoder on one GPU (`STREAMS` ≥ 2) uses the disk store. The serial engine and two-GPU
  runs keep upstream behaviour.
- Prompts with images are never saved (as upstream: their states depend on the image).
- The Flash Next family (qwen4_exp) and the 35B (qwen3_5_moe, patches 0104/0106/0107) have disk adapters (`disk_points.py`). Another family needs its own adapter: the same
  three functions over its own state layout.
- A load stops the engine's other streams for its duration, as a prefill does: about 0.2 s for 30K tokens.

## Adding a NAS later

The engine reads and writes only the local folder. That's deliberate: over this Spark's Wi-Fi (about 8 MB/s measured),
reading a 30K-token point takes ~100 s against ~15 s to recompute it. A NAS can only be a longer-term copy that comes
back to the SSD before it is used. The files are immutable and content-named, so mirroring is plain file copying:

1. **Mount** the NAS share on the host, e.g. `/mnt/nas/prompt-cache` (NFS or SMB; `noatime`).
2. **Push** (a cron job or systemd timer, every 10–30 min):
   `rsync -a --ignore-existing --bwlimit=2000 ~/recipes/prompt-cache/store/ /mnt/nas/prompt-cache/`
   - `--bwlimit` in KB/s keeps the Wi-Fi usable; 2000 ≈ 2 MB/s.
   - Skip the run while the router is busy: check `curl -s localhost:8888/router/status` for `in_flight` 0 first.
   - Copy `index.sqlite` with `sqlite3 index.sqlite ".backup /mnt/nas/prompt-cache/index.sqlite"`, not rsync, so the copy is consistent.
   - Without `--delete`, the NAS keeps points the SSD has expired. Prune it on its own schedule, e.g. files older than 30 days.
3. **Pull back** (by hand or on a schedule): copy wanted `p/` and `b/` files of the current model key back into the
   store folder while the model is stopped, then insert their rows from the NAS index into the local one. A future
   `restore.py` would do this per conversation or per pinned point.
4. **Tier code**: `prompt_store.Tier` is the extension point (`has`, `get`, `put`, `evict`). A second tier that the
   engine reads directly only makes sense on Ethernet. 1 GbE gives ~110 MB/s, which beats recomputing Qwen3.8 above
   ~65 MB/s. The load decision then needs that tier's own measured speed.

Wi-Fi rule of thumb: break-even read speed is roughly *bytes per token × prefill tokens per second*. For Flash Next
that's ~28 KB × 1,800 ≈ 50–65 MB/s. Slower models (lower prefill speed) break even at lower speeds.
