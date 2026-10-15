# spark-serving

One model at a time on a DGX Spark (GB10), behind an OpenAI-compatible router. Start here: [Setup](#setup).

| Path | What |
|---|---|
| `model-router/` | Router on :8888, models on 127.0.0.1:8001. [README](model-router/README.md) |
| `prompt-cache/` | TensorFold SSD prompt cache tools. [README](prompt-cache/README.md) |
| `localpatch/` | Update-safe load-time patches (`sitecustomize.py`) |
| `update-recipes.sh`, `switch-model.sh` | Update recipes, switch models |
| `pull-retry.sh`, `resumable-pull.sh` | Docker pulls over flaky links |
| `bench*.py`, `prefill-bench.py`, `bench-*.txt` | Benchmarks, results |
| `REPORT.md` | Measurements, history |

## Setup

Needs: DGX Spark (GB10, 128 GB), Docker with GPU, git, curl, jq; ~100 GB disk for Flash weights, ~27 GB for the 35B.
While the repos are private: a GitHub token with read access, for `git clone` and as `GH_TOKEN` for `local/build.sh` (it fetches the TensorFold fork).

```bash
mkdir -p ~/recipes && cd ~/recipes   # any dir works: then set "dir" in model-router/models.json
git clone -b local-patches https://github.com/quietvidarr/qwen38-flash-next-dgx-spark
git clone https://github.com/quietvidarr/qwen36-35b-a3b-exl3-tensorfold-dgx-spark
git clone https://github.com/quietvidarr/spark-serving
export GH_TOKEN=<token>              # while private

cd ~/recipes/qwen38-flash-next-dgx-spark                # Flash; its build also makes the 35B's base image
local/download.sh && local/build.sh
cd ~/recipes/qwen36-35b-a3b-exl3-tensorfold-dgx-spark   # finds Flash as a sibling dir, else FLASH_RECIPE in .env
local/download.sh && local/build.sh && local/prebuild.sh

cd ~/recipes/spark-serving/model-router
mkdir -p ~/.cache/tensorfold-prompt-cache
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d --build
curl -s localhost:8888/v1/models
curl -s localhost:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "35b-tf", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 256}'
```
`models.json` has the two TensorFold recipes; vLLM NVFP4 entries are optional, in `models.example-vllm.json`.
The first request starts the model: minutes on the first start (kernel compiles).
Prompt cache: `~/.cache/tensorfold-prompt-cache`, shared by both models. Move it with `PROMPT_CACHE_DIR` in each recipe's `.env` and for `docker compose`; empty in `.env` = off.

Recipes: [qwen38-flash-next-dgx-spark](https://github.com/quietvidarr/qwen38-flash-next-dgx-spark), [qwen36-35b-a3b-exl3-tensorfold-dgx-spark](https://github.com/quietvidarr/qwen36-35b-a3b-exl3-tensorfold-dgx-spark), [TensorFold-spark](https://github.com/quietvidarr/TensorFold-spark).

Notes:
- `YOUR-ORG/Qwen3.8-Flash-Next-NVFP4` in `models.example-vllm.json` is a placeholder.
- No auth. LAN only.
