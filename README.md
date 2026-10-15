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

Needs: DGX Spark (GB10, 128 GB), Docker with the NVIDIA runtime, git, curl. Disk: ~100 GB Flash weights, ~27 GB 35B.

```bash
git clone https://github.com/quietvidarr/spark-serving && cd spark-serving
./install.sh                                                       # router on :8888; re-run any time
curl -X POST localhost:8888/router/install -d '{"model":"35b-tf"}'  # or Install at http://<spark>:8888
curl -s localhost:8888/router/installs                             # progress, log tail
curl -s localhost:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"35b-tf","messages":[{"role":"user","content":"Hi"}]}'
```
Recipes go to `~/recipes` (`dir` in `model-router/models.json`). The 35B installs Flash's recipe and image first.
Switch: name another model (`flash-tf`, `35b-tf`, `any` = whatever is loaded). First start: minutes (kernel compiles).
Prompt cache: `~/.cache/tensorfold-prompt-cache`, shared. vLLM entries: `models.example-vllm.json`.

Recipes: [qwen38-flash-next-dgx-spark](https://github.com/quietvidarr/qwen38-flash-next-dgx-spark), [qwen36-35b-a3b-exl3-tensorfold-dgx-spark](https://github.com/quietvidarr/qwen36-35b-a3b-exl3-tensorfold-dgx-spark), [TensorFold-spark](https://github.com/quietvidarr/TensorFold-spark).

Notes:
- `YOUR-ORG/Qwen3.8-Flash-Next-NVFP4` in `models.example-vllm.json` is a placeholder.
- No auth. LAN only.
