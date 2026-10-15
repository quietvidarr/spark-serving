# spark-serving

One model at a time on a DGX Spark (GB10), behind an OpenAI-compatible router.

| Path | What |
|---|---|
| `model-router/` | Router on :8888, models on 127.0.0.1:8001. [README](model-router/README.md) |
| `prompt-cache/` | TensorFold SSD prompt cache tools. [README](prompt-cache/README.md) |
| `localpatch/` | Update-safe load-time patches (`sitecustomize.py`) |
| `update-recipes.sh`, `switch-model.sh` | Update recipes, switch models |
| `pull-retry.sh`, `resumable-pull.sh` | Docker pulls over flaky links |
| `bench*.py`, `prefill-bench.py`, `bench-*.txt` | Benchmarks, results |
| `REPORT.md` | Setup and measurements |

Recipes: [qwen38-flash-next-dgx-spark](https://github.com/quietvidarr/qwen38-flash-next-dgx-spark), [qwen36-35b-a3b-exl3-tensorfold-dgx-spark](https://github.com/quietvidarr/qwen36-35b-a3b-exl3-tensorfold-dgx-spark), [TensorFold-spark](https://github.com/quietvidarr/TensorFold-spark).

Notes:
- Paths assume `~/recipes`. `/home/USER` and `YOUR-ORG/Qwen3.8-Flash-Next-NVFP4` in `models.json` are placeholders.
- No auth. LAN only.
