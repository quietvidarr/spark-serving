# spark-serving

Local LLM serving on one DGX Spark (GB10): one model at a time behind an OpenAI-compatible router.

- `model-router/` — the router on :8888. Models run on 127.0.0.1:8001. See its [README](model-router/README.md), including the engine forwarding contract.
- `prompt-cache/` — tools for the TensorFold SSD prompt cache: status, live checks, resume scenarios.
- `localpatch/` — update-safe load-time patches (`sitecustomize.py`).
- `update-recipes.sh`, `switch-model.sh` — update recipes, switch models from the CLI.
- `pull-retry.sh`, `resumable-pull.sh` — Docker pulls over flaky Wi-Fi.
- `bench*.py`, `prefill-bench.py`, `bench-*.txt` — benchmarks and results.
- `REPORT.md` — setup and measurement notes.

Recipes:
- [qwen38-flash-next-dgx-spark](https://github.com/quietvidarr/qwen38-flash-next-dgx-spark)
- [qwen36-35b-a3b-exl3-tensorfold-dgx-spark](https://github.com/quietvidarr/qwen36-35b-a3b-exl3-tensorfold-dgx-spark)
- [TensorFold-spark](https://github.com/quietvidarr/TensorFold-spark)

Paths assume `~/recipes`; `/home/USER` in `models.json` is a placeholder. `YOUR-ORG/Qwen3.8-Flash-Next-NVFP4`
in `models.json` is too: set your NVFP4 weights repo there. No auth: LAN only.
