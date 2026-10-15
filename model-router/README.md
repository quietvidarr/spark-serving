# model-router

OpenAI-compatible router on :8888. Runs one model at a time on 127.0.0.1:8001, switches on demand.

| File | What |
|---|---|
| `router.py` | Server |
| `thinking.py` | Thinking/reasoning normalization |
| `models.json` | Models, read at start |
| `test_thinking.py`, `test_install.py` | Unit tests (`test_install.py` needs aiohttp) |
| `ui.html` | Test page |

## Install a model

```bash
curl -X POST localhost:8888/router/install -d '{"model": "35b-tf"}'   # or Install in the UI
curl -s localhost:8888/router/installs                               # state, step, log tail
```
Background job, one at a time: clone `repo`@`branch` into `dir`, build the image (+ `prebuild`), download the weights. Only missing parts run; re-run after a failure.
`requires` models get recipe + image first. Log: `state/logs/install-<model>.log`.
Not installed: `/v1/models` and `/router/status` say `"installed": false`; requests get 409 `not installed: POST /router/install`. No auto-install.

## Deploy

```bash
python3 test_thinking.py && docker compose up --build -d
```
Wait for `/router/status` `in_flight: 0` first. The router re-adopts the running model.
Offline: `FROM model-router-model-router:latest` + `COPY router.py thinking.py /app/`, then `docker compose up -d --no-build`.

## Thinking forwarding

`thinking.py` parses any form below into one intent; the engine adapter writes it. No thinking field = body forwarded unchanged.

| Form | Sent by |
|---|---|
| `chat_template_kwargs.enable_thinking` / `.thinking` / `.reasoning_effort` | vLLM/SGLang clients |
| `enable_thinking` | Qwen/DashScope |
| `thinking: {type, budget_tokens}`, `thinking: "high"` | Anthropic, Z.ai, DeepSeek style |
| `reasoning: {enabled, effort, max_tokens, exclude}` | OpenRouter, Together |
| `think: false` | Ollama |
| `reasoning_effort: none..max` (`off` = none) | OpenAI style |
| `thinking_budget`, `thinking_token_budget`, `max_thinking_tokens` | Native budgets |

- Explicit switch beats effort. `reasoning_effort: none` = off. Budget 0 = off. Off drops effort and budget.
- Effort → budget (`"effort": "budget"` models): minimal 512, low 2048, medium 8192, high 16384, xhigh 32768, max = none. Override: `"effort_budget"`.
- Log: `state/requests.jsonl` `think` field. Field names and small values only, never content.

### models.json keys

| Key | Values |
|---|---|
| `engine` | `vllm`, `tensorfold`, other = generic OpenAI |
| `effort` | `template` (template reads `reasoning_effort`) or `budget` (effort → token budget) |
| `thinking_budget` | `false` = never send a router-made budget |
| `repo`, `branch` | Recipe git repo, branch; cloned into `dir` by `/router/install` |
| `requires` | Models whose recipe + image install first (base image, sibling dir) |
| `weights_install` | Weights download for install; default `weights_update` |
| `prebuild` | Run after the image build (optional; failure is not fatal) |

### Engine dialects

| Engine | Writes |
|---|---|
| vllm | `chat_template_kwargs.enable_thinking`, `reasoning_effort`, `thinking_token_budget`, `include_reasoning: false`. 400 on budget → resend without, stop adding until restart |
| tensorfold | `chat_template_kwargs.enable_thinking`, `reasoning_effort` (template level names), `thinking_budget`. Penalties ignored |
| openai | `chat_template_kwargs.enable_thinking`, `reasoning_effort`. No budget |

## Adding an engine: forwarding contract

| Step | Rule |
|---|---|
| Adapter | Subclass `Adapter` in `thinking.py`, set `name` (= `engine`), register in `ADAPTERS` |
| `budget_field` | Native max-thinking-tokens field, or `None` |
| `efforts` | Accepted `reasoning_effort` values; others not added |
| Hooks | `set_thinking`, `set_effort`, `set_budget`, `set_hide`. Only ADD fields; never overwrite or remove client fields |
| Intent | `thinking` True/False/None (engine default), `effort`, `budget`, `hide` |
| Sampling | `SAMPLING_FIELDS` never touched. Different name → add mapped field, keep original |
| No budget support | Thinking on, not length-limited |
| Engine plumbing | `Engine` class in `router.py` (busy count, context length); `engine_of()` picks both |
| Test | Cases in `test_thinking.py`; live: `none` → no `reasoning_content`, `low` < `xhigh`, `budget_tokens: 200` cuts near 200, `temperature: 0` twice identical |
