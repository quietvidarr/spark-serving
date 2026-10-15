# model-router

OpenAI-compatible router on :8888 that runs one model at a time on 127.0.0.1:8001 and switches on demand.
Files: `router.py` (server), `thinking.py` (thinking/reasoning normalization), `models.json` (models, read at start),
`test_thinking.py` (unit tests), `ui.html` (test page).

Deploy after editing: `python3 test_thinking.py && docker compose up --build -d` (if the base image can't be pulled,
build offline: `FROM model-router-model-router:latest` + `COPY router.py thinking.py /app/`, then
`docker compose up -d --no-build`). Wait for `/router/status` `in_flight: 0` first. The router re-adopts the running model.

## Thinking / reasoning forwarding

Clients say "think / don't think / how much" in different dialects. For POST `.../completions` the router
(`thinking.py`) parses every known form into one canonical intent and lets the model's engine adapter write it in
the engine's own fields. If the request has no thinking field at all, the body is forwarded unchanged.

Forms it reads (strongest on/off switch first):

| Form | Sent by |
|---|---|
| `chat_template_kwargs.enable_thinking` / `.thinking` (bool or `{type}`) / `.reasoning_effort` | vLLM/SGLang clients, chat UIs with custom params |
| `enable_thinking` (top level) | Qwen/DashScope |
| `thinking: {type: enabled\|disabled\|adaptive, budget_tokens}`, `thinking: "high"` | Anthropic style, Z.ai / DeepSeek style |
| `reasoning: {enabled, effort, max_tokens, exclude}` | OpenRouter, Together |
| `think: false` | Ollama clients |
| `reasoning_effort: none\|minimal\|low\|medium\|high\|xhigh\|max` (`off` = none) | OpenAI style clients |
| `thinking_budget`, `thinking_token_budget`, `max_thinking_tokens` | TensorFold / vLLM / misc native budgets |

Precedence: an explicit switch beats what an effort implies (`reasoning_effort: none` = off, any other effort = on);
a budget of 0 = off. Thinking off drops effort and budget.

Effort -> budget (`thinking.EFFORT_BUDGET`, only for models with `"effort": "budget"`):
minimal 512, low 2048, medium 8192, high 16384, xhigh 32768, max = no limit. Override per model with
`"effort_budget": {"low": 1000, ...}`.

models.json per-model keys:
- `engine`: `vllm` | `tensorfold` | anything else = generic OpenAI adapter.
- `effort`: `template` (the chat template reads `reasoning_effort`, e.g. Qwen3.8: low/medium/xhigh) or
  `budget` (template has no effort knob, e.g. Qwen3.6: effort becomes a thinking-token budget).
- `thinking_budget: false`: never send a router-made budget to this model.

Engine dialects:
- **vllm**: `chat_template_kwargs.enable_thinking`, `reasoning_effort` (also handed to the template),
  `thinking_token_budget` (needs `--reasoning-parser`; refused by the V2 model runner), `include_reasoning: false` for
  `reasoning.exclude`. If vLLM answers 400 naming the budget field the router added, it resends without it and
  stops adding it for that model until the router restarts (logged as "refused thinking_token_budget").
- **tensorfold**: `chat_template_kwargs.enable_thinking` (or `.thinking`), `reasoning_effort` (mapped to the
  template's level names), `thinking_budget` (closes the think block after n tokens). No presence/frequency/
  repetition penalties (ignored by the engine).
- **openai** (generic): `chat_template_kwargs.enable_thinking` and `reasoning_effort`; no budget.

The request log (`state/requests.jsonl`) gets `"think": {"in": [...], "added": {...}}` for requests with thinking
fields: field names and small values only, never message content.

## Adding an engine: request-forwarding contract

A new engine (e.g. a custom-kernel engine like "DwarfStar DS4") must get temperature, thinking on/off and a
thinking budget right. To add one:

1. In `thinking.py`, subclass `Adapter`, set `name` (the models.json `engine` value) and register it in `ADAPTERS`:
   - `budget_field`: the engine's native max-thinking-tokens field, or `None` if it has none.
   - `efforts`: the `reasoning_effort` values it accepts (others are not added).
   - override `set_thinking(out, on)`, `set_effort(out, effort)`, `set_budget(out, n)`, `set_hide(out)` when its
     fields differ from the defaults (`chat_template_kwargs.enable_thinking`, top-level `reasoning_effort`).
   - Hooks only ADD fields; never overwrite a field the client set natively; never remove anything.
2. Canonical intent the adapter receives (`thinking.Intent`): `thinking` True/False/None (None = engine default),
   `effort` (minimal..max or None), `budget` (tokens or None; effort-derived when `"effort": "budget"`), `hide`.
3. Sampling fields (`thinking.SAMPLING_FIELDS`: temperature, top_p, top_k, min_p, presence/frequency/repetition
   penalty, seed, max_tokens, max_completion_tokens, stop, n, logit_bias) are never touched by the router; if the
   engine names one differently, map it in the adapter as an added field, keeping the original.
4. If the engine has no budget support, effort/budget requests still turn thinking on but are not length-limited
   (gap). Router-side enforcement would need stream rewriting plus a continuation request; not implemented.
5. Engine-side plumbing (busy count, context length) is the `Engine` class in `router.py`; `engine_of()` picks both.
6. Test: add cases to `test_thinking.py` (`python3 test_thinking.py`, no network), then live through :8888:
   `reasoning_effort: none` -> no `reasoning_content`; `low` vs `xhigh` -> shorter vs longer reasoning;
   `thinking: {type: enabled, budget_tokens: 200}` -> reasoning cut near 200 tokens; `temperature: 0` twice ->
   identical output. Check `state/requests.jsonl` `think.added`.
