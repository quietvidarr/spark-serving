"""Thinking / reasoning request normalization for the model router.

Clients express "think or don't, and how much" in many dialects. parse() reads every known form into one
canonical Intent; render() hands that intent to a per-engine adapter that writes the engine's own fields.

Rules (see README.md, "Adding an engine: request-forwarding contract"):
  * nothing is ever removed from the request: unknown and foreign fields pass through untouched;
  * a field the client already set in the engine's native form is never overwritten;
  * sampling fields (temperature, top_p, top_k, min_p, penalties, seed, max_tokens, stop, ...) are not touched;
  * no thinking field at all in the request -> the body is forwarded byte-for-byte (engine defaults apply).

Pure python, no I/O: tests in test_thinking.py.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
EFFORT_ALIASES = {"off": "none", "disabled": "none", "ultra": "max", "extreme": "max"}

# Effort -> thinking-token budget, used only for models whose chat template has no effort knob
# (models.json "effort": "budget"), e.g. Qwen3.6. None = no limit. Documented in README.md.
EFFORT_BUDGET = {"minimal": 512, "low": 2048, "medium": 8192, "high": 16384, "xhigh": 32768, "max": None}

# Sampling fields the router promises never to touch (documentation + tests).
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
                   "repetition_penalty", "seed", "max_tokens", "max_completion_tokens", "stop", "n", "logit_bias")


@dataclass
class Intent:
    thinking: bool | None = None      # True on, False off, None: client didn't say (engine default)
    effort: str | None = None         # one of EFFORTS (never "none": that is thinking=False)
    budget: int | None = None         # max thinking tokens the client asked for
    hide: bool = False                # OpenRouter reasoning.exclude: don't return the reasoning text
    sources: list = field(default_factory=list)   # which request fields said what (for the request log)

    @property
    def empty(self):
        return self.thinking is None and self.effort is None and self.budget is None and not self.hide


def _effort(v):
    if not isinstance(v, str):
        return None
    v = v.strip().lower()
    v = EFFORT_ALIASES.get(v, v)
    return v if v in EFFORTS else None


def _int(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


def _switch(v):
    """bool, {"type": "enabled"|"disabled"|"adaptive"}, or "enabled"/"disabled" -> bool | None."""
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        v = v.get("type")
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("enabled", "enable", "on", "true", "adaptive", "auto"):
            return True
        if s in ("disabled", "disable", "off", "false", "none"):
            return False
    return None


def parse(body: dict) -> Intent:
    """Every known client form -> Intent. Explicit on/off switches beat what an effort or budget implies."""
    it = Intent()
    if not isinstance(body, dict):
        return it
    ctk = body.get("chat_template_kwargs") if isinstance(body.get("chat_template_kwargs"), dict) else {}
    reasoning = body.get("reasoning") if isinstance(body.get("reasoning"), dict) else {}
    thinking = body.get("thinking")
    switches, efforts, budgets = [], [], []

    # explicit switches, strongest first
    if "enable_thinking" in ctk:                                   # vLLM / SGLang / TensorFold
        switches.append(("chat_template_kwargs.enable_thinking", _switch(ctk["enable_thinking"])))
    if "thinking" in ctk:                                          # DeepSeek-style template kwarg
        switches.append(("chat_template_kwargs.thinking", _switch(ctk["thinking"])))
    if "enable_thinking" in body:                                  # Qwen/DashScope API
        switches.append(("enable_thinking", _switch(body["enable_thinking"])))
    if thinking is not None:
        if isinstance(thinking, str) and _effort(thinking) and _switch(thinking) is None:
            efforts.append(("thinking", _effort(thinking)))        # string form: thinking: "high"
        else:                                                      # Anthropic / Z.ai / DeepSeek / Kimi
            switches.append(("thinking", _switch(thinking)))
        if isinstance(thinking, dict):
            budgets.append(("thinking.budget_tokens", _int(thinking.get("budget_tokens"))))
    if "enabled" in reasoning:                                     # OpenRouter / Together
        switches.append(("reasoning.enabled", _switch(reasoning["enabled"])))
    if "think" in body:                                            # Ollama
        switches.append(("think", _switch(body["think"]) if not isinstance(body["think"], str)
                         else (_effort(body["think"]) != "none" if _effort(body["think"]) else None)))
        if isinstance(body["think"], str) and _effort(body["think"]) not in (None, "none"):
            efforts.append(("think", _effort(body["think"])))

    # efforts
    if "reasoning_effort" in body:                                 # OpenAI style
        efforts.append(("reasoning_effort", _effort(body["reasoning_effort"])))
    if "reasoning_effort" in ctk:
        efforts.append(("chat_template_kwargs.reasoning_effort", _effort(ctk["reasoning_effort"])))
    if "effort" in reasoning:                                      # OpenRouter
        efforts.append(("reasoning.effort", _effort(reasoning["effort"])))

    # budgets
    for key in ("thinking_budget", "thinking_token_budget", "max_thinking_tokens"):
        if key in body:
            budgets.append((key, _int(body[key])))
    if "max_tokens" in reasoning:                                  # OpenRouter
        budgets.append(("reasoning.max_tokens", _int(reasoning["max_tokens"])))

    it.sources = [(k, v) for k, v in (*switches, *efforts, *budgets) if v is not None]
    sw = next((v for _, v in switches if v is not None), None)
    eff = next((v for _, v in efforts if v is not None), None)
    bud = next((v for _, v in budgets if v is not None), None)

    if eff == "none":
        eff, implied = None, False
    elif eff is not None:
        implied = True
    elif bud == 0:
        implied = False
    elif bud is not None:
        implied = True
    else:
        implied = None
    it.thinking = sw if sw is not None else implied
    if it.thinking is False:
        eff, bud = None, None
    it.effort, it.budget = eff, (bud or None)
    it.hide = reasoning.get("exclude") is True
    return it


def budget_for(intent: Intent, model_cfg: dict) -> int | None:
    """The thinking budget to send: the client's own, else (models.json "effort": "budget") the effort's."""
    if intent.thinking is False:
        return None
    if intent.budget is not None:
        return intent.budget
    if intent.effort and model_cfg.get("effort") == "budget":
        table = {**EFFORT_BUDGET, **(model_cfg.get("effort_budget") or {})}
        return table.get(intent.effort)
    return None


class Adapter:
    """Renders an Intent into one engine's request dialect. Subclasses override the three hooks.

    A hook must only ADD a field (setdefault-style): the client's own native field always wins."""
    name = "openai"
    budget_field = None          # native thinking-budget field, or None if the engine has none
    efforts = EFFORTS            # reasoning_effort values the engine accepts

    def render(self, body: dict, intent: Intent, model_cfg: dict | None = None) -> dict:
        model_cfg = model_cfg or {}
        if intent.empty:
            return body
        out = copy.copy(body)
        if intent.thinking is not None:
            self.set_thinking(out, intent.thinking)
        if intent.thinking is not False and intent.effort and model_cfg.get("effort") != "budget":
            self.set_effort(out, intent.effort)
        b = budget_for(intent, model_cfg)
        if b is not None and model_cfg.get("thinking_budget", True) is not False:
            self.set_budget(out, b)
        if intent.hide:
            self.set_hide(out)
        return out

    # hooks -------------------------------------------------------------------------------------------------
    def set_thinking(self, out, on):
        ctk = dict(out.get("chat_template_kwargs") or {}) if isinstance(out.get("chat_template_kwargs"), dict) else {}
        if "enable_thinking" not in ctk:
            ctk["enable_thinking"] = on
            out["chat_template_kwargs"] = ctk

    def set_effort(self, out, effort):
        if "reasoning_effort" not in out and effort in self.efforts:
            out["reasoning_effort"] = effort

    def set_budget(self, out, n):
        if self.budget_field and self.budget_field not in out:
            out[self.budget_field] = n

    def set_hide(self, out):
        pass


class VllmAdapter(Adapter):
    """vLLM >= 0.2x: chat_template_kwargs.enable_thinking, reasoning_effort (passed to the template),
    thinking_token_budget (needs --reasoning-parser; refused by the V2 model runner -> router retries without)."""
    name = "vllm"
    budget_field = "thinking_token_budget"

    def set_hide(self, out):
        out.setdefault("include_reasoning", False)


class TensorFoldAdapter(Adapter):
    """TensorFold: chat_template_kwargs.enable_thinking (or .thinking), reasoning_effort (none..max, mapped to the
    template's names), thinking_budget (closes the think block after n tokens)."""
    name = "tensorfold"
    budget_field = "thinking_budget"


ADAPTERS = {a.name: a() for a in (Adapter, VllmAdapter, TensorFoldAdapter)}


def adapter_for(engine: str) -> Adapter:
    return ADAPTERS.get(engine, ADAPTERS["openai"])


def normalize(body: dict, engine: str, model_cfg: dict | None = None):
    """-> (body to forward, intent). The body is the same object when nothing needed adding."""
    intent = parse(body)
    return adapter_for(engine).render(body, intent, model_cfg), intent


def strip_budget(body: dict, engine: str) -> dict:
    """The body without the budget field the router itself added (engine refused it)."""
    f = adapter_for(engine).budget_field
    return {k: v for k, v in body.items() if k != f}


def added_fields(before: dict, after: dict) -> dict:
    """What render() added (for the request log)."""
    added = {k: v for k, v in after.items() if k not in before}
    ctk_b = before.get("chat_template_kwargs") if isinstance(before.get("chat_template_kwargs"), dict) else {}
    ctk_a = after.get("chat_template_kwargs") if isinstance(after.get("chat_template_kwargs"), dict) else {}
    for k, v in ctk_a.items():
        if k not in ctk_b:
            added[f"chat_template_kwargs.{k}"] = v
    added.pop("chat_template_kwargs", None)
    return added
