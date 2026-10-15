"""Unit tests for thinking.py (plain python, no network): python3 test_thinking.py"""

import copy
import unittest

from thinking import EFFORT_BUDGET, SAMPLING_FIELDS, added_fields, normalize, parse, strip_budget

MSG = [{"role": "user", "content": "hi"}]
SAMPLING = {"temperature": 0, "top_p": 0.9, "top_k": 20, "min_p": 0.05, "presence_penalty": 1.5,
            "frequency_penalty": 0.1, "repetition_penalty": 1.05, "seed": 7, "max_tokens": 300,
            "max_completion_tokens": 300, "stop": ["\n\n"]}
QWEN38 = {"effort": "template"}
QWEN36 = {"effort": "budget"}


def req(**kw):
    return {"model": "m", "messages": MSG, **kw}


class Parse(unittest.TestCase):
    def check(self, body, thinking, effort=None, budget=None):
        it = parse(body)
        self.assertEqual((it.thinking, it.effort, it.budget), (thinking, effort, budget), body)

    def test_nothing(self):
        self.check(req(temperature=0.6), None)
        self.assertTrue(parse(req()).empty)

    def test_openai_reasoning_effort(self):          # OpenAI style
        self.check(req(reasoning_effort="high"), True, "high")
        self.check(req(reasoning_effort="none"), False)
        self.check(req(reasoning_effort="off"), False)
        self.check(req(reasoning_effort="default"), None)   # server default

    def test_chat_template_kwargs(self):              # vLLM native
        self.check(req(chat_template_kwargs={"enable_thinking": False}), False)
        self.check(req(chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True}), True)
        self.check(req(chat_template_kwargs={"thinking": {"type": "disabled"}}), False)
        self.check(req(chat_template_kwargs={"reasoning_effort": "low"}), True, "low")

    def test_top_level_enable_thinking(self):         # Qwen/DashScope
        self.check(req(enable_thinking=False, reasoning_effort="low"), False)
        self.check(req(enable_thinking=True, reasoning_effort="low"), True, "low")

    def test_anthropic_thinking(self):
        self.check(req(thinking={"type": "enabled", "budget_tokens": 2048}), True, None, 2048)
        self.check(req(thinking={"type": "disabled"}), False)
        self.check(req(thinking={"type": "adaptive"}), True)

    def test_pi_zai_deepseek_string(self):
        self.check(req(thinking={"type": "enabled"}, reasoning_effort="medium"), True, "medium")
        self.check(req(thinking={"type": "disabled"}, reasoning_effort="medium"), False)
        self.check(req(thinking="high"), True, "high")

    def test_openrouter_reasoning(self):              # OpenRouter
        self.check(req(reasoning={"effort": "low"}), True, "low")
        self.check(req(reasoning={"enabled": False, "effort": "none"}), False)
        self.check(req(reasoning={"enabled": True, "effort": "medium"}), True, "medium")
        self.check(req(reasoning={"max_tokens": 1500}), True, None, 1500)
        self.assertTrue(parse(req(reasoning={"exclude": True})).hide)

    def test_native_budgets(self):
        self.check(req(thinking_budget=512), True, None, 512)
        self.check(req(thinking_token_budget=0), False)

    def test_ollama_think(self):                      # Ollama
        self.check(req(think=False), False)

    def test_switch_beats_effort(self):
        self.check(req(chat_template_kwargs={"enable_thinking": False}, reasoning_effort="high"), False)
        self.check(req(chat_template_kwargs={"enable_thinking": True}, reasoning_effort="none"), True)

    def test_garbage_is_ignored(self):
        self.check(req(reasoning_effort=5, thinking=[1], reasoning="x", chat_template_kwargs="y"), None)


class Render(unittest.TestCase):
    def test_untouched_when_silent(self):
        for eng in ("vllm", "tensorfold", "openai", "ds4"):
            b = req(**SAMPLING, foo={"bar": 1})
            out, _ = normalize(b, eng, QWEN38)
            self.assertIs(out, b)

    def test_sampling_and_unknown_pass_through(self):
        for eng in ("vllm", "tensorfold", "openai"):
            b = req(**SAMPLING, reasoning={"effort": "low"}, foo=1, thinking={"type": "enabled"})
            orig = copy.deepcopy(b)
            out, _ = normalize(b, eng, QWEN36)
            self.assertEqual(b, orig, "input not mutated")
            for k in (*SAMPLING, "reasoning", "foo", "thinking", "messages", "model"):
                self.assertEqual(out[k], orig[k], (eng, k))
            self.assertTrue(set(SAMPLING) <= set(SAMPLING_FIELDS) | {"max_completion_tokens"})

    def test_vllm_off(self):
        out, _ = normalize(req(reasoning={"enabled": False}), "vllm", QWEN38)
        self.assertEqual(out["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("reasoning_effort", out)
        self.assertNotIn("thinking_token_budget", out)

    def test_vllm_effort_template_model(self):
        out, _ = normalize(req(reasoning={"effort": "low"}), "vllm", QWEN38)
        self.assertEqual(out["reasoning_effort"], "low")
        self.assertTrue(out["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("thinking_token_budget", out)

    def test_vllm_effort_budget_model(self):
        out, _ = normalize(req(reasoning_effort="low"), "vllm", QWEN36)
        self.assertEqual(out["thinking_token_budget"], EFFORT_BUDGET["low"])
        self.assertEqual(out["reasoning_effort"], "low")          # client's own field stays
        out, _ = normalize(req(reasoning_effort="max"), "vllm", QWEN36)
        self.assertNotIn("thinking_token_budget", out)             # max = no limit

    def test_custom_budget_table(self):
        out, _ = normalize(req(reasoning_effort="low"), "tensorfold", {"effort": "budget", "effort_budget": {"low": 99}})
        self.assertEqual(out["thinking_budget"], 99)

    def test_tensorfold_dialect(self):
        out, _ = normalize(req(thinking={"type": "enabled", "budget_tokens": 2048}), "tensorfold", QWEN38)
        self.assertEqual(out["thinking_budget"], 2048)
        self.assertTrue(out["chat_template_kwargs"]["enable_thinking"])
        out, _ = normalize(req(enable_thinking=False), "tensorfold", QWEN38)
        self.assertFalse(out["chat_template_kwargs"]["enable_thinking"])
        out, _ = normalize(req(reasoning={"effort": "xhigh"}), "tensorfold", QWEN38)
        self.assertEqual(out["reasoning_effort"], "xhigh")

    def test_native_values_never_overridden(self):
        b = req(chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
                reasoning={"enabled": False}, reasoning_effort="low", thinking_budget=50)
        out, _ = normalize(b, "tensorfold", QWEN36)
        self.assertEqual(out["chat_template_kwargs"], {"enable_thinking": True, "preserve_thinking": True})
        self.assertEqual(out["reasoning_effort"], "low")
        self.assertEqual(out["thinking_budget"], 50)
        b = req(thinking_token_budget=10, reasoning_effort="high")
        out, _ = normalize(b, "vllm", QWEN36)
        self.assertEqual(out["thinking_token_budget"], 10)

    def test_other_kwargs_kept(self):
        out, _ = normalize(req(chat_template_kwargs={"preserve_thinking": True}, enable_thinking=False), "vllm", QWEN38)
        self.assertEqual(out["chat_template_kwargs"], {"preserve_thinking": True, "enable_thinking": False})

    def test_budget_disabled_and_strip(self):
        out, _ = normalize(req(reasoning={"max_tokens": 100}), "vllm", {"thinking_budget": False})
        self.assertNotIn("thinking_token_budget", out)
        out, _ = normalize(req(reasoning={"max_tokens": 100}), "vllm", {})
        self.assertNotIn("thinking_token_budget", strip_budget(out, "vllm"))

    def test_generic_engine(self):                    # unknown future engine: OpenAI/vLLM-style switches, no budget
        out, _ = normalize(req(thinking={"type": "enabled", "budget_tokens": 900}, reasoning_effort="high"), "ds4", {})
        self.assertTrue(out["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("thinking_budget", out)
        self.assertNotIn("thinking_token_budget", out)

    def test_hide(self):
        out, _ = normalize(req(reasoning={"exclude": True}), "vllm", {})
        self.assertFalse(out["include_reasoning"])

    def test_added_fields(self):
        b = req(reasoning_effort="none")
        out, _ = normalize(b, "vllm", {})
        self.assertEqual(added_fields(b, out), {"chat_template_kwargs.enable_thinking": False})


if __name__ == "__main__":
    unittest.main()
