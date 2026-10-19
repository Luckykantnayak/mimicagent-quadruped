import unittest
from langchain_core.messages import AIMessage
from agents.base_agent import BaseAgent


def response(text, stop="end_turn", tokens=10):
    return AIMessage(content=text, response_metadata={"model": "claude-sonnet-5-5",
                     "stop_reason": stop, "usage": {"input_tokens": 5, "output_tokens": tokens}})


class FakeLLM:
    model = "claude-sonnet-5-5"

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class APIOutputTests(unittest.TestCase):
    def test_success_unchanged_no_retry(self):
        llm = FakeLLM(response("CODE"))
        self.assertEqual(BaseAgent(llm).generate("prompt").content, "CODE")
        self.assertEqual(llm.calls, [{}])

    def test_empty_budget_fallback_and_usage(self):
        llm = FakeLLM(response([{"type": "thinking", "thinking": "", "signature": "secret"}], "max_tokens", 24000), response("CODE"))
        result = BaseAgent(llm).generate("prompt")
        self.assertEqual(result.content, "CODE")
        self.assertEqual(result.response_metadata["usage"]["output_tokens"], 24010)
        self.assertEqual(llm.calls[1], {"thinking": {"type": "between_tools"}, "output_config": {"effort": "high"}})

    def test_truncated_text_also_retried(self):
        llm = FakeLLM(response("class Broken(", "max_tokens"), response("CODE"))
        self.assertEqual(BaseAgent(llm).generate("prompt").content, "CODE")
        self.assertEqual(len(llm.calls), 2)

    def test_retry_bounded(self):
        llm = FakeLLM(response("", "max_tokens"), response("", "max_tokens"))
        with self.assertRaisesRegex(RuntimeError, "no further automatic retries"):
            BaseAgent(llm).generate("prompt")
        self.assertEqual(len(llm.calls), 2)

    def test_non_budget_empty_not_retried(self):
        llm = FakeLLM(response("", "refusal"))
        with self.assertRaises(RuntimeError):
            BaseAgent(llm).generate("prompt")
        self.assertEqual(len(llm.calls), 1)

    def test_thinking_not_forwarded(self):
        llm = FakeLLM(response([{"type": "thinking", "thinking": "private", "signature": "secret"}, {"type": "text", "text": "CODE"}]))
        self.assertEqual(BaseAgent(llm).generate("prompt").content, "CODE")
