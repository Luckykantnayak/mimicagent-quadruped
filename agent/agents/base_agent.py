# agents/base_agent.py
from langchain_core.messages import AIMessage
import json


def text_content(response):
    if isinstance(response.content, str):
        return response.content
    return "\n".join(b["text"] for b in response.content
                     if isinstance(b, dict) and b.get("type") == "text")

class BaseAgent:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, messages) -> AIMessage:
        response = self.llm.invoke(messages)
        attempts = []
        usage = {}
        for attempt in range(2):
            meta = response.response_metadata
            text = text_content(response)
            record = {"attempt": attempt + 1, "model": meta.get("model", getattr(self.llm, "model", None)),
                      "stop_reason": meta.get("stop_reason"), "usage": meta.get("usage", {}),
                      "text_characters": len(text)}
            attempts.append(record)
            print("[llm] " + json.dumps(record), flush=True)
            for key, value in record["usage"].items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
            if meta.get("stop_reason") != "max_tokens" and text.strip():
                # This pipeline has no tool-use conversation; only visible text
                # belongs in subsequent node prompts, never thinking signatures.
                return response.model_copy(update={"content": text, "response_metadata": {
                    **meta, "usage": usage, "api_attempts": attempts}})
            model = str(record["model"] or "")
            if attempt == 0 and meta.get("stop_reason") == "max_tokens" and model.startswith("claude-sonnet-5-5"):
                response = self.llm.invoke(messages, thinking={"type": "between_tools"},
                                           output_config={"effort": "high"})
                continue
            raise RuntimeError("Claude returned empty or truncated output; no further automatic retries. "
                               + json.dumps(attempts))
