import json
import re


_ROLE_PREFIX = re.compile(r"^\s*(AI|Assistant|Human|System)\s*:\s*", re.IGNORECASE)


def _candidates(text: str):
    """
    Yield progressively more forgiving slices of an LLM message that might be JSON.

    Responses arrive in several shapes: a bare object, a ```json fence, an untagged
    ``` fence, or an object wrapped in prose ("Here is the result: {...}"). Handling
    only the first two is what made this parser fail intermittently and abort whole
    skill runs.
    """
    yield text                                            # 1. as given

    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        yield m.group(1)                                  # 2. ```json fence

    m = re.search(r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        yield m.group(1)                                  # 3. untagged ``` fence

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        yield text[start:end + 1]                         # 4. object buried in prose

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        yield text[start:end + 1]                         # 5. top-level array


def extract_json_from_message(text: str):
    """
    Extract JSON from an LLM message.

    Tolerates a role prefix ("AI: "), ```json or untagged ``` fences, prose around
    the object, Python-style True/False/None, and raw control characters inside
    string values. Every one of those has been observed in practice, and each used
    to kill the run with a JSONDecodeError.
    """
    if not text or not text.strip():
        raise ValueError("Cannot extract JSON: the model returned an empty message.")

    text = _ROLE_PREFIX.sub("", text)

    last_error = None
    for candidate in _candidates(text):
        candidate = candidate.strip()
        if not candidate:
            continue
        # strict=False permits literal newlines/tabs inside strings, which the model
        # emits when a description carries its own formatting.
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError as e:
            last_error = e
        # Python-style literals leak in when a prompt shows them in its schema.
        try:
            repaired = re.sub(r"\bTrue\b", "true", candidate)
            repaired = re.sub(r"\bFalse\b", "false", repaired)
            repaired = re.sub(r"\bNone\b", "null", repaired)
            return json.loads(repaired, strict=False)
        except json.JSONDecodeError as e:
            last_error = e

    raise json.JSONDecodeError(
        f"No parseable JSON in model message "
        f"({last_error.msg if last_error else 'no candidates'}). "
        f"First 200 chars: {text[:200]!r}",
        text, 0,
    )


def extract_python_code_from_message(text: str):
    """
    Extract python code from an LLM message that may contain:
    - a role prefix ('AI:')
    - ```python ... ``` fences, or an untagged ``` fence
    """
    if not text:
        return ""

    text = _ROLE_PREFIX.sub("", text)

    for pattern in (r"```python\s*(.*?)\s*```", r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()

    return text.strip()
