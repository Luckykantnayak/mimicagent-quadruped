import json
import re


def extract_json_from_message(text: str):
    """
    Extract JSON from an LLM message that may contain:
    - 'AI:' prefix
    - ```json ... ``` fences
    """
    # Remove leading "AI:" if present
    text = re.sub(r"^\s*AI:\s*", "", text)

    # Extract fenced JSON if present
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # Strip whitespace
    text = text.strip()

    # Validate / parse JSON
    return json.loads(text)


def extract_python_code_from_message(text: str):
    """
    Extract python code from an LLM message that may contain:
    - 'AI:' prefix
    - ```python ... ``` fences
    """
    # Remove leading "AI:" if present
    text = re.sub(r"^\s*AI:\s*", "", text)

    # Extract fenced python if present
    fence_match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # Strip whitespace
    text = text.strip()

    # Validate / parse JSON
    return text