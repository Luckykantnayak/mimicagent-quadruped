import os

from langchain_anthropic import ChatAnthropic

from prompts.prompt_loader import load_yaml_config

def load_llm():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export your Anthropic API key first:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-..."
        )

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg", "config.yaml")
    llm_cfg = load_yaml_config(cfg_path)["llm"]

    return ChatAnthropic(
        model_name=llm_cfg["model"],
        temperature=llm_cfg.get("temperature", 1.0),
        max_tokens_to_sample=llm_cfg.get("max_tokens", 20000),
        api_key=api_key,
        # langchain-anthropic defaults extended thinking ON for some models (e.g.
        # claude-sonnet-5), which returns message content as a LIST of blocks and
        # spends the token budget on thinking — the pipeline's extractors expect a
        # plain-string completion (as Opus returns), so disable it.
        thinking={"type": "disabled"},
        # Ride out transient API errors (429/5xx/529) with the SDK's backoff.
        max_retries=8,
        timeout=120.0,
    )


if __name__ == "__main__":
    llm = load_llm()
    resp = llm.invoke("Say 'OK' and nothing else.")
    print(resp)
    print(resp.response_metadata['usage']['input_tokens'])

    if hasattr(resp, "raw"):
        usage = resp.raw.get("usage")
        if usage:
            print("Prompt tokens:", usage.get("prompt_tokens"))
            print("Completion tokens:", usage.get("completion_tokens"))
            print("Total tokens:", usage.get("total_tokens"))
        else:
            print("No usage info found in raw response")