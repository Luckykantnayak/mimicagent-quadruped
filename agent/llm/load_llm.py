import json

from langchain_openai import ChatOpenAI

def load_llm(model):
    llm = ChatOpenAI(
        model=model,
        base_url="http://127.0.0.1:1234/v1",
        api_key="not-needed",
        temperature=0.,
        max_tokens=8000
    )
    return llm


if __name__ == '__main__':
    llm = load_llm("gemma-3-4b-it")
