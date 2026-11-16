"""Local LLM call via Ollama. Model is configurable: RECRUIT_RAG_MODEL=llama3.2:3b"""
import os

DEFAULT_MODEL = "gemma3:1b"


def invoke_ai(system_message: str, user_message: str) -> str:
    try:
        import ollama
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install Ollama's Python client: pip install ollama") from exc

    response = ollama.chat(
        model=os.getenv("RECRUIT_RAG_MODEL", DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        options={"temperature": 0.1},
    )
    return response["message"]["content"]
