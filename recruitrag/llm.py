"""Access to the local LLM served by Ollama.

Set RECRUIT_RAG_MODEL to switch models, e.g.  export RECRUIT_RAG_MODEL=llama3.2:3b
"""
import os

DEFAULT_MODEL = "gemma3:1b"


def model_name() -> str:
    return os.getenv("RECRUIT_RAG_MODEL", DEFAULT_MODEL)


def ask_llm(system: str, user: str, temperature: float = 0.1) -> str:
    """Send one system+user exchange to the local model and return its text.

    Raises RuntimeError with an actionable message if Ollama or the model is unavailable,
    so callers (CLI, web app) can show it instead of a stack trace."""
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("The 'ollama' Python package is missing: pip install ollama") from exc

    model = model_name()
    try:
        reply = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": temperature},
        )
    except Exception as exc:
        raise RuntimeError(
            f"No answer from Ollama model '{model}'. Is the Ollama app running, and have you run "
            f"'ollama pull {model}'? ({exc})"
        ) from exc
    return reply["message"]["content"]
