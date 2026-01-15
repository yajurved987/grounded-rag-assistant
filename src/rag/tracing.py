import os
from langsmith import Client

from rag.settings import settings


def init_tracing() -> None:
    """
    Initialize LangSmith tracing if credentials are present.
    Safe across LangSmith versions.
    """

    if not settings.LANGCHAIN_API_KEY:
        print("⚠️ LangSmith API key not found. Tracing disabled.")
        return

    # Enable tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    # Project name should be set via env, NOT Client()
    if settings.LANGCHAIN_PROJECT:
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

    # Initialize client (no extra kwargs)
    Client(api_key=settings.LANGCHAIN_API_KEY)

    print("🟢 LangSmith tracing enabled")
    print(f"📁 Project: {settings.LANGCHAIN_PROJECT}")
