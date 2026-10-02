"""Chat model and embedding model factories.

API keys are read from the environment (a local .env file, or GitHub Codespaces secrets).
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def get_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=temperature)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))


def has_api_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))
