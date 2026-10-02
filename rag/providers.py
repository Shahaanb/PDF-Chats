"""Chat model and embedding model factories for the supported providers.

API keys are read from the environment (a local .env file, or GitHub Codespaces secrets).
Model names can be overridden with the environment variables listed in .env.example.
"""
from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# Gemini is listed first because it is the default: its free tier covers this project,
# while our OpenAI account has no API credits left. Set LLM_PROVIDER=openai to switch.
PROVIDERS = {
    "google": {"label": "Google Gemini", "key": "GOOGLE_API_KEY",
               "chat": ("GOOGLE_CHAT_MODEL", "gemini-3.5-flash-lite"),
               "embed": ("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")},
    "openai": {"label": "OpenAI", "key": "OPENAI_API_KEY",
               "chat": ("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
               "embed": ("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")},
}

LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def available_providers() -> list[str]:
    """Providers whose API key is set; LLM_PROVIDER (if set and available) comes first."""
    names = [name for name, p in PROVIDERS.items() if os.getenv(p["key"])]
    preferred = os.getenv("LLM_PROVIDER")
    if preferred in names:
        names.remove(preferred)
        names.insert(0, preferred)
    return names


def default_provider() -> str | None:
    names = available_providers()
    return names[0] if names else None


def chat_model_name(provider: str) -> str:
    var, default = PROVIDERS[provider]["chat"]
    return os.getenv(var, default)


def embedding_model_name(provider: str) -> str:
    var, default = PROVIDERS[provider]["embed"]
    return os.getenv(var, default)


def get_chat_model(provider: str, temperature: float = 0.0) -> BaseChatModel:
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=chat_model_name(provider), temperature=temperature)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=chat_model_name(provider), temperature=temperature)
    raise ValueError(f"Unknown provider: {provider}")


def embeddings_label(provider: str) -> str:
    if os.getenv("EMBEDDINGS", "local") == "local":
        return f"local {os.getenv('LOCAL_EMBEDDING_MODEL', LOCAL_EMBEDDING_MODEL)}"
    return embedding_model_name(provider)


def get_embeddings(provider: str) -> Embeddings:
    """Embedding model for indexing and retrieval.

    By default a small local model (FastEmbed, ONNX on the CPU) is used: Gemini's free tier
    allows only 100 embedding requests a minute, which a few papers already exceed, and local
    embeddings keep the documents on the machine. Set EMBEDDINGS=provider to use the chat
    provider's embedding API instead.
    """
    if os.getenv("EMBEDDINGS", "local") == "local":
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL),
                                   doc_embed_type="passage")
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=embedding_model_name(provider))
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=embedding_model_name(provider))
    raise ValueError(f"Unknown provider: {provider}")
