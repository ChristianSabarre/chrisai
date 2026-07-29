"""Mistral-backed embedding and chat helpers shared by app.py and web_app.py."""

import logging
import os
import time
from typing import Dict, List, Optional

from mistralai.client import Mistral

logger = logging.getLogger(__name__)

EMBED_MODEL = os.environ.get("MISTRAL_EMBED_MODEL", "mistral-embed")
CHAT_MODEL = os.environ.get("MISTRAL_CHAT_MODEL", "mistral-small-latest")
CHAT_TEMPERATURE = float(os.environ.get("MISTRAL_TEMPERATURE", "0.3"))

# mistral-embed accepts 8192 tokens per input. Batch requests stay well under the
# per-request total so a single oversized patch note can't fail the whole batch.
MAX_BATCH_ITEMS = 16
MAX_BATCH_CHARS = 20000
MAX_ITEM_CHARS = 28000

_client: Optional[Mistral] = None


def get_client() -> Mistral:
    global _client
    if _client is None:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY is not set. Create a key at https://console.mistral.ai "
                "and export it before starting the app."
            )
        _client = Mistral(api_key=api_key)
    return _client


def _with_retries(fn, *, attempts: int = 4, what: str = "Mistral call"):
    """Retry on rate limits / transient errors with exponential backoff."""
    delay = 1.0
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts:
                raise
            logger.warning("%s failed (attempt %d/%d): %s", what, attempt, attempts, e)
            time.sleep(delay)
            delay *= 2


def _batch(texts: List[str]) -> List[List[str]]:
    batches: List[List[str]] = []
    current: List[str] = []
    current_chars = 0
    for text in texts:
        if len(current) >= MAX_BATCH_ITEMS or (
            current and current_chars + len(text) > MAX_BATCH_CHARS
        ):
            batches.append(current)
            current, current_chars = [], 0
        current.append(text)
        current_chars += len(text)
    if current:
        batches.append(current)
    return batches


class MistralEmbedding:
    """ChromaDB-compatible embedding function backed by mistral-embed.

    Unlike the previous Gemini implementation this batches requests and raises on
    failure rather than inserting zero vectors, which would silently poison the index.
    """

    def __call__(self, input: List[str]) -> List[List[float]]:
        cleaned = [(t or "").strip()[:MAX_ITEM_CHARS] or " " for t in input]
        embeddings: List[List[float]] = []

        for batch in _batch(cleaned):
            response = _with_retries(
                lambda b=batch: get_client().embeddings.create(
                    model=EMBED_MODEL, inputs=b
                ),
                what=f"embed batch of {len(batch)}",
            )
            for item in sorted(response.data, key=lambda d: d.index):
                embeddings.append(item.embedding)

        if len(embeddings) != len(cleaned):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(cleaned)}, got {len(embeddings)}"
            )
        return embeddings

    # Chroma calls these on query/ingest paths. mistral-embed is symmetric, so both
    # delegate to __call__ -- there is no document/query task_type to distinguish.
    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def name(self) -> str:
        return "mistral-embedding"


def _extract_text(content) -> str:
    """Assistant content is a string or a list of content chunks depending on model."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            text = getattr(chunk, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)
    return str(content)


def chat(
    system_prompt: str,
    user_prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Send a grounded RAG turn to Mistral and return the reply text."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    response = _with_retries(
        lambda: get_client().chat.complete(
            model=CHAT_MODEL,
            messages=messages,
            temperature=CHAT_TEMPERATURE,
        ),
        what="chat completion",
    )
    return _extract_text(response.choices[0].message.content).strip()
