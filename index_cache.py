"""Persisted embedding vectors, so starting up does not re-embed the corpus.

Embedding the corpus costs roughly a minute and about sixty API calls, and the
result is identical every time the patch notes have not changed. On a host that
wipes its disk when the service idles, that cost is paid by whoever happens to
load the page first.

The vectors are written once, keyed by a fingerprint of the corpus and the
chunking parameters, and reloaded on every later start. Build the cache during
deployment with build_index.py; the file is not committed, since regenerating
it on each refresh would add several megabytes to history every time.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import retrieval

logger = logging.getLogger(__name__)

CACHE_PATH = Path(os.environ.get("EMBEDDING_CACHE", "embeddings.npz"))


def corpus_fingerprint(dataset: Sequence[Dict[str, Any]]) -> str:
    """Identity of the indexed corpus, so a stale cache is never reused.

    Covers the chunking parameters as well as the data: changing how notes are
    split invalidates vectors built under the old scheme.
    """
    h = hashlib.sha256()
    h.update(f"v2:{retrieval.CHUNK_MAX_CHARS}:{retrieval.CHUNK_MIN_CHARS}".encode())
    for record in dataset:
        h.update((record.get("title") or "").encode("utf-8"))
        h.update(str(len(record.get("final_content") or "")).encode())
    return h.hexdigest()[:16]


def load(fingerprint: str, expected_count: int) -> Optional[List[List[float]]]:
    """Return cached vectors, or None if absent or not matching this corpus."""
    if not CACHE_PATH.exists():
        return None

    try:
        with np.load(CACHE_PATH, allow_pickle=False) as data:
            cached_fingerprint = str(data["fingerprint"].item())
            vectors = data["vectors"]
    except Exception as e:
        logger.warning(f"Could not read {CACHE_PATH}: {e}")
        return None

    if cached_fingerprint != fingerprint:
        logger.info(f"Embedding cache is for a different corpus "
                    f"({cached_fingerprint} != {fingerprint}); ignoring it")
        return None

    if len(vectors) != expected_count:
        logger.warning(f"Embedding cache holds {len(vectors)} vectors, "
                       f"expected {expected_count}; ignoring it")
        return None

    return vectors.tolist()


def save(fingerprint: str, vectors: Sequence[Sequence[float]]) -> None:
    try:
        np.savez_compressed(
            CACHE_PATH,
            vectors=np.asarray(vectors, dtype=np.float32),
            fingerprint=np.array(fingerprint),
        )
        size_mb = CACHE_PATH.stat().st_size / 1e6
        logger.info(f"Wrote {len(vectors)} embeddings to {CACHE_PATH} ({size_mb:.1f} MB)")
    except Exception as e:
        # A cache that cannot be written is a slow start, not a broken app.
        logger.warning(f"Could not write {CACHE_PATH}: {e}")
