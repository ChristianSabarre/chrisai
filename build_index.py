"""Precompute embeddings for the patch corpus.

Run during deployment so the running service starts from a cached index instead
of embedding 900-odd chunks on the first request:

    python build_index.py

Writes embeddings.npz next to the corpus. Safe to re-run; it exits immediately
when the existing cache already matches the corpus.
"""

import json
import logging
import sys
from pathlib import Path

import index_cache
import retrieval
from mistral_client import MistralEmbedding

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build_index")

PATCH_NOTES_FILE = Path("feedme_patchnotes.json")


def main() -> int:
    if not PATCH_NOTES_FILE.exists():
        logger.error(f"{PATCH_NOTES_FILE} not found. Run scrape.py first.")
        return 1

    dataset = json.loads(PATCH_NOTES_FILE.read_text(encoding="utf-8"))
    documents, _, _ = retrieval.build_chunks(dataset)
    fingerprint = index_cache.corpus_fingerprint(dataset)

    logger.info(f"{len(dataset)} patches -> {len(documents)} chunks ({fingerprint})")

    if index_cache.load(fingerprint, len(documents)) is not None:
        logger.info("Cache already matches this corpus; nothing to do.")
        return 0

    logger.info("Embedding via Mistral; this takes about a minute...")
    vectors = MistralEmbedding()(documents)
    index_cache.save(fingerprint, vectors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
