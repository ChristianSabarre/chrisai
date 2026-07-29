"""Chunking and retrieval over the patch-note corpus.

Two problems drove this module. Whole patch notes were indexed as single
documents, so a query naming a version could not find it: the version occupies
a handful of characters in a document thousands of characters long, and it
barely registers in the embedding. And a section buried in a long note competed
with the whole rest of that note for similarity.

So notes are split on their section headings, every chunk is stamped with the
patch it came from, and a query that names a version is answered by an exact
metadata lookup rather than a similarity search.
"""

import re
from typing import Any, Dict, List, Optional, Sequence

# Sections are usually well under this; the cap only splits unusually long ones.
CHUNK_MAX_CHARS = 4000
# Below this a section is merged into the next one rather than embedded alone.
CHUNK_MIN_CHARS = 200
# Ceiling on assembled context, to bound prompt size on big patches.
CONTEXT_MAX_CHARS = 24000

HEADING_PREFIX = "## "
VERSION_IN_QUERY = re.compile(r"\b(\d{1,2})\s*\.\s*(\d{1,2})\b")
LATEST_IN_QUERY = re.compile(
    r"\b(latest|newest|most recent|current|this patch|last patch|just changed)\b",
    re.IGNORECASE,
)


def _split_sections(content: str) -> List[Dict[str, str]]:
    """Break a note into (heading, body) sections on '## ' marker lines.

    Riot nests a platform qualifier under each section, so the markup runs
    "AGENT UPDATES" then "ALL PLATFORMS" then the actual text. A heading with no
    body of its own is therefore a section label for what follows, and gets
    carried down as a parent rather than dropped -- otherwise every chunk ends
    up labelled "ALL PLATFORMS" and the real section name is lost.
    """
    raw: List[tuple] = []
    heading = ""
    buffer: List[str] = []

    def flush():
        raw.append((heading, "\n".join(buffer).strip()))

    for line in content.split("\n"):
        if line.startswith(HEADING_PREFIX):
            flush()
            heading = line[len(HEADING_PREFIX):].strip()
            buffer = []
        else:
            buffer.append(line)
    flush()

    sections: List[Dict[str, str]] = []
    parent = ""
    for head, body in raw:
        if not body:
            parent = head
            continue
        if parent and head and head != parent:
            label = f"{parent} - {head}"
        else:
            label = head or parent
        sections.append({"heading": label, "body": body})
    return sections


def _split_long_body(body: str) -> List[str]:
    """Split an oversized section on line boundaries, never mid-line."""
    if len(body) <= CHUNK_MAX_CHARS:
        return [body]

    parts, current, size = [], [], 0
    for line in body.split("\n"):
        if current and size + len(line) > CHUNK_MAX_CHARS:
            parts.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        parts.append("\n".join(current))
    return parts


def chunk_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split one patch note into embeddable chunks.

    Every chunk opens with the patch title, date and section name. That header
    is what makes a version number searchable: it appears in the embedded text
    of each chunk instead of once in a document of several thousand characters.
    """
    content = record.get("final_content") or ""
    if not content.strip():
        return []

    title = record.get("title") or "Unknown Patch"
    published = record.get("published") or ""
    patch_key = record.get("patch_key") or ""

    sections = _split_sections(content) or [{"heading": "", "body": content}]

    # Fold tiny sections into the following one so headings do not become chunks
    # on their own.
    merged: List[Dict[str, str]] = []
    for section in sections:
        if merged and len(section["body"]) < CHUNK_MIN_CHARS:
            merged[-1]["body"] += "\n" + (
                f"{section['heading']}\n{section['body']}"
                if section["heading"] else section["body"]
            )
        else:
            merged.append(dict(section))

    chunks: List[Dict[str, Any]] = []
    for section in merged:
        for part in _split_long_body(section["body"]):
            label = f"{title} ({published})"
            if section["heading"]:
                label += f" - {section['heading']}"
            chunks.append({
                "text": f"{label}\n{part}",
                "metadata": {
                    "source": "valorant_patch_notes",
                    "title": title,
                    "published": published,
                    "section": section["heading"] or "GENERAL",
                    # Chroma rejects None; the April Fools notes have no version.
                    **({"patch_key": patch_key} if patch_key else {}),
                    **({"patch": record["patch"]}
                       if isinstance(record.get("patch"), (int, float)) else {}),
                },
            })

    for i, chunk in enumerate(chunks):
        chunk["metadata"]["chunk_index"] = i
        chunk["metadata"]["chunk_count"] = len(chunks)
    return chunks


def build_chunks(dataset: Sequence[Dict[str, Any]]):
    """Chunk the whole corpus into parallel documents/metadatas/ids lists."""
    documents, metadatas, ids = [], [], []
    for record_index, record in enumerate(dataset):
        for chunk in chunk_record(record):
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
            ids.append(f"p{record_index}_c{chunk['metadata']['chunk_index']}")
    return documents, metadatas, ids


def resolve_patch_key(query: str, known_keys: Sequence[str]) -> Optional[str]:
    """Find which known patch a query names, if any.

    Matches the literal string first, then by numeric value, so "patch 10.0"
    still resolves to the note published as 10.00.
    """
    match = VERSION_IN_QUERY.search(query)
    if not match:
        return None

    candidate = f"{int(match.group(1))}.{match.group(2)}"
    if candidate in known_keys:
        return candidate

    try:
        wanted = float(candidate)
    except ValueError:
        return None

    for key in known_keys:
        try:
            if float(key) == wanted:
                return key
        except ValueError:
            continue
    return None


def _rows_from_get(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    return [{"document": d, "metadata": m} for d, m in zip(docs, metas)]


def _rows_from_query(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    rows = []
    for i, doc in enumerate(docs):
        rows.append({
            "document": doc,
            "metadata": metas[i] if i < len(metas) else {},
            "distance": dists[i] if i < len(dists) else None,
        })
    return rows


def search(collection, query: str, known_keys: Sequence[str],
           latest_key: Optional[str] = None, k: int = 6) -> Dict[str, Any]:
    """Retrieve chunks for a query.

    A query naming a version, or asking for the newest patch, is answered by an
    exact metadata lookup. Everything else falls back to similarity search.
    """
    mode = "patch_lookup"
    patch_key = resolve_patch_key(query, known_keys)

    if not patch_key and latest_key and LATEST_IN_QUERY.search(query):
        patch_key, mode = latest_key, "latest_lookup"

    if patch_key:
        result = collection.get(where={"patch_key": patch_key},
                                include=["documents", "metadatas"])
        rows = _rows_from_get(result)
        if rows:
            rows.sort(key=lambda r: r["metadata"].get("chunk_index", 0))
            return {"rows": rows, "mode": mode, "patch_key": patch_key}

    rows = _rows_from_query(collection.query(query_texts=[query], n_results=k))
    return {"rows": rows, "mode": "semantic", "patch_key": None}


def format_context(rows: Sequence[Dict[str, Any]]) -> str:
    """Assemble retrieved chunks into the context block sent to the model."""
    parts, total = [], 0
    for row in rows:
        text = row["document"]
        if total + len(text) > CONTEXT_MAX_CHARS:
            break
        parts.append(text)
        total += len(text)
    return "\n\n---\n\n".join(parts) if parts else "No relevant patch notes found."


def cited_sources(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Distinct patches behind a set of retrieved chunks, newest first."""
    seen, sources = set(), []
    for row in rows:
        meta = row.get("metadata") or {}
        title = meta.get("title")
        if title and title not in seen:
            seen.add(title)
            sources.append({"title": title, "published": meta.get("published", "")})
    return sorted(sources, key=lambda s: s["published"], reverse=True)
