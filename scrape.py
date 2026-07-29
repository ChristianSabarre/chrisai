"""Scrape VALORANT patch notes into the JSON file the app loads.

The archive page embeds its full result set in the Next.js __NEXT_DATA__ payload,
so the whole catalogue is reachable with plain HTTP requests. An earlier version
drove a headless browser to click through "Show More" pagination, which stopped
partway and left the corpus missing every patch before 5.01.

Usage:
    python scrape.py            # fetch anything missing, refresh the JSON
    python scrape.py --refresh  # re-fetch every article, ignoring the checkpoint
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

BASE = "https://playvalorant.com"
LOCALE = "en-us"
ARCHIVE_URL = f"{BASE}/{LOCALE}/news/tags/patch-notes/"

CHECKPOINT = Path("valorant_patch_notes_checkpoint.jsonl")
OUT_JSON = Path("feedme_patchnotes.json")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ValorantPatchScraper/2.0)"}
REQUEST_DELAY = 0.6

ARTICLE_PATH = re.compile(
    rf"/{LOCALE}/news/game-updates/[a-z0-9\-]*patch-notes[a-z0-9\-]*"
)
SLUG_VERSION = re.compile(r"patch-notes-(\d+)-(\d+)")
TITLE_VERSION = re.compile(r"(\d+\.\d+)")


def discover_patch_urls() -> List[str]:
    """Pull every patch-note article path out of the archive's Next.js payload."""
    html = requests.get(ARCHIVE_URL, headers=HEADERS, timeout=30).text

    blob = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not blob:
        raise RuntimeError(
            "No __NEXT_DATA__ block on the archive page. The site layout changed; "
            "the discovery step needs updating."
        )

    paths = set(ARTICLE_PATH.findall(blob.group(1)))
    paths |= set(ARTICLE_PATH.findall(html))
    return sorted(f"{BASE}{p.rstrip('/')}/" for p in paths)


def patch_key(url: str, title: str) -> Optional[str]:
    """Version string for an article, preferring the URL slug over the title.

    Kept as a string so the digits survive: 10.00 and 10.0 are different
    articles but collapse to the same float. Titles alone are not reliable
    either -- the 2023 April Fools article is titled "VALORANT Patch Notes
    2004", which would read as version 2004.
    """
    m = SLUG_VERSION.search(url)
    if m:
        return f"{int(m.group(1))}.{m.group(2)}"

    m = TITLE_VERSION.search(title)
    return m.group(1) if m else None


def patch_number(url: str, title: str) -> Optional[float]:
    key = patch_key(url, title)
    try:
        return float(key) if key else None
    except ValueError:
        return None


# Riot marks section names with h1, the same tag as the article title, and uses
# h2 for the platform qualifier under each section.
HEADING_TAGS = ("h1", "h2", "h3", "h4")
BLOCK_TAGS = HEADING_TAGS + ("p", "li")
FURNITURE = {"related articles", "related news", "recent news"}


def extract_blocks(body, title: str = "") -> str:
    """Flatten an article body to text, keeping headings and avoiding repeats.

    Headings are marked with a leading '## ' so the chunker can split on
    sections. Elements nested inside an already-emitted block are skipped: a
    <p> inside an <li> would otherwise contribute its text twice, once as part
    of the list item and once on its own, which had been doubling the size of
    the corpus.
    """
    emitted = set()
    lines = []

    for el in body.find_all(BLOCK_TAGS):
        if any(id(parent) in emitted for parent in el.parents):
            continue

        text = el.get_text(" ", strip=True)
        if not text:
            continue

        is_heading = el.name in HEADING_TAGS
        if is_heading and (text.strip().lower() in FURNITURE or text.strip() == title.strip()):
            continue

        emitted.add(id(el))
        lines.append(f"## {text}" if is_heading else text)

    return "\n".join(lines).strip()


def parse_article(url: str) -> Optional[Dict]:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""

    published = ""
    time_el = soup.find("time")
    if time_el and time_el.get("datetime"):
        published = time_el["datetime"]
    elif time_el:
        published = time_el.get_text(strip=True)
    else:
        meta = soup.find("meta", {"property": "article:published_time"})
        if meta and meta.get("content"):
            published = meta["content"]

    body = (
        soup.select_one("div[itemprop='articleBody']")
        or soup.select_one("div.nexus-article")
        or soup.select_one("article")
        or soup
    )
    content = extract_blocks(body, title)

    if not title or not content:
        return None

    return {
        "title": title,
        "url": url,
        "published": published,
        "patch_number": patch_number(url, title),
        "content": content,
    }


def normalize_date(raw: str) -> str:
    """Reduce an ISO timestamp to a plain date, which is what the app expects."""
    if not raw:
        return ""
    return raw[:10] if re.match(r"\d{4}-\d{2}-\d{2}", raw) else raw


def to_record(raw: Dict) -> Dict:
    """Convert a scraped article into the shape the app loads.

    The version is recomputed rather than read from the checkpoint, so entries
    written by the previous scraper (which stored it as a string, taken from the
    title) come out consistent with newly fetched ones.
    """
    url, title = raw.get("url", ""), raw.get("title", "")
    return {
        "title": title,
        "patch": patch_number(url, title),
        "patch_key": patch_key(url, title),
        "published": normalize_date(raw.get("published", "")),
        "final_content": raw["content"],
    }


def url_key(url: str) -> str:
    """Identity of an article, ignoring the trailing slash.

    The previous scraper stored URLs without a trailing slash and discovery now
    emits them with one, so keying on the raw string counted the same article
    twice and duplicated it in the corpus.
    """
    return url.rstrip("/")


def load_checkpoint() -> Dict[str, Dict]:
    scraped = {}
    if CHECKPOINT.exists():
        with CHECKPOINT.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("url"):
                    scraped[url_key(obj["url"])] = obj
    return scraped


def append_checkpoint(item: Dict) -> None:
    with CHECKPOINT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def rewrite_checkpoint(scraped: Dict[str, Dict]) -> None:
    """Collapse the append-only log back to one row per article."""
    with CHECKPOINT.open("w", encoding="utf-8") as f:
        for item in scraped.values():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch every article instead of resuming")
    args = parser.parse_args()

    print(f"Discovering patch notes from {ARCHIVE_URL}")
    urls = discover_patch_urls()
    print(f"  found {len(urls)} articles")

    scraped = {} if args.refresh else load_checkpoint()
    print(f"  {len(scraped)} already in checkpoint")

    todo = [u for u in urls if url_key(u) not in scraped]
    print(f"  {len(todo)} to fetch\n")

    failures = []
    for i, url in enumerate(todo, 1):
        try:
            item = parse_article(url)
            if item:
                append_checkpoint(item)
                scraped[url_key(url)] = item
                print(f"  [{i}/{len(todo)}] {item['title']}")
            else:
                failures.append((url, "no title or body"))
                print(f"  [{i}/{len(todo)}] SKIP (empty) {url}")
        except Exception as e:
            failures.append((url, str(e)))
            print(f"  [{i}/{len(todo)}] FAIL {url}: {e}")
        time.sleep(REQUEST_DELAY)

    rewrite_checkpoint(scraped)

    records = [to_record(v) for v in scraped.values()]
    records = [r for r in records if r["final_content"]]
    records.sort(key=lambda r: r["published"])

    OUT_JSON.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nWrote {len(records)} patch notes to {OUT_JSON}")
    if records:
        print(f"  range: {records[0]['published']} ({records[0]['title']}) "
              f"to {records[-1]['published']} ({records[-1]['title']})")
    if failures:
        print(f"\n{len(failures)} failures:")
        for url, err in failures[:10]:
            print(f"  {url}: {err}")
    return 1 if failures and not records else 0


if __name__ == "__main__":
    sys.exit(main())
