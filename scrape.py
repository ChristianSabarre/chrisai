# file: scrape_valorant_patch_notes_checkpoint.py
import asyncio
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from playwright.async_api import async_playwright

BASE = "https://playvalorant.com"
ARCHIVE_PATH = "/{locale}/news/tags/patch-notes/"
LOCALE = "en-us"
ARCHIVE_URL = f"{BASE}{ARCHIVE_PATH.format(locale=LOCALE)}"

CHECKPOINT = Path("valorant_patch_notes_checkpoint.jsonl")
OUT_CSV = Path("valorant_patch_notes.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ValorantPatchScraper/1.2)"
}

def extract_patch_number(title: str) -> Optional[str]:
    m = re.search(r'(\b(?:v)?\d+(?:\.\d+){0,2}\b)', title, flags=re.IGNORECASE)
    return m.group(1) if m else None

def parse_article(url: str) -> Dict:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Title
    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""

    # Published date
    date_text = ""
    time_el = soup.find("time")
    if time_el and time_el.get("datetime"):
        date_text = time_el["datetime"]
    elif time_el:
        date_text = time_el.get_text(strip=True)
    else:
        meta_date = soup.find("meta", {"property": "article:published_time"})
        if meta_date and meta_date.get("content"):
            date_text = meta_date["content"]

    # Article body
    paragraphs = []
    body = (
        soup.select_one("div[itemprop='articleBody']") or
        soup.select_one("div.nexus-article") or
        soup.select_one("article")
    )
    if body:
        for p in body.find_all(["p", "li"]):
            text = p.get_text(" ", strip=True)
            if text:
                paragraphs.append(text)
    else:
        for p in soup.find_all("p"):
            text = p.get_text(" ", strip=True)
            if text:
                paragraphs.append(text)

    content = "\n".join(paragraphs).strip()

    return {
        "title": title,
        "url": url,
        "published": date_text,
        "patch_number": extract_patch_number(title) or "",
        "content": content
    }

async def collect_archive_links() -> List[str]:
    links = set()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=HEADERS["User-Agent"])
        page = await context.new_page()
        await page.goto(ARCHIVE_URL, wait_until="domcontentloaded")

        # Keep clicking "Show More" until it's gone
        while True:
            try:
                button = page.locator("button:has-text('Show More')")
                if await button.is_visible():
                    await button.click()
                    await page.wait_for_timeout(1200)  # wait for load
                else:
                    break
            except Exception:
                break

        # Collect all article links
        anchors = await page.locator("a[href*='/news/game-updates/']").all()
        for a in anchors:
            href = await a.get_attribute("href")
            if href:
                if href.startswith("/"):
                    links.add(BASE + href)
                elif href.startswith(BASE):
                    links.add(href)

        await browser.close()

    print(f"Raw links found: {len(links)}")
    patch_links = [l for l in links if "patch-notes" in l.lower()]
    print(f"Patch note links: {len(patch_links)}")

    return sorted(set(patch_links))

def load_checkpoint() -> Dict[str, Dict]:
    scraped = {}
    if CHECKPOINT.exists():
        with CHECKPOINT.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    scraped[obj["url"]] = obj
                except Exception:
                    continue
    return scraped

def append_checkpoint(item: Dict):
    with CHECKPOINT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    print(f"[1/3] Collecting patch links from: {ARCHIVE_URL}")
    loop = asyncio.get_event_loop()
    all_links = loop.run_until_complete(collect_archive_links())

    print(f"Found {len(all_links)} patch links")

    scraped = load_checkpoint()
    print(f"[checkpoint] Already have {len(scraped)} scraped patches")

    results = list(scraped.values())
    seen = set(scraped.keys())

    print("[2/3] Fetching new patch pages…")
    for url in tqdm(sorted(set(all_links))):
        if url in seen:
            continue
        try:
            item = parse_article(url)
            if item["title"] and item["content"]:
                append_checkpoint(item)
                results.append(item)
            seen.add(url)
            time.sleep(0.6)
        except Exception as e:
            print(f"Error parsing {url}: {e}")

    print(f"[3/3] Saving {len(results)} patch notes to CSV…")
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"Done.\n- Checkpoint JSONL: {CHECKPOINT.resolve()}\n- CSV:   {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
