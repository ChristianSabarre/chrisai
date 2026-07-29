"""Report which patch notes a refresh brought in, against the committed copy.

Used by the scheduled workflow so its log says what actually arrived rather
than only that something changed.
"""

import json
import subprocess
import sys

CORPUS = "feedme_patchnotes.json"


def committed_corpus():
    result = subprocess.run(
        ["git", "show", f"HEAD:{CORPUS}"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def main() -> int:
    new = json.load(open(CORPUS, encoding="utf-8"))
    old = committed_corpus()

    if old is None:
        print(f"{len(new)} patches (no committed copy to compare against)")
        return 0

    seen = {p.get("title") for p in old}
    added = [p for p in new if p.get("title") not in seen]

    print(f"{len(old)} -> {len(new)} patches")
    for patch in sorted(added, key=lambda p: p.get("published", "")):
        print(f"  + {patch.get('published')}  {patch.get('title')}")

    if not added:
        print("  (no new patches; existing notes were re-scraped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
