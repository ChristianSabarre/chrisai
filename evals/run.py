"""Evaluate retrieval and answer quality against a fixed set of questions.

Two passes, because they cost very differently:

    python evals/run.py                 # retrieval only, no chat calls
    python evals/run.py --answers       # also generate and grade answers

Retrieval grading is exact: each case names the patches that must come back, and
those expectations were read off the corpus rather than guessed. Answer grading
is deliberately lenient -- it checks that a required term is present, that a
question with no answer in the corpus is refused, and that no phrase from the
old fabricated prompt examples has reappeared.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import retrieval  # noqa: E402
from prompts import build_system_prompt, build_user_prompt, describe_corpus  # noqa: E402

CASES = Path(__file__).parent / "cases.json"

# Phrases that only ever came from invented few-shot examples. Their return
# would mean the prompt has regressed.
FABRICATIONS = ["18s", "14s", "tailwind cooldown was reduced", "leer duration"]

REFUSAL_MARKERS = [
    "don't see", "do not see", "nothing in my notes", "not in my notes",
    "only run up to", "only go up to", "no notes", "nothing about",
    "don't have", "do not have", "isn't in", "is not in", "nothing on that",
]


def load_app():
    """Import the app, which builds or reuses the index at import time."""
    import web_app
    if web_app.collection is None:
        raise SystemExit("index unavailable; check MISTRAL_API_KEY")
    return web_app


def patches_in(rows):
    keys = []
    for row in rows:
        key = (row.get("metadata") or {}).get("patch_key")
        if key and key not in keys:
            keys.append(key)
    return keys


def grade_retrieval(case, result):
    found = patches_in(result["rows"])
    problems = []

    expected_mode = case.get("expect_mode")
    if expected_mode and result["mode"] != expected_mode:
        problems.append(f"mode={result['mode']} expected {expected_mode}")

    if case.get("expect_any_patch") and not found:
        problems.append("no patches retrieved")

    wanted = case.get("expect_patches")
    if wanted and not any(w in found for w in wanted):
        problems.append(f"none of {wanted} retrieved (got {found[:5]})")

    return problems, found


def grade_answer(case, answer):
    problems = []
    lower = answer.lower()

    for bad in FABRICATIONS:
        if bad in lower:
            problems.append(f"fabricated phrase {bad!r}")

    refused = any(m in lower for m in REFUSAL_MARKERS)
    if case.get("should_refuse") and not refused:
        problems.append("did not refuse a question outside the corpus")
    if not case.get("should_refuse") and refused and case.get("answer_should_mention"):
        problems.append("refused a question the corpus covers")

    for term in case.get("answer_should_mention", []):
        if term.lower() not in lower:
            problems.append(f"missing expected term {term!r}")

    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", action="store_true",
                        help="also generate answers and grade them (uses the chat API)")
    parser.add_argument("--filter", default="",
                        help="only run cases whose id or category contains this")
    args = parser.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if args.filter:
        cases = [c for c in cases
                 if args.filter in c["id"] or args.filter in c["category"]]

    app = load_app()
    keys = app.patch_keys
    latest = (app.latest_patch or {}).get("patch_key")
    corpus = describe_corpus(app.dataset)

    print(f"{len(cases)} cases | {app.collection.count()} chunks | latest {latest}\n")

    failures = []
    by_category = {}

    for case in cases:
        # Cases carrying history exercise the follow-up rewrite, which costs an
        # extra call; cases without one skip it entirely.
        history = app.normalize_history(case.get("history"))
        search_query = app.condense_question(case["question"], history)

        if case.get("expect_condensed_contains"):
            wanted = case["expect_condensed_contains"].lower()
            if wanted not in search_query.lower():
                case.setdefault("_extra", []).append(
                    f"condensed to {search_query!r}, missing {wanted!r}")

        result = retrieval.search(app.collection, search_query,
                                  known_keys=keys, latest_key=latest, k=6)
        problems, found = grade_retrieval(case, result)
        problems += case.pop("_extra", [])

        answer = ""
        if args.answers:
            answer = app.mistral_chat(
                build_system_prompt(corpus),
                build_user_prompt(retrieval.format_context(result["rows"]),
                                  search_query),
            )
            problems += grade_answer(case, answer)

        ok = not problems
        stats = by_category.setdefault(case["category"], [0, 0])
        stats[1] += 1
        if ok:
            stats[0] += 1
        else:
            failures.append((case, problems, found, answer))

        print(f"  {'PASS' if ok else 'FAIL'}  {case['id']:<18} "
              f"{result['mode']:<14} {found[:3]}")

    print("\nby category:")
    for category, (passed, total) in sorted(by_category.items()):
        print(f"  {category:<15} {passed}/{total}")

    total_pass = sum(p for p, _ in by_category.values())
    total = sum(t for _, t in by_category.values())
    print(f"\nTOTAL {total_pass}/{total} passed")

    if failures:
        print("\nfailures:")
        for case, problems, found, answer in failures:
            print(f"\n  [{case['id']}] {case['question']}")
            for p in problems:
                print(f"      - {p}")
            if answer:
                print(f"      answer: {answer[:200]}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
