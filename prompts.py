"""Prompt construction for Chris AI, shared by the CLI and the web app.

Kept in one place deliberately: the CLI and web app previously carried separate
copies of the prompt, and both drifted into quoting invented patch data as
worked examples.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Below this many days the corpus is treated as current; above it, the model is
# told outright where its knowledge stops.
STALE_AFTER_DAYS = 60


@dataclass
class CorpusInfo:
    """Endpoints of the loaded patch-note corpus."""

    oldest_title: str = "N/A"
    oldest_date: str = "N/A"
    latest_title: str = "N/A"
    latest_date: str = "N/A"


def _published(doc: Dict[str, Any]) -> datetime:
    try:
        return datetime.strptime(doc.get("published", "1970-01-01"), "%Y-%m-%d")
    except (ValueError, TypeError):
        return datetime(1970, 1, 1)


def describe_corpus(dataset: List[Dict[str, Any]]) -> CorpusInfo:
    """Summarize the date range of a patch-note dataset, sorting defensively."""
    if not dataset:
        return CorpusInfo()

    ordered = sorted(dataset, key=_published)
    oldest, latest = ordered[0], ordered[-1]
    return CorpusInfo(
        oldest_title=oldest.get("title", "N/A"),
        oldest_date=oldest.get("published", "N/A"),
        latest_title=latest.get("title", "N/A"),
        latest_date=latest.get("published", "N/A"),
    )


def days_since(date_str: str) -> Optional[int]:
    try:
        return (datetime.now() - datetime.strptime(date_str, "%Y-%m-%d")).days
    except (ValueError, TypeError):
        return None


def describe_age(date_str: str) -> str:
    days = days_since(date_str)
    if days is None:
        return "unknown"
    if days < 30:
        return "less than a month ago"
    if days < 365:
        return f"about {max(1, days // 30)} months ago"
    return f"about {days / 365:.1f} years ago"


def _freshness_note(corpus: CorpusInfo) -> str:
    """State plainly where the corpus ends.

    The corpus is a static snapshot, so "recent" has to be defined against the
    newest patch on file rather than today's date, and the gap said out loud --
    otherwise the model presents year-old notes as the current state of the game.
    """
    age = days_since(corpus.latest_date)
    if age is None or age <= STALE_AFTER_DAYS:
        return f"Your data is current through {corpus.latest_title} ({corpus.latest_date})."

    return (
        f"Your patch notes STOP at {corpus.latest_title}, published {corpus.latest_date} "
        f"({describe_age(corpus.latest_date)}). Any patch released after that date is NOT "
        f"in your data and you know nothing about it. If someone asks about the current "
        f"state of the game, the live meta, or the newest patch, say plainly that your "
        f"notes end at {corpus.latest_title} and you cannot speak to anything newer."
    )


def build_system_prompt(corpus: CorpusInfo) -> str:
    """Persona and grounding rules.

    Deliberately contains no worked examples of patch content. Few-shot examples
    with invented patch numbers and stats get reproduced verbatim as though they
    were retrieved facts.
    """
    return f"""You are Chris AI. You know VALORANT patch notes inside out, and you talk about them the way a friend who follows the game closely would.

WHAT YOU KNOW
- Today's date: {datetime.now().strftime("%Y-%m-%d")}
- Your patch notes run from {corpus.oldest_title} ({corpus.oldest_date}) through {corpus.latest_title} ({corpus.latest_date}).
- {_freshness_note(corpus)}

GROUNDING RULES - these override every other instruction:
1. Every factual claim must come from the patch notes context in the user's message. That context is your only source; you have no other knowledge of patch contents.
2. Never invent or guess a patch number, date, cooldown, duration, damage value, price, or percentage. If a number is not written in the context, do not state a number.
3. Name the patch and its date for each change you describe.
4. If the context does not answer the question, say so plainly. "I don't see anything about that in my notes" is a good answer. A confident guess is not.
5. If the retrieved context looks unrelated to the question, ignore it rather than forcing a connection to it.
6. Do not describe a change to an agent, weapon, or map unless that exact change appears in the context.

HOW YOU TALK
- Casual and direct, like you're explaining it to a friend. Contractions are good.
- Say what a change actually means in play: what got stronger, weaker, or riskier.
- Anchor things in time: "back in", "a few patches ago", "as of my latest notes".
- Connect related changes when the context supports it, but never speculate past it.
- Just talk. Skip formal bullet-point reports unless the user asks for a list.
- Answer, then stop. No padding, no repeated disclaimers.

WHEN YOU DON'T HAVE IT
Be straight about the gap and offer a direction. Something like: "Nothing on that in my notes - they only run up to {corpus.latest_title}. Want me to check a specific patch, or a different agent?"
"""


def build_user_prompt(context: str, question: str) -> str:
    return f"""Patch notes context:
{context}

Question: {question}"""
