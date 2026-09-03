"""Colour and emphasis for the terminal, and nothing when it is not one.

Every helper here returns plain text unless stdout is a terminal that has not
asked to be left alone. That is not politeness: `dyp ask ... > results.txt` and
`dyp ask ... | grep` are ordinary things to do, and escape codes in a file are
corruption. The check is made once at import against the real stream.

Honours `NO_COLOR` (any value disables) and `CLICOLOR_FORCE` (any value enables
even when piped), which are the two conventions worth following.
"""

from __future__ import annotations

import os
import re
import sys

_CODES = {
    "bold": "1", "dim": "2", "italic": "3", "underline": "4",
    "red": "31", "green": "32", "yellow": "33", "blue": "34",
    "magenta": "35", "cyan": "36", "grey": "90",
}


def _enabled() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("CLICOLOR_FORCE"):
        return True
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


COLOUR = _enabled()


def style(text: str, *names: str) -> str:
    """`text` wrapped in the named styles, or unchanged without a terminal."""
    if not COLOUR or not names or not text:
        return text
    codes = ";".join(_CODES[n] for n in names if n in _CODES)
    return f"\033[{codes}m{text}\033[0m" if codes else text


def bold(text: str) -> str:
    return style(text, "bold")


def dim(text: str) -> str:
    return style(text, "grey")


# Words too common to be worth marking. Highlighting every "the" in a passage
# makes the passage harder to read, which is the opposite of the point.
STOPWORDS = frozenset("""
a an and are as at be but by can could do does for from had has have how i if
in is it its of on or that the their there these they this to was were what
when where which who why will with would you your
""".split())

# Three letters, because two-letter stems match inside longer words constantly
# and the marking then lands mid-word.
MIN_STEM = 3


def _stem(word: str) -> str:
    """A crude English stem, so `receptors` marks `receptor` and vice versa."""
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def terms(query: str) -> list[str]:
    """The words from `query` worth marking in a passage."""
    found = []
    for word in re.findall(r"\w+", query.lower()):
        if len(word) >= MIN_STEM and word not in STOPWORDS:
            found.append(_stem(word))
    return sorted(set(found), key=len, reverse=True)


def mark(text: str, wanted: list[str], *names: str) -> str:
    """Emphasise the query's words where they occur in `text`.

    Matches on a stem and at a word boundary, so `receptor` marks `receptors`
    but `on` does not light up half of `conduction`. One pass over the text with
    a single alternation, so overlapping stems cannot double-wrap and leave
    escape codes stranded inside each other.
    """
    if not COLOUR or not wanted or not text:
        return text
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in wanted) + r")(\w{0,3})\b",
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: style(m.group(0), *(names or ("bold", "yellow"))), text)


def rule(width: int = 60) -> str:
    return dim("─" * width)


# --- choosing which part of a passage to show ---------------------------------

_SENTENCE = re.compile(r"[^.!?]*[.!?]+[\s]*|[^.!?]+$")


def _sentences(text: str) -> list[str]:
    found = [m.group(0) for m in _SENTENCE.finditer(text) if m.group(0).strip()]
    return found or ([text] if text else [])


def best_extract(text: str, wanted: list[str], budget: int = 700) -> tuple[str, bool, bool]:
    """The stretch of `text` densest in the query's words, and whether it was cut.

    Returns `(extract, cut_before, cut_after)`.

    A passage is 3,500 bytes and a terminal shows a fraction of it, so *which*
    fraction matters. Showing the opening shows wherever the chunker happened to
    start; the answer is as likely to be in the middle.

    Sentences are scored by how many distinct query stems they contain, and the
    best-scoring run that fits the budget wins -- a sliding window over
    sentences, so the extract still begins and ends on one. No model, no index,
    one pass over a few thousand characters; the cost is not worth optimising
    and neither is the accuracy, because a reader who wants certainty has
    `--full` and the byte offset.

    With no query words present it returns the opening, which is the honest
    default: nothing in the passage is more relevant than anything else.
    """
    if not text:
        return "", False, False
    parts = _sentences(text)
    if not wanted or len(text) <= budget:
        return (text[:budget], False, len(text) > budget) if len(text) > budget \
            else (text, False, False)

    hits = []
    for part in parts:
        low = part.lower()
        hits.append(sum(1 for stem in wanted if stem in low))

    best_score, best_from, best_to = -1, 0, 1
    for first in range(len(parts)):
        size, score = 0, 0
        for last in range(first, len(parts)):
            size += len(parts[last])
            if size > budget and last > first:
                break
            score += hits[last]
            # Prefer the longer run on a tie: more context for the same density.
            if score > best_score or (score == best_score and size > 0
                                      and last - first > best_to - best_from):
                best_score, best_from, best_to = score, first, last + 1
    extract = "".join(parts[best_from:best_to]).strip()
    return extract, best_from > 0, best_to < len(parts)


# --- progress bars ------------------------------------------------------------

# Eighth-width blocks, so a bar moves on most updates rather than sitting still
# for a whole cell. At 24 cells that is 192 distinguishable positions, which is
# finer than the eye needs and finer than the numbers beside it.
_EIGHTHS = " ▏▎▍▌▋▊▉█"


def bar(done: float, total: float, width: int = 24) -> str:
    """A proportional bar, drawn to an eighth of a character.

    Returns plain block characters with no colour when styling is off, so a
    redirected log still shows the shape.
    """
    if total <= 0:
        return "─" * width
    share = max(0.0, min(1.0, done / total))
    eighths = round(share * width * 8)
    full, rest = divmod(eighths, 8)
    cells = "█" * full + (_EIGHTHS[rest] if rest and full < width else "")
    filled = cells.ljust(width, "·")
    if not COLOUR:
        return filled
    return style(filled[:len(cells)], "cyan") + dim(filled[len(cells):])


def elide(text: str, width: int, keep_tail: int = 12) -> str:
    """Shorten to `width`, cutting the middle rather than the end.

    Book titles in a real library differ at the end far more than in the
    middle -- `Springer.Neuroscience.2002.eBook` against
    `Springer.Neuroscience.2009.eBook` -- so trimming the tail hides exactly
    what distinguishes them, and two different books scroll past looking
    identical.
    """
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    tail = min(keep_tail, width - 2)
    head = width - tail - 1
    return f"{text[:head]}…{text[-tail:]}"
