"""Step 04: measure retrieval before tuning it.

This exists before any quality work, and the reason is specific: a prior tool
had 1,150 passing tests that could not see a regression halving answer recall,
because every document in its test corpus was shorter than one chunk. Tests that
cannot see quality are worse than no tests, because they are trusted.

What is measured here is whether the returned *passage* contains the answer --
not whether the right book appeared. Document-level hit rates are blind to chunk
selection, which is the thing most likely to be wrong.
"""

from __future__ import annotations

import json
import re
import statistics
import sqlite3
from math import comb
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dyprys import search
from dyprys.db import locate
from dyprys.text import read_span

@dataclass(frozen=True)
class Question:
    text: str
    answer: str          # a term the returned passage must contain
    kind: str = "lookup"  # lookup | descriptive | mechanism | oblique


@dataclass
class Tally:
    answerable: int = 0
    hit_at_1: int = 0
    hit_at_5: int = 0


@dataclass
class EvalReport:
    asked: int = 0
    answerable: int = 0          # the term exists somewhere in the embedded corpus
    unanswerable: list[str] = field(default_factory=list)
    hit_at_1: int = 0
    hit_at_5: int = 0
    lexical_hits: int = 0
    lexical_asked: int = 0
    routed_ok: int = 0      # a book holding the answer survived stage 1
    routed_asked: int = 0
    # Routing recall on its own says almost nothing, and said it convincingly for
    # a long time. On 117 books every answer term in the set sat in a median of
    # 65 of them, so hitting one of those with 3 books picked at *random* is
    # 83.9% before stage 1 does any work -- and the reported figure was 109/110.
    # These three make that visible instead of leaving it to be rediscovered.
    routed_chance: float = 0.0     # the same measure, for books picked at random
    kept_flat_top: int = 0         # routed set holds the book flat search chose
    kept_flat_asked: int = 0
    answer_books: list[int] = field(default_factory=list)  # fan-out per question
    # A *share* of the corpus is only comparable between runs on the same corpus.
    # Adding 1,369 books of unrelated material took the reported scanned fraction
    # from 6.6% to 5.55% while the chunks actually read went from 3,535 to 3,835:
    # the denominator grew, the work did not shrink. Both numbers are reported.
    scanned: float = 1.0
    scanned_chunks: float = 0.0
    seconds: float = 0.0
    misses: list[tuple[str, str]] = field(default_factory=list)
    # Answers not ranked *first*, as distinct from not returned at all. recall@1
    # is the headline number, and without this the harness could not support a
    # paired test on its own headline: comparing two arms meant comparing their
    # recall@5 misses and hoping recall@1 moved the same way. It does not --
    # expansion was measured +9 at recall@1 and +2 at recall@5.
    misses_at_1: list[tuple[str, str]] = field(default_factory=list)
    by_kind: dict[str, Tally] = field(default_factory=dict)
    # Recall split by how much vocabulary a question shares with the passage that
    # answers it. This is the difficulty measure that does not depend on anyone's
    # opinion of their own questions.
    by_overlap: dict[str, Tally] = field(default_factory=dict)

    @property
    def redundancy(self) -> int:
        """Median number of books holding the answer -- how easy routing has it.

        A question whose answer is in one book tests book selection. One whose
        answer is in sixty tests almost nothing: lose the best book and another
        will do. A set with a high median cannot measure a router.
        """
        if not self.answer_books:
            return 0
        return int(statistics.median(self.answer_books))

    @property
    def recall_at_1(self) -> float:
        return self.hit_at_1 / self.answerable if self.answerable else 0.0

    @property
    def recall_at_5(self) -> float:
        return self.hit_at_5 / self.answerable if self.answerable else 0.0


def load_questions(path: str | Path) -> list[Question]:
    """Objects `{q, a, kind}`, or the older `[question, answer]` pairs."""
    questions = []
    for entry in json.loads(Path(path).read_text()):
        if isinstance(entry, dict):
            questions.append(Question(entry["q"], entry["a"], entry.get("kind", "lookup")))
        else:
            questions.append(Question(entry[0], entry[1]))
    return questions


_TERM_PATTERNS: dict[str, re.Pattern] = {}


def contains_term(text: str, term: str) -> bool:
    """Does `text` contain `term` as a word, rather than inside a longer one?

    A plain `term in text` counts "Saxon" as containing "axon", "produce" as
    containing "rod", "ganglia" as containing "glia" and "biographer" as
    containing "raphe". On this corpus -- a thousand Gutenberg novels beside the
    neuroscience -- that was **23% of all book-hits**, and it inflated the metric
    in the one direction a metric must never be inflated: a passage was scored as
    holding the answer when it held an unrelated word.

    The boundary is on the left only, because the question sets deliberately use
    prefixes as answers -- "electroencephalog" is meant to match
    "electroencephalography", and "compensat" to match "compensation" and
    "compensatory".
    """
    # Substring first: it is a fast C-level scan and a strict superset of what
    # the pattern can match, so the regex only runs on the few chunks that could
    # possibly match. Without this, locating 110 terms across 284,627 chunks
    # went from seconds to over ten minutes.
    if term not in text:
        return False
    pattern = _TERM_PATTERNS.get(term)
    if pattern is None:
        pattern = _TERM_PATTERNS[term] = re.compile(r"\b" + re.escape(term))
    return pattern.search(text) is not None


def overlap_with(question: str, passage: str) -> float:
    """Share of a question's content words that appear in the passage.

    A question whose own wording is scattered through the answering passage is
    easy for keyword search and says little about retrieval; one that shares
    almost nothing is the case worth measuring. Stopwords are excluded because
    every passage contains them.
    """
    words = {w for w in re.findall(r"\w+", question.lower()) if w not in _STOPWORDS and len(w) > 2}
    if not words:
        return 0.0
    lowered = passage.lower()
    return sum(1 for w in words if _stem(w) in lowered) / len(words)


def _stem(word: str) -> str:
    """Crudely strip a plural or gerund ending.

    Without this, "forms" would not match "forming" and a question sharing the
    passage's vocabulary would be scored as though it shared none -- which would
    flatter the set by classing easy questions as hard, the exact error this
    measure exists to prevent.
    """
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


_STOPWORDS = {
    "the", "what", "which", "how", "why", "does", "did", "are", "and", "for", "that",
    "this", "with", "from", "into", "when", "where", "who", "was", "were", "has", "have",
    "its", "his", "her", "their", "them", "they", "you", "not", "but", "can", "cannot",
    "some", "someone", "something", "one", "two", "call", "called", "make", "makes",
    "made", "say", "said", "people", "mean", "means", "after", "before", "than", "then",
    "there", "here", "about", "over", "under", "through", "between", "part", "parts",
}


def chunk_texts(conn: sqlite3.Connection, model_id: int):
    """Every embedded chunk, as `(chunk_id, book_id, text)`.

    Only embedded chunks: a term sitting in a book that has not been embedded is
    not retrievable, so counting it as ground truth would flatter the score.
    """
    for start, end in search.embedded_ranges(conn, model_id):
        # Streamed, not fetchall(). A fully embedded library merges to a single
        # range, so materialising one meant holding a row per chunk in the
        # corpus -- 14M of them at 5,000 books. That is the failure this project
        # exists to avoid, and it had already been fixed once in lexical_safety.
        # Nothing in the loop touches the database, so the cursor stays open
        # safely.
        for row in conn.execute(
            "SELECT c.id, c.byte_offset, c.byte_length, c.content_hash, "
            "       src.path, src.book_id "
            "FROM chunks c "
            "JOIN segments seg ON c.id >= seg.chunk_start "
            "                 AND c.id < seg.chunk_start + seg.chunk_count "
            "JOIN sources src ON src.id = seg.source_id "
            "WHERE c.id >= ? AND c.id < ? ORDER BY c.id",
            (start, end),
        ):
            text = read_span(
                row["path"], row["byte_offset"], row["byte_length"], row["content_hash"]
            )
            if text is not None:
                yield row["id"], row["book_id"], text


@dataclass
class Located:
    """Where a term actually appears, so a claim about it can be checked."""

    chunks: set[int] = field(default_factory=set)
    books: set[int] = field(default_factory=set)

    def __bool__(self) -> bool:
        return bool(self.chunks)


def locate_terms(
    conn: sqlite3.Connection, model_id: int, terms: list[str]
) -> dict[str, Located]:
    """Which chunks, and which books, contain each term.

    One pass over the embedded corpus. The book set is what makes routing recall
    measurable: a routing failure and a stage-2 failure need different fixes, and
    without this they look identical.
    """
    wanted = [t.lower() for t in terms]
    found: dict[str, Located] = {t: Located() for t in wanted}
    for chunk_id, book_id, text in chunk_texts(conn, model_id):
        lowered = text.lower()
        for term in wanted:
            if contains_term(lowered, term):
                found[term].chunks.add(chunk_id)
                found[term].books.add(book_id)
    return found


def evaluate(
    conn: sqlite3.Connection,
    store,
    embedder,
    model_id: int,
    questions: list[Question],
    k: int = 5,
    search_fn: Callable | None = None,
    scanned: float = 1.0,
    route_fn: Callable | None = None,
    flat_fn: Callable | None = None,
) -> EvalReport:
    """Answer recall for one search implementation.

    `search_fn(question_text, k) -> list[Hit]` so routing, fusion and reranking
    can each be measured against exactly this, on exactly these questions.

    The callable takes the *text*, not a vector: BM25 needs the words, vector
    search needs the embedding, and a hybrid needs both. Handing over the query
    itself is the only contract all three can share.

    `flat_fn(question_text) -> list[Hit]` is the same search with routing off.
    Given it, the report says how often the routed set still held the book flat
    search picked first -- which is the question routing recall was supposed to
    answer and does not, once the answer is in most of the library.
    """
    if search_fn is None:
        def search_fn(question_text, k):
            return search.flat_search(
                conn, store, embedder.embed_query(question_text), model_id, k
            )

    report = EvalReport(asked=len(questions), scanned=scanned)
    where = locate_terms(conn, model_id, [q.answer for q in questions])
    corpus_books = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    chances: list[float] = []

    # Only the searches being measured are timed. With routing on, this loop
    # *also* runs an unrouted search per question, to report whether stage 1
    # kept the book flat search chose -- a diagnostic, not part of answering.
    # Counting it made the harness report routing as slower than flat (695 ms
    # against 593) when routing is in fact 7.5x faster end to end (58 ms against
    # 433). A latency number that includes the instrument is worse than none.
    searching = 0.0
    for question in questions:
        found_in = where[question.answer.lower()]
        if not found_in:
            report.unanswerable.append(question.answer)
            continue
        report.answerable += 1

        # Difficulty, measured against a passage that really does answer it.
        sample = _chunk_text(conn, min(found_in.chunks))
        share = overlap_with(question.text, sample) if sample else 0.0
        band = "shares wording" if share >= 0.5 else "shares little"

        for tally in (report.by_kind.setdefault(question.kind, Tally()),
                      report.by_overlap.setdefault(band, Tally())):
            tally.answerable += 1

        report.answer_books.append(len(found_in.books))

        if route_fn is not None:
            report.routed_asked += 1
            routed = route_fn(question.text)
            if routed & found_in.books:
                report.routed_ok += 1
            chances.append(_chance(corpus_books, len(found_in.books), len(routed)))

            if flat_fn is not None:
                unrouted = flat_fn(question.text)
                if unrouted:
                    best = locate(conn, unrouted[0].chunk_id)
                    if best is not None:
                        report.kept_flat_asked += 1
                        report.kept_flat_top += best["book_id"] in routed

        began = time.monotonic()
        hits = search_fn(question.text, k)
        searching += time.monotonic() - began
        passages = search.resolve(conn, hits)
        contains = [
            p.text is not None and contains_term(p.text.lower(), question.answer.lower())
            for p in passages
        ]
        if contains and contains[0]:
            report.hit_at_1 += 1
            report.by_kind[question.kind].hit_at_1 += 1
            report.by_overlap[band].hit_at_1 += 1
        else:
            report.misses_at_1.append((question.text, question.answer))
        if any(contains):
            report.hit_at_5 += 1
            report.by_kind[question.kind].hit_at_5 += 1
            report.by_overlap[band].hit_at_5 += 1
        else:
            report.misses.append((question.text, question.answer))
    report.seconds = searching
    if chances:
        report.routed_chance = sum(chances) / len(chances)
    return report


def _chance(books: int, holding: int, breadth: int) -> float:
    """Routing recall for `breadth` books picked at random, not by a router.

    The baseline the measure needs beside it: with the answer in `holding` of
    `books`, missing every one of them takes all `breadth` picks from the other
    `books - holding`, so the recall is one minus that.
    """
    missing = books - holding
    if breadth <= 0 or missing < breadth:
        return 1.0
    return 1.0 - comb(missing, breadth) / comb(books, breadth)


def lexical_safety(
    conn: sqlite3.Connection,
    store,
    embedder,
    model_id: int,
    samples: int = 20,
    k: int = 5,
    search_fn: Callable | None = None,
    seed: int = 0,
) -> tuple[int, int]:
    """Does an exact phrase still find the passage it was taken from?

    This is the check that catches a semantic-ranking change quietly breaking
    keyword search -- a regression no answer-recall number would show, because
    the answers would still be found by other routes.

    Only the sampled chunks are ever read. Choosing them by ordinal rather than
    by listing the corpus keeps this O(samples), not O(library): reading every
    passage to pick twenty of them is precisely the mistake this project exists
    to avoid, and it would have been invisible until the corpus was large.
    """
    import random

    if search_fn is None:
        def search_fn(question_text, k):
            return search.flat_search(
                conn, store, embedder.embed_query(question_text), model_id, k
            )

    ranges = search.embedded_ranges(conn, model_id)
    total = sum(end - start for start, end in ranges)
    if total == 0:
        return 0, 0

    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(total), min(samples, total)))

    hits = asked = 0
    for chunk_id in (_nth_embedded(ranges, n) for n in chosen):
        text = _chunk_text(conn, chunk_id)
        if text is None:
            continue
        phrase = _distinctive_phrase(text, rng)
        if phrase is None:
            continue
        asked += 1
        found = search_fn(phrase, k)
        if any(h.chunk_id == chunk_id for h in found):
            hits += 1
    return hits, asked


def _nth_embedded(ranges: list[tuple[int, int]], n: int) -> int:
    """The n-th embedded chunk id, counting across ranges in order."""
    for start, end in ranges:
        span = end - start
        if n < span:
            return start + n
        n -= span
    raise IndexError(n)


def _chunk_text(conn: sqlite3.Connection, chunk_id: int) -> str | None:
    """One chunk, by one seek."""
    row = conn.execute(
        "SELECT c.byte_offset, c.byte_length, c.content_hash, src.path FROM chunks c "
        "JOIN segments seg ON c.id >= seg.chunk_start "
        "                 AND c.id < seg.chunk_start + seg.chunk_count "
        "JOIN sources src ON src.id = seg.source_id WHERE c.id = ?",
        (chunk_id,),
    ).fetchone()
    if row is None:
        return None
    return read_span(row["path"], row["byte_offset"], row["byte_length"], row["content_hash"])


def _distinctive_phrase(text: str, rng, words: int = 9) -> str | None:
    """A verbatim run of words from the middle of the passage.

    Words are taken consecutively and unfiltered, because the probe has to be a
    phrase that genuinely occurs. Dropping short words first and joining what
    remained produced probes that were *not* in the text -- only 7 of 40 were --
    so the phrase query could not match and the measure was quietly falling
    through to the looser word-set search. It still measured something real, but
    not the thing it is named after.
    """
    tokens = text.split()
    if len(tokens) < words * 2:
        return None
    start = rng.randrange(len(tokens) // 4, max(len(tokens) // 4 + 1, len(tokens) - words))
    return " ".join(tokens[start : start + words])
