"""Step 03: exact search over the vectors.

Deliberately the simple version.  It scans every embedded chunk, which is what
step 05's routing must be checked against -- if routing and flat search disagree,
flat search is right by definition.  Results carry chunk ids and scores; text is
fetched only when a passage is actually shown.
"""

from __future__ import annotations

import fnmatch
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dyprys import db
from dyprys.text import EXACT, MISSING, Span, locate_span
from dyprys.vectors import VectorStore


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    score: float


@dataclass(frozen=True)
class Passage:
    """A hit resolved for display.

    `text` is None only when the passage could not be *proved* -- the file is
    unreadable, or it was edited in a way that leaves these bytes nowhere in it.
    `state` says which, and distinguishes a passage found intact at a new offset
    from one found where it was recorded. See `text.locate_span`.
    """

    chunk_id: int
    score: float
    title: str
    path: str
    chapter: int
    text: str | None
    state: str = EXACT
    # Where the text was actually found, which is not always where it was
    # recorded -- see `text.locate_span`. Carried so a citation can name a byte
    # in a file rather than only a chunk id.
    offset: int = 0


def scope_books(conn: sqlite3.Connection, pattern: str) -> set[int]:
    """Books matching `pattern`, by glob or substring, on title or path.

    One pattern covers the cases that matter: a shelf (`-c neuroscience/`, the
    directory books were added from), a single work (`-c Kandel`), or a related
    set (`-c "*Imaging*"`). Matching is case-insensitive.
    """
    needle = pattern.lower()
    globbing = any(c in pattern for c in "*?[")
    matched = set()
    for row in conn.execute("SELECT id, title, key FROM books"):
        title, key = row["title"].lower(), row["key"].lower()
        if globbing:
            if fnmatch.fnmatch(title, needle) or fnmatch.fnmatch(key, needle):
                matched.add(row["id"])
        elif needle in title or needle in key:
            matched.add(row["id"])
    return matched


def embedded_ranges(
    conn: sqlite3.Connection, model_id: int, book_ids: set[int] | None = None
) -> list[tuple[int, int]]:
    """Half-open chunk-id ranges holding a vector for this model.

    Only the embedded prefix of each segment, which is why searching a
    half-embedded library is exact rather than approximately right.  With
    `book_ids`, only those books -- which is the same restriction stage 2 of
    routing applies, reached from a user's flag instead of from centroids.
    """
    sql = (
        "SELECT seg.chunk_start, p.n_embedded FROM segment_progress p "
        "JOIN segments seg ON seg.id = p.segment_id "
        "JOIN sources src ON src.id = seg.source_id "
        "WHERE p.model_id = ? AND p.n_embedded > 0"
    )
    params: list = [model_id]
    if book_ids is not None:
        if not book_ids:
            return []
        sql += f" AND src.book_id IN ({','.join('?' * len(book_ids))})"
        params.extend(sorted(book_ids))
    return _merge(
        (row["chunk_start"], row["chunk_start"] + row["n_embedded"])
        for row in conn.execute(sql + " ORDER BY seg.chunk_start", params)
    )


def _merge(ranges) -> list[tuple[int, int]]:
    """Join ranges that touch, so neighbouring segments become one span.

    Chunk ids are dense and allocated in book order, so a fully embedded library
    collapses to a single range and only partly embedded segments create a
    boundary. This is not a tidiness measure. Unmerged, a 1,000-book library
    produced one range per segment, and BM25 built one SQL OR-clause per range:
    past roughly 1,000 segments SQLite refuses the query outright with
    "expression tree is too large". It also turns stage 2 into one large matrix
    multiply instead of thousands of small ones.
    """
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def flat_search(
    conn: sqlite3.Connection,
    store: VectorStore,
    query: np.ndarray,
    model_id: int,
    k: int = 5,
    book_ids: set[int] | None = None,
) -> list[Hit]:
    """Exact cosine over every embedded chunk, best first.

    Vectors are unit length at write time, so cosine is a dot product and the
    whole scan is one matrix multiply per contiguous range.
    """
    hits: list[Hit] = []
    for start, end in embedded_ranges(conn, model_id, book_ids):
        # score() decodes in blocks; slicing here would allocate a float32
        # copy of everything scanned -- 42 GB for a flat scan at 5,000 books.
        scores = store.score(start, end - start, query)
        # Only the top k of this range can matter to the top k overall.
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k] if len(scores) > k else range(len(scores))
        hits.extend(Hit(start + int(i), float(scores[i])) for i in top)
    hits.sort(key=lambda h: -h.score)
    return hits[:k]


def scanned_fraction(
    conn: sqlite3.Connection, model_id: int, book_ids: set[int] | None = None
) -> float:
    """Share of the embedded corpus a search touched -- the cost of any claim.

    Unscoped flat search is 1.0 by definition. Scoped, it is the same number
    routing will have to report, computed the same way, so the two are
    comparable without special-casing either.
    """
    everything = sum(e - s for s, e in embedded_ranges(conn, model_id))
    if not everything:
        return 0.0
    touched = sum(e - s for s, e in embedded_ranges(conn, model_id, book_ids))
    return touched / everything


def _source_shift(located: sqlite3.Row) -> int:
    """How far this source's bytes have moved since it was indexed.

    A single edit displaces everything after it by exactly the change in the
    file's size, and that size is recorded per source at ingest -- so this one
    number is the only extra candidate offset a lookup needs. Zero when the file
    is the size it was, which is the overwhelmingly common case and costs one
    `stat`.
    """
    try:
        return Path(located["path"]).stat().st_size - located["size_bytes"]
    except (OSError, IndexError, KeyError, TypeError):
        return 0


def _span_of(
    located: sqlite3.Row | None, row: sqlite3.Row | None, shifts: dict[int, int]
) -> Span:
    """A hit's text, looked for across whatever single edit the file has had.

    Shared by display and by duplicate suppression so the two can never hold
    different opinions about the same passage. They did: suppression read the
    recorded offset strictly, so on an edited book every passage came back
    unreadable and suppression quietly switched itself off -- for results that
    display was showing in full.

    `shifts` caches one `stat` per source across a result list.
    """
    if located is None or row is None:
        return Span(None, MISSING, 0)
    source = located["source_id"]
    if source not in shifts:
        shifts[source] = _source_shift(located)
    return locate_span(
        located["path"],
        row["byte_offset"],
        row["byte_length"],
        row["content_hash"],
        shifts[source],
    )


def resolve(conn: sqlite3.Connection, hits: list[Hit]) -> list[Passage]:
    """Turn ids and scores into displayable passages, one seek each."""
    passages = []
    shifts: dict[int, int] = {}
    for hit in hits:
        row = conn.execute(
            "SELECT byte_offset, byte_length, content_hash FROM chunks WHERE id = ?",
            (hit.chunk_id,),
        ).fetchone()
        located = db.locate(conn, hit.chunk_id)
        span = _span_of(located, row, shifts)
        passages.append(
            Passage(
                chunk_id=hit.chunk_id,
                score=hit.score,
                title=located["title"],
                path=located["path"],
                chapter=located["source_ordinal"],
                text=span.text,
                state=span.state,
                offset=span.offset,
            )
        )
    return passages


# --- near-duplicate suppression ----------------------------------------------

# Word 3-shingles: long enough that ordinary shared phrasing does not register,
# short enough to survive the small wording differences between two editions of
# the same passage.
SHINGLE = 3
DUPLICATE_AT = 0.10


def _shingles(text: str, size: int = SHINGLE) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + size]) for i in range(max(0, len(words) - size + 1))}


def overlap(first: str, second: str) -> float:
    """Jaccard overlap of word shingles: how literally alike two passages are."""
    a, b = _shingles(first), _shingles(second)
    return len(a & b) / len(a | b) if a | b else 0.0


def drop_near_duplicates(
    conn: sqlite3.Connection,
    hits: list[Hit],
    k: int = 5,
    threshold: float = DUPLICATE_AT,
) -> list[Hit]:
    """Keep the best hit of each near-identical group, in rank order.

    This is *not* the document-level deduplication §3 rejects. That capped every
    file at one passage and cost real answers; this drops a passage only when it
    is textually almost the same as one already kept, so a book may still
    contribute several genuinely different passages.

    The signal is lexical, and deliberately not vector similarity. Measured over
    result pairs on a corpus holding thirteen books in two translations, cosine
    could not tell "the same passage in the other edition" (median 0.829) from "a
    different passage in the same file" (median 0.828); suppressing at cosine
    0.75 removed 91% of the second kind. At a Jaccard of 0.10 the collateral was
    zero in every bucket measured, because literal overlap is what actually
    distinguishes a duplicate from a neighbour on the same subject.

    **Off by default.** Whether a near-duplicate is noise depends entirely on the
    library. On a corpus of thirteen books in two translations this cost two
    questions at recall@1 (25/40 to 23/40): the second edition of a passage was
    not redundancy, it was the comparison the reader wanted. On a corpus with no
    duplicates it gained three (69/110 to 72/110), and both movements are within
    noise at those sample sizes. So this is a preference about what results
    should look like, not a measured improvement, and it is offered rather than
    applied.
    """
    kept: list[Hit] = []
    kept_text: list[str] = []
    shifts: dict[int, int] = {}
    for hit in hits:
        row = conn.execute(
            "SELECT byte_offset, byte_length, content_hash FROM chunks WHERE id = ?",
            (hit.chunk_id,),
        ).fetchone()
        located = db.locate(conn, hit.chunk_id)
        text = _span_of(located, row, shifts).text
        # A passage that cannot be read cannot be compared; keep it rather than
        # silently dropping a result on the strength of a failed read.
        if text is None or all(overlap(text, seen) < threshold for seen in kept_text):
            kept.append(hit)
            kept_text.append(text or "")
        if len(kept) >= k:
            break
    return kept
