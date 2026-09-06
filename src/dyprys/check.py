"""What has drifted, and what work is outstanding.

The index records enough to answer both questions exactly -- content hashes for
drift, per-model prefixes for progress -- but recording is not the same as being
able to ask.  This is the asking.

Nothing here writes.  It reports, and names the command that would fix each
thing, because re-embedding is a decision the user makes deliberately.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from dyprys.ingest import _hash


@dataclass
class Drift:
    """Source files that no longer match what was indexed."""

    missing: list[str] = field(default_factory=list)   # gone from disk
    changed: list[str] = field(default_factory=list)   # different content
    intact: int = 0

    @property
    def clean(self) -> bool:
        return not self.missing and not self.changed


@dataclass
class ModelWork:
    """Outstanding embedding for one model, split by what it will cost."""

    id: int
    name: str
    dim: int
    embedded: int
    to_copy: int   # a vector already exists; carry it across an edit
    to_embed: int  # must actually run the model
    failed: int
    chunking: int | None = None    # which way of splitting this model works on
    chunk_target: int | None = None

    @property
    def outstanding(self) -> int:
        return self.to_copy + self.to_embed


# A *word* longer than this, at the 90th percentile of a book, means the
# extraction lost its word boundaries. Measured across the corpus: English prose
# 8-10, French 9, Latin 10, Middle English 7, maths notation 7, OpenStax 10 --
# and two PDFs whose embedded font map pdftotext could not follow, 62 and 80.
#
# Deliberately not a test for English words. That would flag the French, Latin
# and Middle English books in any real library, and the maths sheets, none of
# which are broken. Word *length* is a property of writing, not of a language.
#
# "Word" means a token that is mostly letters, and that qualifier is load-
# bearing. Counting every token flagged two perfectly good books: Gutenberg's
# "Number e to one million places" and the 1990 US Census, which are legitimate
# books that happen to be tables of digits. Telling someone to delete those is
# worse than not looking.
GLUED_TOKEN_LENGTH = 30


def _word_lengths(text: str) -> list[int]:
    """Lengths of the tokens that are mostly letters; digits are not words."""
    out = []
    for token in text.split():
        letters = sum(c.isalpha() for c in token)
        if letters * 2 >= len(token) and letters:
            out.append(len(token))
    return out


@dataclass
class Garbled:
    """A book whose text is not words -- worth knowing before embedding it."""

    title: str
    chunks: int
    p90_token: int


@dataclass
class Empty:
    """A book the index holds but that yielded no text to search."""

    title: str
    key: str
    bytes_on_disk: int


@dataclass
class Survey:
    books: int
    sources: int
    live_chunks: int      # reachable from a segment
    dead_chunks: int      # superseded by a re-ingest, reclaimable
    lexical_chunks: int   # present in the BM25 index
    drift: Drift
    models: list[ModelWork]
    garbled: list[Garbled] = field(default_factory=list)
    empty: list[Empty] = field(default_factory=list)


def survey(conn: sqlite3.Connection, deep: bool = False) -> Survey:
    """Compare the index against the filesystem and against each model."""
    books, sources, total = conn.execute(
        "SELECT (SELECT COUNT(*) FROM books), (SELECT COUNT(*) FROM sources), "
        "       (SELECT COUNT(*) FROM chunks)"
    ).fetchone()
    live = conn.execute(
        "SELECT COALESCE(SUM(chunk_count), 0) FROM segments"
    ).fetchone()[0]

    from dyprys.lexical import indexed_count

    return Survey(
        books=books,
        sources=sources,
        live_chunks=live,
        dead_chunks=total - live,
        lexical_chunks=indexed_count(conn),
        drift=_drift(conn, deep),
        models=_model_work(conn, live),
        garbled=garbled_books(conn),
        empty=empty_books(conn),
    )


def empty_books(conn: sqlite3.Connection, only: set[int] | None = None) -> list[Empty]:
    """Books with a source on disk but not one chunk to search.

    Extraction can fail by producing nothing at all -- a 29-byte file where a
    pitchbook should be -- and the result is a book that is listed, counted and
    reported as intact while no query can ever return it. Nothing else notices:
    `garbled_books` samples a book's chunks and a book without chunks has none
    to sample, drift compares bytes on disk against what was ingested and both
    agree, and coverage is a share of zero, which is complete.

    Unlike garbling this needs no threshold and no sample. A book with no
    chunks is unsearchable as a matter of fact, not of degree.
    """
    rows = conn.execute(
        "SELECT b.id, b.title, b.key, COALESCE(SUM(src.size_bytes), 0) AS bytes "
        "FROM books b JOIN sources src ON src.book_id = b.id "
        "LEFT JOIN segments seg ON seg.source_id = src.id "
        "GROUP BY b.id HAVING COALESCE(SUM(seg.chunk_count), 0) = 0 "
        "ORDER BY b.title"
    ).fetchall()
    return [Empty(r["title"], r["key"], r["bytes"]) for r in rows
            if only is None or r["id"] in only]


def garbled_books(
    conn: sqlite3.Connection,
    only: set[int] | None = None,
    per_book: int = 3,
    limit: int = 20,
) -> list[Garbled]:
    """Books whose text came out of extraction without word boundaries.

    Sampled, not exhaustive: three chunks a book is enough when the signal is a
    six-fold gap, and reading every passage to check them is the mistake this
    project exists to avoid. 0.83 ms a book, against the ~13 ms a book ingest
    itself costs.

    `only` restricts the scan to particular books, which is what `dyp add` wants
    -- the answer is most useful at ingest, before any GPU has been spent on the
    text and while re-extracting is still cheap.

    This warns and never acts. Skipping the chunks at embed time would have
    saved 0.15% of the work on the corpus that motivated it, in exchange for a
    heuristic silently dropping content; the first version of that heuristic
    flagged two perfectly good books. Warning is the whole of the benefit.
    """
    from dyprys.text import read_span

    found = []
    for book in conn.execute(
        "SELECT b.id, b.title, seg.chunk_start, seg.chunk_count, src.path "
        "FROM books b JOIN sources src ON src.book_id = b.id "
        "JOIN segments seg ON seg.source_id = src.id ORDER BY b.id"
    ):
        if only is not None and book["id"] not in only:
            continue
        step = max(1, book["chunk_count"] // per_book)
        lengths: list[int] = []
        for offset in range(0, book["chunk_count"], step):
            row = conn.execute(
                "SELECT byte_offset, byte_length, content_hash FROM chunks WHERE id = ?",
                (book["chunk_start"] + offset,),
            ).fetchone()
            if row is None:
                continue
            text = read_span(book["path"], row["byte_offset"], row["byte_length"],
                             row["content_hash"])
            if text:
                lengths.extend(_word_lengths(text))
            if len(lengths) > 4000:
                break
        if len(lengths) < 50:
            continue
        lengths.sort()
        p90 = lengths[int(len(lengths) * 0.9)]
        if p90 > GLUED_TOKEN_LENGTH:
            found.append(Garbled(book["title"], book["chunk_count"], p90))
        if len(found) >= limit:
            break
    return found


def _drift(conn: sqlite3.Connection, deep: bool) -> Drift:
    """Which source files have moved on since they were indexed.

    Size and mtime are the fast check.  They can prove a file *changed* only in
    the sense of being suspicious -- so `deep` re-hashes to be certain, which is
    what you want before discarding work.
    """
    drift = Drift()
    for row in conn.execute(
        "SELECT path, size_bytes, mtime, content_hash FROM sources ORDER BY path"
    ):
        path = Path(row["path"])
        try:
            info = path.stat()
        except OSError:
            drift.missing.append(row["path"])
            continue

        if deep:
            looks_same = _hash(path.read_bytes()) == row["content_hash"]
        else:
            looks_same = (
                info.st_size == row["size_bytes"] and info.st_mtime == row["mtime"]
            )

        if looks_same:
            drift.intact += 1
        else:
            drift.changed.append(row["path"])
    return drift


def _model_work(conn: sqlite3.Connection, live: int) -> list[ModelWork]:
    work = []
    for model in conn.execute(
            "SELECT id, name, dim, chunking_id FROM models ORDER BY id"):
        # A model embeds one chunking, so its outstanding work is that
        # chunking's chunks and not the library's. Counting all of them told a
        # user to embed chunks this model will never touch, and the hint said so
        # again after every run -- a loop with no exit.
        target = None
        mine = live
        if model["chunking_id"] is not None:
            row = conn.execute(
                "SELECT c.target, COALESCE(SUM(seg.chunk_count), 0) AS chunks "
                "FROM chunkings c LEFT JOIN segments seg ON seg.chunking_id = c.id "
                "WHERE c.id = ?", (model["chunking_id"],)).fetchone()
            if row:
                target, mine = row["target"], row["chunks"]
        embedded = conn.execute(
            "SELECT COALESCE(SUM(n_embedded), 0) FROM segment_progress WHERE model_id = ?",
            (model["id"],),
        ).fetchone()[0]
        to_copy = conn.execute(
            "SELECT COUNT(*) FROM chunk_carry WHERE model_id = ?", (model["id"],)
        ).fetchone()[0]
        failed = conn.execute(
            "SELECT COUNT(*) FROM chunk_failures WHERE model_id = ?", (model["id"],)
        ).fetchone()[0]
        work.append(
            ModelWork(
                id=model["id"],
                name=model["name"],
                dim=model["dim"],
                embedded=embedded,
                to_copy=to_copy,
                to_embed=max(0, mine - embedded - to_copy),
                failed=failed,
                chunking=model["chunking_id"],
                chunk_target=target,
            )
        )
    return work
