"""BM25 over chunks, and the fusion of it with vector search.

Vector search is weak at exactly the thing keyword search is trivially good at:
finding the passage containing a phrase you remember. Measured on this corpus,
an exact nine-word phrase found its own passage 70% of the time at 1,414 chunks
and 17% at 11,512 -- it degrades as the library grows, which is the direction
that matters. This module is the other half.
"""

from __future__ import annotations

import re
import sqlite3

from dyprys.search import Hit, embedded_ranges

# FTS5's MATCH grammar treats punctuation as syntax, so a natural question like
# "what is long-term potentiation?" is a syntax error rather than a query.
# Terms are extracted and quoted individually instead.
_WORD = re.compile(r"\w+", re.UNICODE)


def _terms(text: str) -> list[str]:
    """Query words, dropping only single *letters*.

    A single digit is not noise -- "chapter 7", "figure 3", "T1", "CA1" all turn
    on it, and dropping digits silently rewrote "Paragraph 7 of book 1" into
    "paragraph of book", which then matched nothing as a phrase and everything
    as an OR.
    """
    return [t for t in _WORD.findall(text.lower()) if len(t) > 1 or t.isdigit()]


def to_match_query(text: str) -> str:
    """A user's words as a safe FTS5 OR-query, or an empty string if none."""
    return " OR ".join('"%s"' % t for t in _terms(text))


def to_phrase_query(text: str) -> str:
    """The same words as an ordered phrase, or empty if there are too few.

    Tried before the OR-query, because the two answer different questions.
    Someone recalling a sentence wants the passage containing that sentence; an
    OR of its words instead ranks by how many common words a chunk happens to
    hold, which is how "Paragraph 7 of book 1" retrieves the chunk about
    paragraph 0.
    """
    terms = _terms(text)
    if len(terms) < 2:
        return ""
    return '"%s"' % " ".join(terms)


def index_chunks(conn: sqlite3.Connection, rows: list[tuple[int, str]]) -> None:
    """Add `(chunk_id, text)` pairs to the lexical index."""
    conn.executemany("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", rows)


def drop_chunks(conn: sqlite3.Connection, chunk_ids: list[int]) -> None:
    """Remove superseded chunks, so a stale id can never win a query."""
    conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(c,) for c in chunk_ids])


def indexed_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]


def search_bm25(
    conn: sqlite3.Connection,
    query: str,
    k: int = 5,
    model_id: int | None = None,
    book_ids: set[int] | None = None,
    phrase_everywhere: bool = True,
    origin: dict[int, str] | None = None,
) -> list[Hit]:
    """Best-matching chunks by BM25, best first.

    Restricted to chunks that also have a vector when `model_id` is given, so
    the two halves of a hybrid search are always drawn from the same population
    -- otherwise fusion would quietly favour whichever half saw more of the
    library.

    **The phrase attempt ignores `book_ids`.** `book_ids` comes from a router
    that scores books by vector similarity, and on 117 books that router keeps
    the right book for 109/110 natural questions but only 25/60 verbatim
    phrases. Confining the phrase search to its choices took lexical safety from
    20/20 to 9/20 -- it tied the safety net to the very failure it exists to
    catch. Set `phrase_everywhere=False` to measure the old behaviour.
    """
    if model_id is None and book_ids is not None:
        # The scope is derived from the model's embedded ranges, so without a
        # model there is nothing to derive it from -- and silently searching the
        # whole corpus is the wrong answer to "search these books".
        raise ValueError("book_ids needs a model_id: the scope comes from its ranges")
    scope = None
    if model_id is not None:
        scope = embedded_ranges(conn, model_id, book_ids)
        if not scope:
            return []

    # An exact phrase first, then the looser OR of its words. A remembered
    # sentence returns its own passage; a natural question, which rarely matches
    # as a phrase, falls through to the OR.
    phrase, words = to_phrase_query(query), to_match_query(query)

    # A phrase either occurs in the library or it does not: it is a lookup, not
    # a ranking, and it is cheap because an ordered match on several words is
    # rare. An OR of common words is a ranking, and there routing's premise --
    # that the answer sits in a topically near book -- is the one that holds.
    wide = scope
    if phrase and phrase_everywhere and book_ids is not None:
        wide = embedded_ranges(conn, model_id, None)

    found: list[Hit] = []
    seen: set[int] = set()
    for attempt, match, where in (("phrase", phrase, wide), ("words", words, scope)):
        if not match or len(found) >= k:
            continue
        for hit in _match(conn, match, k, where):
            if hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                found.append(hit)
                if origin is not None:
                    origin[hit.chunk_id] = attempt
    return found[:k]


# SQLite refuses an expression tree deeper than 1000 by default, and the scope
# becomes one OR-clause per range. Merged ranges make that 1 or 2 in practice --
# embedding advances in id order, so only the segment in flight is ever partial
# -- but "in practice" is how the first version of this passed every test and
# broke at 1,000 books. Past this many ranges the filter moves into Python.
MAX_RANGE_CLAUSES = 64


def _match(
    conn: sqlite3.Connection, match: str, k: int, scope: list[tuple[int, int]] | None
) -> list[Hit]:
    sql = "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts WHERE chunks_fts MATCH ?"
    params: list = [match]
    in_sql = scope is not None and len(scope) <= MAX_RANGE_CLAUSES

    if in_sql:
        sql += " AND (%s)" % " OR ".join("(rowid >= ? AND rowid < ?)" for _ in scope)
        for start, end in scope:
            params.extend((start, end))

    sql += " ORDER BY score LIMIT ?"
    # When filtering afterwards, ask for more than k so the survivors can still
    # fill the requested list.
    params.append(k if in_sql else k * 20)

    # bm25() is negative, most relevant most negative; flip it so that for every
    # Hit in the system, larger means better.
    hits = [Hit(row["rowid"], -row["score"]) for row in conn.execute(sql, params)]
    if in_sql or scope is None:
        return hits
    allowed = lambda cid: any(start <= cid < end for start, end in scope)
    return [hit for hit in hits if allowed(hit.chunk_id)][:k]


def reciprocal_rank_fusion(
    rankings: list[list[Hit]],
    k: int = 5,
    weights: list[float] | None = None,
    damping: int = 60,
) -> list[Hit]:
    """Combine ranked lists by rank, not by score.  Earlier lists win ties.

    Cosine similarities and BM25 scores are not on the same scale and never will
    be -- on a single-domain corpus the cosines compress into a narrow band while
    BM25 stays unbounded. Fusing on rank sidesteps the comparison entirely, which
    is why this is the default rather than a weighted sum of raw scores.

    **Ties are not an edge case here, they are the common case.** A passage at
    rank *r* of one list scores exactly what a different passage scores at rank
    *r* of another, so any two lists tie constantly: on 138 questions the top of
    the fused list was decided by an exact tie **82 times**. Which side wins them
    is worth more than any other choice in this module -- vector-first 77/110
    recall@1 against lexical-first 59/110 -- and it used to be decided by nothing
    more than the insertion order of a dict. It is now an explicit sort key, so
    it survives a refactor and can be tested.

    Callers therefore pass the ranking they trust on a tie *first*. `_searcher`
    passes vector before BM25: when both halves put something different at rank
    one, the vector hit is right more often, because BM25's top hit on a natural
    question is frequently keyword-dense and off-topic.

    **`weights` is a trap, and equal weights are load-bearing.** Nudging the
    vector half up looks like gentle tuning and is not:

        weights        rare r@1  rare r@5  main r@1  main r@5  lexical safety
        equal            13/28     24/28    77/110    97/110      29/30
        [1.3, 1.0]       13/28     21/28    77/110    99/110    ->  5/30
        [2.0, 1.0]       14/28     21/28    77/110    99/110    ->  5/30
        [1.0, 1.3]       12/28     20/28    59/110    83/110      30/30

    A 1.3x weight buys two questions at recall@5 and costs 24 of 30 exact-phrase
    probes, because every vector hit then outranks every lexical hit at the same
    position and BM25's rank-one match is pushed off the end. That is the
    vector-only failure the hybrid exists to prevent, reintroduced by a knob that
    reads as harmless. The tie-break above gives the vector half the benefit of
    the doubt without letting it dominate; a weight does not.

    **`damping` does nothing here.** 10, 30, 60 and 120 give identical results on
    every measure above, because with two lists it only rescales an ordering it
    cannot change. 60 is the value from the original paper and there is no reason
    to touch it.
    """
    weights = weights or [1.0] * len(rankings)
    totals: dict[int, float] = {}
    first_seen: dict[int, int] = {}
    for index, (ranking, weight) in enumerate(zip(rankings, weights)):
        for rank, hit in enumerate(ranking):
            totals[hit.chunk_id] = totals.get(hit.chunk_id, 0.0) + weight / (damping + rank + 1)
            first_seen.setdefault(hit.chunk_id, index)
    ordered = sorted(totals.items(), key=lambda kv: (-kv[1], first_seen[kv[0]], kv[0]))
    return [Hit(chunk_id, score) for chunk_id, score in ordered[:k]]


def provenance(sides: list[tuple[str, list[Hit]]], hits: list[Hit],
               rename: dict[str, dict[int, str]] | None = None) -> dict[int, str]:
    """Where each fused hit came from, as the rank it held in each half.

    This is what the fused *score* cannot say. An RRF score is
    `sum(1/(60 + rank))`, so its whole range is 0.016 to 0.033 for two lists --
    a perfect match reads as 0.033, which looks like 3%, and two results that
    each led one half tie at 0.016 with nothing to separate them. It is an
    artefact of the damping constant, not a property of the match, and printing
    it invites exactly the reading it cannot bear.

    A rank per half is the real basis for the ordering, and it says the thing a
    reader actually wants: whether this passage was found because it *means* the
    query or because it *contains* it.
    """
    where: dict[int, list[str]] = {}
    for name, ranking in sides:
        per_hit = (rename or {}).get(name, {})
        for rank, hit in enumerate(ranking, 1):
            label = per_hit.get(hit.chunk_id, name)
            where.setdefault(hit.chunk_id, []).append(f"{label} {rank}")
    return {hit.chunk_id: " · ".join(where.get(hit.chunk_id, [])) or "—" for hit in hits}


def backfill(conn: sqlite3.Connection, progress=None) -> int:
    """Build the lexical index for a library ingested before it existed.

    Streams one chunk at a time: reading every passage into memory to index them
    is the failure this project exists to avoid, and it would work fine right up
    until the library got large.
    """
    from dyprys.text import read_span

    conn.execute("DELETE FROM chunks_fts")
    total = conn.execute("SELECT COALESCE(SUM(chunk_count), 0) FROM segments").fetchone()[0]
    done = 0
    for segment in conn.execute(
        "SELECT seg.chunk_start, seg.chunk_count, src.path FROM segments seg "
        "JOIN sources src ON src.id = seg.source_id ORDER BY seg.chunk_start"
    ).fetchall():
        batch = []
        for row in conn.execute(
            "SELECT id, byte_offset, byte_length, content_hash FROM chunks "
            "WHERE id >= ? AND id < ? ORDER BY id",
            (segment["chunk_start"], segment["chunk_start"] + segment["chunk_count"]),
        ):
            text = read_span(
                segment["path"], row["byte_offset"], row["byte_length"], row["content_hash"]
            )
            if text is not None:
                batch.append((row["id"], text))
        index_chunks(conn, batch)
        conn.commit()
        done += segment["chunk_count"]
        if progress:
            progress(done, total)
    return indexed_count(conn)
