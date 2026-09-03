"""Step 05: route to a few books, then search only those.

Flat search is exact but reads the whole vector array, which at 5,000 books is
42.7 GB per query. Routing narrows the field first: each book is reduced to a
small set of topical directions, a query is scored against every book's
directions, and stage 2 searches only the best few.

The point is the *shape* of the cost, not today's speed. Stage 2 depends on how
many books you route to, not on how many you own.

A single centroid per book would be too coarse -- a textbook spans many subjects
and its average is a direction pointing at none of them -- so a book is a set of
directions. The design doc's figure is 16.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dyprys import db
from dyprys.search import Hit, embedded_ranges, flat_search
from dyprys.vectors import VectorStore

# Measured on 117 books, against the 28 questions whose answers sit in a median
# of 2 books -- the only ones able to see stage 1 at all:
#
#   k/book   routing recall   kept flat's top book   recall@1   centroid file
#        4          20/28            19/28             8/28          1.4 MB
#        8          22/28            19/28            10/28          2.9 MB
#       16          27/28            22/28            13/28          5.8 MB
#       32          27/28            22/28            13/28         11.5 MB
#       64          26/28            22/28            12/28         23.0 MB
#
# Four and eight are plainly too coarse -- 8 and 10 against 13 clears the noise
# floor below. Sixteen is the knee, and 32 and 64 are indistinguishable from it
# on every column that matters. On the *main* set "kept flat's top book" does keep
# climbing -- 71, 78, 81 of 110 -- while recall@1 goes 76, 74, 78, which is
# noise. So finer routing agrees with flat search more often and returns
# equally good passages either way: when stage 1 picks a different book, that
# book's passage is just as good. Agreement is a diagnostic, not an objective.
# How many starts k-means gets per book. Measured over five seeds on the
# rare-answer set, restarts=1 against restarts=8:
#
#                    routing recall   two builds agree on
#   restarts=1           24-27/28        75% of books
#   restarts=8           25-26/28        80% of books
#
# It buys stability, not recall: answer recall@1 spans 11-13/28 either way.
# Kept because the cost is 0.3s -> 2.1s over 117 books and the alternative is
# an index whose routing depends on an arbitrary constant. Do not expect it to
# raise a score.
RESTARTS = 8

# THE NOISE FLOOR. Because the seed alone moves rare-set recall@1 across 11-13
# and routing recall across 25-26, **a routing change worth fewer than about
# two questions on that set is not a result.** Several comparisons already in
# this file sit under that line and are recorded as "no difference" rather than
# as a winner. Anything measured on one seed inherits that seed's luck; seed 0,
# which everything here was first measured on, happened to be the best of five.
NOISE_FLOOR_QUESTIONS = 2

CENTROIDS_PER_BOOK = 16

# Re-derived on the rare-answer set, which is the only one that can see stage 1.
# "chance" is the same routing recall for books picked at random.
#
#   B    rare r@1  rare r@5  routing  chance  |  main r@1  lexical  scanned
#   1       9/28     17/28    17/28     2%    |   70/110    23/30     1.1%
#   2      11/28     20/28    22/28     5%    |   70/110    23/30     2.6%
#   3      13/28     22/28    24/28     7%    |   73/110    23/30     4.1%
#   5      13/28     23/28    27/28    11%    |   76/110    25/30     6.6%
#   8      13/28     23/28    27/28    16%    |   76/110    26/30    10.3%
#  12      13/28     23/28    27/28    23%    |   75/110    26/30    16.2%
#  flat    13/28     24/28       —      —     |   77/110    29/30   100.0%
#
# One and two lose real ground -- 9 and 11 against 13 is well past the noise
# floor below.
#
# Three against five is NOT distinguishable, and the "76 against 73" that first
# justified this default is noise. Tested paired on the frozen 117-book index:
# seven questions only B=5 gets right, four only B=3 gets right, exact two-sided
# p = 0.549. On the rare set it is one and one, p = 1.000.
#
# Five is kept anyway, on a stated ground rather than a recall difference that
# is not there: routing recall is the *ceiling* on what stage 2 can reach, and
# it is 27/28 against 24/28. That headroom is what a future reranker would have
# to work with. The cost is a share of the library, and it falls as the library
# grows -- five of 5,000 books is 0.1%. If that ever stops being cheap, three is
# the free saving, because nothing measurable is lost by it.
DEFAULT_BOOKS = 5


def spherical_kmeans(
    vectors: np.ndarray, k: int, iterations: int = 25, seed: int = 0,
    restarts: int = RESTARTS,
) -> np.ndarray:
    """`k` unit-length directions summarising unit-length `vectors`.

    Spherical rather than Euclidean because the vectors are normalised and
    compared by dot product: the quantity that matters is direction, and a mean
    that is not renormalised drifts inside the sphere where no real vector lives.

    Run `restarts` times from different starts, keeping the best. k-means finds
    a local optimum decided by where it starts, and on this corpus the seed
    alone moved answer recall@1 by three questions in 28 and left two runs
    agreeing on only 73% of the books they routed to. One seeded run is not a
    profile of a book, it is a sample of one.
    """
    if len(vectors) <= k:
        return vectors.copy()

    best, best_score = None, -np.inf
    for attempt in range(max(1, restarts)):
        candidate = _one_kmeans(vectors, k, iterations, seed * 1000 + attempt)
        # The objective the clustering maximises, and the quantity routing
        # actually uses: how well each chunk is covered by its nearest
        # direction. Scoring by anything else would optimise the wrong thing.
        score = float((vectors @ candidate.T).max(axis=1).sum())
        if score > best_score:
            best, best_score = candidate, score
    return best


def _one_kmeans(vectors: np.ndarray, k: int, iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = vectors[rng.choice(len(vectors), size=k, replace=False)].copy()

    for _ in range(iterations):
        assignment = (vectors @ centroids.T).argmax(axis=1)
        moved = np.zeros_like(centroids)
        for cluster in range(k):
            members = vectors[assignment == cluster]
            # An empty cluster is reseeded rather than left at zero, which would
            # otherwise score 0 against every query and silently waste a slot.
            moved[cluster] = (
                members.sum(axis=0) if len(members) else vectors[rng.integers(len(vectors))]
            )
        lengths = np.linalg.norm(moved, axis=1, keepdims=True)
        np.divide(moved, lengths, out=moved, where=lengths > 0)
        if np.allclose(moved, centroids):
            break
        centroids = moved
    return centroids


@dataclass
class BuildReport:
    books: int = 0
    centroids: int = 0
    skipped: int = 0  # nothing embedded yet


def build_centroids(
    conn: sqlite3.Connection,
    directory: Path,
    store: VectorStore,
    model_id: int,
    per_book: int = CENTROIDS_PER_BOOK,
    progress=None,
) -> BuildReport:
    """Profile every book that has vectors.  Idempotent and re-runnable."""
    books = conn.execute("SELECT id, title FROM books ORDER BY id").fetchall()
    report = BuildReport()

    plan: list[tuple[int, np.ndarray]] = []
    for book in books:
        vectors = _book_vectors(conn, store, model_id, book["id"])
        if vectors is None or not len(vectors):
            report.skipped += 1
            continue
        plan.append((book["id"], spherical_kmeans(vectors, per_book)))
        report.books += 1
        if progress:
            progress(report.books, len(books), book["title"])

    total = sum(len(c) for _, c in plan)
    path = db.centroids_path(directory, model_id)
    path.unlink(missing_ok=True)
    centroids = VectorStore(path, dim=store.dim, rows=max(total, 1))

    with conn:
        conn.execute("DELETE FROM book_centroids WHERE model_id = ?", (model_id,))
        row = 1
        for book_id, block in plan:
            centroids.write_many(row, block)
            conn.execute(
                "INSERT INTO book_centroids "
                "(model_id, book_id, centroid_start, centroid_count, built_from) "
                "VALUES (?, ?, ?, ?, ?)",
                (model_id, book_id, row, len(block), _embedded_chunks(conn, model_id, book_id)),
            )
            row += len(block)
    centroids.flush()
    report.centroids = total
    return report


def _book_vectors(conn, store: VectorStore, model_id: int, book_id: int) -> np.ndarray | None:
    """Every embedded vector for one book, as one array."""
    ranges = _book_ranges(conn, model_id, book_id)
    if not ranges:
        return None
    blocks = [store.slice(start, end - start) for start, end in ranges]
    return np.concatenate(blocks) if len(blocks) > 1 else blocks[0]


def _book_ranges(conn, model_id: int, book_id: int) -> list[tuple[int, int]]:
    return [
        (row["chunk_start"], row["chunk_start"] + row["n_embedded"])
        for row in conn.execute(
            "SELECT seg.chunk_start, p.n_embedded FROM segment_progress p "
            "JOIN segments seg ON seg.id = p.segment_id "
            "JOIN sources src ON src.id = seg.source_id "
            "WHERE p.model_id = ? AND src.book_id = ? AND p.n_embedded > 0 "
            "ORDER BY seg.chunk_start",
            (model_id, book_id),
        )
    ]


def _embedded_chunks(conn, model_id: int, book_id: int) -> int:
    return sum(end - start for start, end in _book_ranges(conn, model_id, book_id))


def is_built(conn: sqlite3.Connection, model_id: int) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM book_centroids WHERE model_id = ? LIMIT 1", (model_id,)
        ).fetchone()
    )


def stale_books(conn: sqlite3.Connection, model_id: int) -> int:
    """Books whose embedded chunk count no longer matches their profile."""
    stale = 0
    for row in conn.execute(
        "SELECT book_id, built_from FROM book_centroids WHERE model_id = ?", (model_id,)
    ).fetchall():
        if _embedded_chunks(conn, model_id, row["book_id"]) != row["built_from"]:
            stale += 1
    return stale


def route(
    conn: sqlite3.Connection,
    centroids: VectorStore,
    query: np.ndarray,
    model_id: int,
    books: int = DEFAULT_BOOKS,
    candidates: set[int] | None = None,
) -> list[tuple[int, float]]:
    """The best `books` book ids for this query, best first.

    A book is scored by its *best* direction, not its average: a query about one
    chapter of a broad textbook should reach it on the strength of that chapter.
    """
    rows = conn.execute(
        "SELECT book_id, centroid_start, centroid_count FROM book_centroids "
        "WHERE model_id = ? ORDER BY centroid_start",
        (model_id,),
    ).fetchall()
    if not rows:
        return []

    # Every centroid in one matrix multiply, then the per-book maximum by
    # segment. The obvious loop -- slice one book's block, dot it, take the max
    # -- costs a Python iteration and a BLAS call per *book*, which is invisible
    # at 117 books and 4 ms at 1,486. Stage 1 is the one part of a routed query
    # that grows with the size of the library, so it is the one part worth
    # writing this way.
    starts = np.fromiter((r["centroid_start"] - 1 for r in rows), dtype=np.intp, count=len(rows))
    counts = np.fromiter((r["centroid_count"] for r in rows), dtype=np.intp, count=len(rows))
    span = int(starts[-1] + counts[-1])
    similarity = centroids.slice(1, span) @ query
    # reduceat needs the boundaries; a book with no centroids would make it read
    # the wrong segment, and `build_centroids` never writes one.
    best = np.maximum.reduceat(similarity, starts)

    if candidates is not None:
        keep = np.fromiter((r["book_id"] in candidates for r in rows),
                           dtype=bool, count=len(rows))
        if not keep.any():
            return []
        best = np.where(keep, best, -np.inf)

    order = np.argsort(-best)[:books]
    return [(rows[i]["book_id"], float(best[i])) for i in order
            if np.isfinite(best[i])]


def two_stage_search(
    conn: sqlite3.Connection,
    store: VectorStore,
    centroids: VectorStore,
    query: np.ndarray,
    model_id: int,
    k: int = 5,
    books: int = DEFAULT_BOOKS,
    candidates: set[int] | None = None,
) -> tuple[list[Hit], set[int]]:
    """Route, then search exactly inside the routed books."""
    routed = {book_id for book_id, _ in route(conn, centroids, query, model_id, books, candidates)}
    if not routed:
        return [], set()
    return flat_search(conn, store, query, model_id, k, book_ids=routed), routed


def scanned_fraction(conn: sqlite3.Connection, model_id: int, routed: set[int]) -> float:
    """Share of the embedded corpus stage 2 actually touched."""
    everything = sum(e - s for s, e in embedded_ranges(conn, model_id))
    if not everything:
        return 0.0
    touched = sum(e - s for s, e in embedded_ranges(conn, model_id, routed))
    return touched / everything
