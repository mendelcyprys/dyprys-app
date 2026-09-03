"""Step 02: turn chunks into vectors, resumably.

Three properties this has to hold, all of which are about interruption rather
than throughput, because the corpus takes hundreds of hours and being stopped is
normal:

* **Chunk-level durability.** An embedded chunk is never recomputed.
* **Ordering.** The vector is written and flushed *before* the count that claims
  it exists. A crash in between re-embeds one chunk -- wasted work, not a row
  pointing at nothing.
* **Ascending within a segment.** Progress is a single count of the embedded
  prefix, so the work must advance in id order for that count to mean anything.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from dyprys import db
from dyprys.text import read_span
from dyprys.vectors import VectorStore


@dataclass
class EmbedReport:
    embedded: int = 0   # ran the model
    copied: int = 0     # carried across an edit; near-free
    failed: int = 0     # source unreadable; zero vector written, recorded
    segments: int = 0
    stopped: str = "complete"  # or "limit" / "time" / "interrupted"


def embed_pending(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder,
    model_id: int,
    limit: int | None = None,
    batch_size: int = 8,
    seconds: float | None = None,
    should_stop: Callable[[], bool] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    book_ids: set[int] | None = None,
    chunking_id: int | None = None,
    duty: float = 1.0,
) -> EmbedReport:
    """Embed what is outstanding for `model_id`.

    `duty` is the share of wall time to spend working, 1.0 being flat out. It
    pauses *between* batches, never inside one, so a batch is still committed
    whole and stopping is unaffected.

    There is no thermal headroom to reclaim by doing this: measured over seven
    continuous minutes on an M4 the rate is flat at 5.2 chunks/s, with no
    decline. So a duty of 0.8 costs very close to the 20% it says, and buys a
    responsive machine rather than free time. macOS offers no GPU quota, so this
    is the only way to ask for less than all of it -- `taskpolicy -b` reduces
    throughput about 11% but is not adjustable.

    `chunking_id` restricts the work to one way of splitting the text. A model
    embeds one chunking and only one: its vectors are a single population and
    every score compares within it, so two granularities in there means a search
    returns a passage and a piece of that same passage as separate results.
    Without this a second `dyp add --target N` silently doubled both the
    embedding bill and the result list.

    `book_ids` restricts the work to particular books, which is what makes a
    mixed library usable: at 3,453 books "embed the neuroscience shelf first" is
    an ordinary thing to want, and without it the only choice is all of it in
    whatever order ids were allocated. It narrows the *whole* operation --
    including the total that progress is reported against, so a scoped run does
    not report itself as 4% of a library it was never asked to touch.

    `limit` caps chunks, `seconds` caps wall time, and `should_stop` lets the
    caller ask for a halt -- a Ctrl-C, say.  All three stop *between* batches,
    after the current one is written and committed: the point of stopping is to
    bound how long you wait, never to throw away what was computed. A prior tool capped
    sessions at 30 minutes and discarded partially embedded documents on expiry,
    so resume runs never converged; that is the failure this signature exists to
    avoid.
    """
    report = EmbedReport()
    budget = limit if limit is not None else -1

    sql = ("SELECT seg.id, seg.chunk_start, seg.chunk_count, src.path "
           "FROM segments seg JOIN sources src ON src.id = seg.source_id ")
    where, params = [], []
    if book_ids is not None:
        if not book_ids:
            return report          # an empty scope is no work, never all of it
        where.append("src.book_id IN (%s)" % ",".join("?" * len(book_ids)))
        params += sorted(book_ids)
    if chunking_id is not None:
        where.append("seg.chunking_id = ?")
        params.append(chunking_id)
    if where:
        sql += "WHERE " + " AND ".join(where) + " "
    segments = conn.execute(sql + "ORDER BY seg.chunk_start", params).fetchall()

    total = _outstanding(conn, model_id, book_ids, chunking_id)
    seen = 0
    deadline = time.monotonic() + seconds if seconds is not None else None

    for segment in segments:
        if budget == 0 or _expired(deadline) or (should_stop and should_stop()):
            break
        done = db.embedded_prefix(conn, model_id, segment["id"])
        if done >= segment["chunk_count"]:
            continue
        report.segments += 1

        while (
            done < segment["chunk_count"]
            and budget != 0
            and not _expired(deadline)
            and not (should_stop and should_stop())
        ):
            take = min(batch_size, segment["chunk_count"] - done)
            if budget > 0:
                take = min(take, budget)
            first = segment["chunk_start"] + done

            batch_began = time.monotonic()
            _fill(conn, store, embedder, model_id, segment["path"], first, take, report)
            worked = time.monotonic() - batch_began

            # Vectors first, and durably, before anything claims they exist.
            store.flush()
            with conn:
                db.set_embedded_prefix(conn, model_id, segment["id"], done + take)
                conn.execute(
                    "DELETE FROM chunk_carry WHERE model_id = ? "
                    "AND new_chunk_id >= ? AND new_chunk_id < ?",
                    (model_id, first, first + take),
                )

            done += take
            seen += take
            if 0 < duty < 1:
                # Pause between batches, never inside one. Sleeping in
                # proportion to the batch just done keeps the ratio right
                # whatever the batch size or the model's speed.
                time.sleep(worked * (1 - duty) / duty)
            if budget > 0:
                budget -= take
            if progress:
                progress(seen, total, Path(segment["path"]).stem)

    if _outstanding(conn, model_id):
        if should_stop and should_stop():
            report.stopped = "interrupted"
        elif _expired(deadline):
            report.stopped = "time"
        elif budget == 0:
            report.stopped = "limit"
    return report


def _expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _fill(conn, store, embedder, model_id, path, first, count, report) -> None:
    """Give ids [first, first+count) a vector, by copy where possible."""
    rows = conn.execute(
        "SELECT id, byte_offset, byte_length, content_hash FROM chunks "
        "WHERE id >= ? AND id < ? ORDER BY id",
        (first, first + count),
    ).fetchall()
    carries = dict(
        conn.execute(
            "SELECT new_chunk_id, old_chunk_id FROM chunk_carry "
            "WHERE model_id = ? AND new_chunk_id >= ? AND new_chunk_id < ?",
            (model_id, first, first + count),
        )
    )

    to_embed: list[tuple[int, str]] = []
    for row in rows:
        old = carries.get(row["id"])
        if old is not None:
            store.copy(old, row["id"])
            report.copied += 1
            continue

        text = read_span(path, row["byte_offset"], row["byte_length"], row["content_hash"])
        if text is None:
            # The source moved on without being re-indexed. Write a zero vector,
            # which scores 0 against every query and so can never win, and record
            # why -- one bad chunk must not block the rest of its segment.
            store.write(row["id"], np.zeros(store.dim, dtype=np.float32))
            _record_failure(conn, model_id, row["id"], "source unreadable or changed")
            report.failed += 1
            continue
        to_embed.append((row["id"], text))

    if to_embed:
        vectors = embedder.embed_documents([text for _, text in to_embed])
        for (chunk_id, _), vector in zip(to_embed, vectors):
            store.write(chunk_id, vector)
        report.embedded += len(to_embed)


def _record_failure(conn, model_id: int, chunk_id: int, reason: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO chunk_failures (model_id, chunk_id, reason, failed_at) "
        "VALUES (?, ?, ?, ?)",
        (model_id, chunk_id, reason, db.now()),
    )


def _outstanding(conn: sqlite3.Connection, model_id: int,
                 book_ids: set[int] | None = None,
                 chunking_id: int | None = None) -> int:
    """Chunks with no vector under this model, within `book_ids` if given.

    Both halves are scoped the same way. Counting live chunks library-wide
    against progress for a few books would report a scoped run as a tiny
    fraction of work it was never asked to do.
    """
    if book_ids is not None and not book_ids:
        return 0
    where, params = [], []
    if book_ids is not None:
        where.append("src.book_id IN (%s)" % ",".join("?" * len(book_ids)))
        params += sorted(book_ids)
    if chunking_id is not None:
        where.append("seg.chunking_id = ?")
        params.append(chunking_id)
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    live = conn.execute(
        "SELECT COALESCE(SUM(seg.chunk_count), 0) FROM segments seg "
        "JOIN sources src ON src.id = seg.source_id" + clause, params).fetchone()[0]
    done = conn.execute(
        "SELECT COALESCE(SUM(p.n_embedded), 0) FROM segment_progress p "
        "JOIN segments seg ON seg.id = p.segment_id "
        "JOIN sources src ON src.id = seg.source_id "
        "WHERE p.model_id = ?" + (" AND " + " AND ".join(where) if where else ""),
        [model_id, *params]).fetchone()[0]
    return live - done


def store_for(conn: sqlite3.Connection, directory, model_id: int, dim: int) -> VectorStore:
    """The vector file for one model, sized to the whole library.

    The storage format comes from the model row rather than from the file, so a
    caller can never open an int8 file as fp32 -- which would read successfully
    and return nonsense.
    """
    rows = conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()[0]
    return VectorStore(
        db.vectors_path(directory, model_id),
        dim=dim,
        rows=rows,
        quantisation=db.model_quantisation(conn, model_id),
    )
