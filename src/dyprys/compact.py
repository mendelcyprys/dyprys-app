"""Reclaiming dead chunk ids, and with them the ability to remove a book.

A chunk's id *is* its row in every model's vector file, and ids are dense. That
is what makes a routed book one contiguous slice, and it is also why nothing can
simply be deleted: removing ids leaves holes, and closing the holes renumbers
everything that points into them.

So compaction rewrites all of it at once — vectors, chunk rows, segment starts,
recorded failures, and the lexical index. The design document called this a
compaction operation rather than a delete, and this is what that costs.

Two properties make it safe to do in place:

* **New ids are never larger than old ones.** Compaction only closes gaps, so
  every live chunk moves down or stays put.
* **Runs are copied in ascending order.** Run *i* writes to `[T, T+C)` where
  `T <= S`, and the next run starts at `S' >= S + C >= T + C`, so a write can
  never land on a row that has not been read yet.

Together those mean no temporary copy of a 40 GB vector file is needed.

What they do *not* give is a safe replay. A run longer than the gap it closes
writes over rows it has still to read, so repeating it reads what the first pass
already shifted. Compaction is interruptible by design, and a resume used to
restart the current model at run 0 -- silently attaching vectors to the wrong
passages, worst when the book removed was the *smallest*, since a small gap is
one every later segment overruns. So progress is now recorded per run, and the
overlapping runs alone are staged through a scratch file: staging is repeatable
because it reads a source nothing has touched, and unstaging is repeatable
because it reads the scratch. Every step is therefore safe to redo, which is
what "resumable" has to mean.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from dyprys import db
from dyprys.vectors import VectorStore

STATE_KEY = "compaction"

# Which half of the work is outstanding.  The vector files are moved first;
# once the rows are renumbered there is no going back to that phase.
VECTORS = "vectors"
LEXICAL = "lexical"


@dataclass(frozen=True)
class Run:
    """A stretch of live chunk ids moving from `old_start` down to `new_start`."""

    segment_id: int
    old_start: int
    new_start: int
    count: int

    @property
    def moves(self) -> bool:
        return self.old_start != self.new_start

    @property
    def self_overlaps(self) -> bool:
        """Does this run write over rows it has still to read?

        New ids are never larger than old ones, so the destination sits below
        the source; it reaches back into it when the run is longer than the gap
        it is closing. Such a move is *not* repeatable -- a second pass reads
        rows the first pass already shifted -- which is what makes it the one
        thing a resume cannot simply redo.
        """
        return self.moves and self.new_start + self.count > self.old_start


@dataclass
class Plan:
    runs: list[Run] = field(default_factory=list)
    live: int = 0
    dead: int = 0

    @property
    def worth_doing(self) -> bool:
        return self.dead > 0


def outstanding_carries(conn: sqlite3.Connection) -> int:
    """Carries point at superseded rows, which compaction is about to move."""
    return conn.execute("SELECT COUNT(*) FROM chunk_carry").fetchone()[0]


def plan(conn: sqlite3.Connection) -> Plan:
    """Where every live chunk will end up.

    Segments are the natural unit: a segment's chunks are already contiguous, so
    the whole mapping is a handful of runs rather than one row per chunk.
    """
    total = conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()[0]
    segments = conn.execute(
        "SELECT id, chunk_start, chunk_count FROM segments ORDER BY chunk_start"
    ).fetchall()

    runs, cursor = [], 1
    for segment in segments:
        runs.append(Run(segment["id"], segment["chunk_start"], cursor, segment["chunk_count"]))
        cursor += segment["chunk_count"]
    live = cursor - 1
    return Plan(runs=runs, live=live, dead=total - live)


# --- crash safety ------------------------------------------------------------


def _save_state(conn: sqlite3.Connection, state: dict) -> None:
    with conn:
        db.set_meta(conn, STATE_KEY, json.dumps(state))


def _load_state(conn: sqlite3.Connection) -> dict | None:
    raw = db.get_meta(conn, STATE_KEY)
    return json.loads(raw) if raw else None


def _clear_state(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("DELETE FROM meta WHERE key = ?", (STATE_KEY,))


def interrupted(conn: sqlite3.Connection) -> bool:
    """True when a previous compaction stopped part-way through."""
    return _load_state(conn) is not None


# --- doing it ----------------------------------------------------------------


@dataclass
class CompactReport:
    reclaimed: int = 0
    live: int = 0
    models: int = 0
    resumed: bool = False


def compact(
    conn: sqlite3.Connection,
    directory: Path,
    progress=None,
) -> CompactReport:
    """Close every gap in the chunk id space.  Resumable; safe to re-run."""
    if outstanding_carries(conn):
        raise RuntimeError(
            f"{outstanding_carries(conn):,} carries still point at superseded rows. "
            f"Run `dyp embed` to consume them first — compacting now would move the "
            f"rows they are waiting to copy."
        )

    state = _load_state(conn)
    resumed = state is not None
    if state is None:
        shape = plan(conn)
        if not shape.worth_doing:
            return CompactReport(reclaimed=0, live=shape.live)
        state = {
            "runs": [[r.segment_id, r.old_start, r.new_start, r.count] for r in shape.runs],
            "live": shape.live,
            "dead": shape.dead,
            "models_done": [],
            "at": {},          # model id -> how far through the runs it got
            "phase": VECTORS,
        }
        _save_state(conn, state)

    runs = [Run(*r) for r in state["runs"]]
    models = conn.execute("SELECT id, dim FROM models ORDER BY id").fetchall()

    if state.get("phase", VECTORS) == VECTORS:
        for model in models:
            if model["id"] in state["models_done"]:
                continue
            key = str(model["id"])

            def record(run_index, staged, key=key):
                state.setdefault("at", {})[key] = {"run": run_index, "staged": staged}
                _save_state(conn, state)

            _move_vectors(directory, model["id"], model["dim"], runs, state["live"],
                          db.model_quantisation(conn, model["id"]), progress,
                          at=state.get("at", {}).get(key), record=record)
            state["models_done"].append(model["id"])
            state.get("at", {}).pop(key, None)
            _save_state(conn, state)

        for model in models:
            _scratch_path(directory, model["id"]).unlink(missing_ok=True)

        # Renumbering and the move to the next phase land in one transaction:
        # rebuilding `chunks` a second time, over ids that are already the new
        # ones, would be its own corruption.
        _rewrite_rows(conn, runs, state)

    # Contentless FTS5 keeps no text, so its rowids cannot be renumbered in
    # place -- there is nothing to re-insert. It is rebuilt from the source
    # files, which is fast and is the only store that can be reconstructed.
    from dyprys.lexical import backfill
    backfill(conn)
    _clear_state(conn)
    return CompactReport(
        reclaimed=state["dead"], live=state["live"], models=len(models), resumed=resumed
    )


def _scratch_path(directory: Path, model_id: int) -> Path:
    """Where an overlapping run parks its rows so the move can be repeated."""
    return Path(directory) / f".compact-{model_id:03d}.scratch"


def _move_vectors(directory, model_id, dim, runs, live, quantisation, progress=None,
                  at=None, record=None) -> None:
    """Slide every live row down into its new place, then truncate the tail.

    Every step here is safe to repeat, which is what lets `at` be a simple count
    of finished runs rather than a promise about where the process died:

    * A run that does not self-overlap is copied straight across. Its source is
      untouched -- earlier runs write strictly below it -- so redoing it reads
      the same bytes and writes the same bytes.
    * A run that does self-overlap is staged into a scratch file first. Staging
      reads that same untouched source, so it repeats; unstaging reads the
      scratch, so it repeats too. `staged` says which of the two a resume owes.
    * `truncate` to a length it may already have is a no-op.
    """
    from dyprys.vectors import bytes_per_row

    path = db.vectors_path(directory, model_id)
    if not path.exists():
        return
    width = bytes_per_row(quantisation, dim)
    rows = max(path.stat().st_size // width, live)
    at = at or {}
    done, staged = at.get("run", 0), at.get("staged")

    store = VectorStore(path, dim=dim, rows=rows, quantisation=quantisation)
    scratch = None
    try:
        for index in range(done, len(runs)):
            run = runs[index]
            if run.self_overlaps:
                if scratch is None:
                    scratch = _open_scratch(directory, model_id, dim, quantisation, runs)
                if staged != index:
                    scratch.write_raw(1, store.read_raw(run.old_start, run.count))
                    scratch.flush()
                    staged = index
                    if record:
                        record(index, index)
                store.write_raw(run.new_start, scratch.read_raw(1, run.count))
            elif run.moves:
                store.move_rows(run.old_start, run.new_start, run.count)

            # Rows first and durably, then the count that claims they are there
            # -- the same ordering embedding uses, for the same reason.
            store.flush()
            staged = None
            if record:
                record(index + 1, None)
            if progress:
                progress(model_id, index + 1, len(runs))
        store.truncate(live)
    finally:
        store.close()
        if scratch is not None:
            scratch.close()


def _open_scratch(directory, model_id, dim, quantisation, runs) -> VectorStore:
    """A staging file one run wide.  Only the overlapping runs ever touch it."""
    widest = max((r.count for r in runs if r.self_overlaps), default=1)
    return VectorStore(
        _scratch_path(directory, model_id), dim=dim, rows=max(widest, 1),
        quantisation=quantisation,
    )


def _rewrite_rows(conn: sqlite3.Connection, runs: list[Run], state: dict) -> None:
    """Renumber chunks, segments and failures, and leave the vector phase behind.

    The chunks table is rebuilt rather than updated in place: moving a primary
    key row by row invites a transient collision, and a fresh table also drops
    the dead rows for free by simply not selecting them.

    Unlike a vector move, this one cannot be repeated: the second pass would
    look for the old ids and find the new ones already sitting there. So the
    phase advances inside the same transaction, and a crash either rolls the
    whole renumber back or lands on the far side of it.
    """
    with conn:
        conn.execute("DROP TABLE IF EXISTS remap")
        conn.execute(
            "CREATE TEMP TABLE remap (segment_id INTEGER, old_start INTEGER, "
            "new_start INTEGER, n INTEGER)"
        )
        conn.executemany(
            "INSERT INTO remap VALUES (?, ?, ?, ?)",
            [(r.segment_id, r.old_start, r.new_start, r.count) for r in runs],
        )

        conn.execute(
            "CREATE TABLE chunks_compacted (id INTEGER PRIMARY KEY, byte_offset INTEGER "
            "NOT NULL, byte_length INTEGER NOT NULL, content_hash INTEGER NOT NULL)"
        )
        conn.execute(
            "INSERT INTO chunks_compacted (id, byte_offset, byte_length, content_hash) "
            "SELECT c.id - r.old_start + r.new_start, c.byte_offset, c.byte_length, "
            "       c.content_hash "
            "FROM chunks c JOIN remap r "
            "  ON c.id >= r.old_start AND c.id < r.old_start + r.n"
        )
        # Failures reference chunks; anything whose chunk did not survive goes.
        conn.execute(
            "DELETE FROM chunk_failures WHERE chunk_id NOT IN "
            "(SELECT c.id FROM chunks c JOIN remap r "
            "   ON c.id >= r.old_start AND c.id < r.old_start + r.n)"
        )
        conn.execute(
            "UPDATE chunk_failures SET chunk_id = chunk_id - "
            "(SELECT r.old_start - r.new_start FROM remap r "
            "  WHERE chunk_failures.chunk_id >= r.old_start "
            "    AND chunk_failures.chunk_id < r.old_start + r.n)"
        )
        conn.execute("DROP TABLE chunks")
        conn.execute("ALTER TABLE chunks_compacted RENAME TO chunks")

        conn.execute(
            "UPDATE segments SET chunk_start = "
            "(SELECT r.new_start FROM remap r WHERE r.segment_id = segments.id) "
            "WHERE id IN (SELECT segment_id FROM remap)"
        )
        conn.execute("DROP TABLE remap")

        state["phase"] = LEXICAL
        db.set_meta(conn, STATE_KEY, json.dumps(state))


def remove_books(conn: sqlite3.Connection, book_ids: set[int]) -> int:
    """Forget these books.  Their chunks become dead space until compaction.

    Deliberately two steps: dropping the metadata is instant and reversible by
    re-adding the file, while reclaiming the ids rewrites every store. Doing
    them together would make `dyp remove` an operation nobody could take back.
    """
    if not book_ids:
        return 0
    marks = ",".join("?" * len(book_ids))
    ids = sorted(book_ids)
    with conn:
        conn.execute(
            f"DELETE FROM segments WHERE source_id IN "
            f"(SELECT id FROM sources WHERE book_id IN ({marks}))",
            ids,
        )
        conn.execute(f"DELETE FROM book_centroids WHERE book_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM sources WHERE book_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM books WHERE id IN ({marks})", ids)
    return len(ids)
