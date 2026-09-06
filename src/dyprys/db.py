"""The SQLite store: schema, connection, and the derivations it guarantees.

The central modelling decision is that three things a naive schema conflates are
kept apart, because each varies independently of the others:

    book      the work -- what a reader names, what a result is attributed to
    source    a file on disk. An EPUB extracted per chapter gives one book many
              sources; a plain text dump gives one book one source
    chunking  a way of splitting text. Model A may want 900-token chunks and
              model B 512, over the very same file

A `segment` is where they meet: one source, split one way, occupying a
consecutive block of chunk ids.  Because ids are allocated per *book* and
subdivided across its sources, a segment, a source and a whole book are each a
contiguous slice of the vector array -- so routing may work at any of those
granularities without the storage changing.

What is not here: no text, and no embedding state on the chunk row.  Embedding
state is per model, because a library is routinely half-embedded under one model
and untouched under another.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 15

DEFAULT_DATA_DIR = Path("data")
DB_FILENAME = "dyprys.sqlite"

# An index is recognised by what is in it, not by what the file is called. The
# schema and vector format are stable across renames and across a fork (dyprys
# began as one), so rather than hardcode another tool's filename we find any
# .sqlite in the directory that carries these tables. The fast path never probes:
# a present DB_FILENAME wins outright, so this costs nothing in normal use.
_SIGNATURE_TABLES = frozenset({"books", "chunks", "segments", "models", "chunkings"})


def _is_index(path: Path) -> bool:
    """Does `path` look like one of our index databases, by its tables?

    A plain connection, not a read-only URI: the URI form breaks on a space in
    the name and cannot open a WAL-mode database whose sidecar files are absent,
    both of which a real index can have. `query_only` keeps the probe from
    writing to a file it may end up rejecting.
    """
    try:
        conn = sqlite3.connect(str(path))
        try:
            conn.execute("PRAGMA query_only = ON")
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return _SIGNATURE_TABLES <= names


def _index_candidates(directory: Path) -> list[Path]:
    """Existing .sqlite files in `directory` that are one of our indexes."""
    return sorted(p for p in Path(directory).glob("*.sqlite") if _is_index(p))


def db_path(directory: Path) -> Path:
    """The database file to open in `directory`.

    A present `dyprys.sqlite` wins immediately. Otherwise the directory is
    searched for a `.sqlite` that is actually one of our indexes -- verified by
    its tables, not its name -- so a renamed database, or one written by a
    sibling tool, is found without guessing blindly. With none present, the path
    a new index will be created at is returned.
    """
    directory = Path(directory)
    primary = directory / DB_FILENAME
    if primary.exists():
        return primary
    found = _index_candidates(directory)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        names = ", ".join(p.name for p in found)
        raise ValueError(
            f"{directory} holds more than one index database ({names}); "
            f"rename the one to keep to {DB_FILENAME}")
    return primary


def index_exists(directory: Path) -> bool:
    """Whether `directory` already holds an index this build can open."""
    directory = Path(directory)
    if (directory / DB_FILENAME).exists():
        return True
    return bool(_index_candidates(directory))

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- The work. What a search result is attributed to.
CREATE TABLE IF NOT EXISTS books (
    id       INTEGER PRIMARY KEY,
    key      TEXT    NOT NULL UNIQUE,  -- the file or directory that defines it
    title    TEXT    NOT NULL,
    added_at TEXT    NOT NULL
);

-- A file belonging to a book. One for a plain text dump; one per chapter for an
-- EPUB extraction. `ordinal` is reading order, so chapter 2 follows chapter 1
-- in the vector array as well as on the shelf.
CREATE TABLE IF NOT EXISTS sources (
    id           INTEGER PRIMARY KEY,
    book_id      INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    ordinal      INTEGER NOT NULL,
    path         TEXT    NOT NULL UNIQUE,
    size_bytes   INTEGER NOT NULL,
    -- Size and mtime are a cheap negative check: if either differs the file has
    -- certainly changed. Matching both means *probably* unchanged, so it is only
    -- ever used to skip re-hashing, never to conclude a file did change.
    mtime        REAL    NOT NULL,
    content_hash TEXT    NOT NULL,
    ingested_at  TEXT    NOT NULL,
    UNIQUE (book_id, ordinal)
);

-- A way of splitting text, deduplicated: most libraries have exactly one row.
CREATE TABLE IF NOT EXISTS chunkings (
    id              INTEGER PRIMARY KEY,
    chunker_version INTEGER NOT NULL,
    target          INTEGER NOT NULL,
    overlap         INTEGER NOT NULL,
    UNIQUE (chunker_version, target, overlap)
);

-- One source, split one way. The unit that owns a block of chunk ids.
CREATE TABLE IF NOT EXISTS segments (
    id          INTEGER PRIMARY KEY,
    source_id   INTEGER NOT NULL REFERENCES sources(id)   ON DELETE CASCADE,
    chunking_id INTEGER NOT NULL REFERENCES chunkings(id) ON DELETE CASCADE,
    chunk_start INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    UNIQUE (source_id, chunking_id)
);

-- Lets locate() find a chunk's owner by range in one indexed lookup.
CREATE INDEX IF NOT EXISTS segments_by_start ON segments(chunk_start);


CREATE TABLE IF NOT EXISTS chunks (
    -- id doubles as the row index into every model's vector file (row = id - 1).
    -- Ids are dense and allocated per book as one block. Never AUTOINCREMENT:
    -- it leaves gaps after a rollback, and a gap here is a hole in every file.
    id           INTEGER PRIMARY KEY,
    byte_offset  INTEGER NOT NULL,
    byte_length  INTEGER NOT NULL,
    -- First 8 bytes of the chunk's SHA-256, as a signed integer. When a source
    -- file is edited, chunks whose text is unchanged keep this hash, so their
    -- vectors can be copied forward instead of recomputed. Deliberately not
    -- indexed: salvage only ever compares one source's old chunks against its
    -- new ones, which is a few thousand rows held in memory.
    content_hash INTEGER NOT NULL
);

-- Vectors worth copying rather than recomputing. Written when a source file is
-- edited: chunks whose text survived the edit map from their new id to the old
-- one that already has a vector. Embedding consumes these in ascending id order
-- -- a copy where a row exists, a real embedding where it does not -- so the
-- prefix advances normally and one changed paragraph costs one chunk, not a book.
--
-- Keyed by model, and necessarily so: whether the old vector exists at all is a
-- per-model fact, and the progress row that recorded it is deleted along with
-- the old segment. The carry has to capture that fact while it is still known.
CREATE TABLE IF NOT EXISTS chunk_carry (
    model_id     INTEGER NOT NULL REFERENCES models(id)  ON DELETE CASCADE,
    new_chunk_id INTEGER NOT NULL REFERENCES chunks(id)  ON DELETE CASCADE,
    old_chunk_id INTEGER NOT NULL,
    PRIMARY KEY (model_id, new_chunk_id)
);

-- Lexical half of the search. Contentless: FTS5 keeps only the index, not the
-- text, which is what holds this to the 2-3 GB budgeted at 1,000 books instead
-- of a second copy of the library. rowid is the chunk id, so a BM25 hit and a
-- vector hit are the same kind of thing and can be fused without translation.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='',
    contentless_delete=1,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS models (
    id         INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL UNIQUE,  -- identity of the weights, not a nickname
    dim        INTEGER NOT NULL,
    -- How this model's vector file is stored: "fp32" or "int8". Recorded rather
    -- than inferred, because a file of the wrong assumed width still reads --
    -- it just returns nonsense, silently.
    quantisation TEXT  NOT NULL DEFAULT 'fp32',
    -- Enough to find and verify the weights again. Someone restoring a backup
    -- has the vectors but perhaps not the model that made them, and without
    -- these the only way to learn whether a candidate file is the right one is
    -- to embed with it and see.
    file_name   TEXT,
    file_bytes  INTEGER,
    file_sha256 TEXT,
    source_uri  TEXT,
    -- Where the weights were last opened from, so `dyp embed` can pick up
    -- without being told again. A hint, never trusted: the digest above is what
    -- says whether the file found there is still the right one.
    file_path   TEXT,
    -- Which chunking this model's vectors were made from. A model embeds one
    -- and only one: its vectors are a single population, and mixing two
    -- granularities into it means a search returns the same text twice at two
    -- sizes. NULL on models from before v14 and on any not yet embedded; bound
    -- on the first embed and refused a change after.
    chunking_id INTEGER REFERENCES chunkings(id),
    -- A nickname for the command line. `name` is the identity of the weights
    -- and cannot be a nickname; this is the short thing a person types.
    -- Uniqueness is the index below, not a column constraint: ALTER TABLE ADD
    -- COLUMN cannot carry one, so declaring it here would give a fresh index a
    -- rule that every migrated index quietly lacks.
    alias       TEXT,
    created_at TEXT    NOT NULL
);


-- Embedding progress, per model per segment. Chunks within a segment are
-- embedded in ascending order, so one count says exactly which are done:
-- ids [chunk_start, chunk_start + n_embedded). That keeps the embedded region
-- contiguous, and keeps this table at segments x models rows rather than
-- chunks x models.
CREATE TABLE IF NOT EXISTS segment_progress (
    model_id   INTEGER NOT NULL REFERENCES models(id)   ON DELETE CASCADE,
    segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    n_embedded INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (model_id, segment_id)
);

-- A chunk the model could not embed. Rare, and recorded rather than retried
-- forever: a zero vector is written so the prefix can advance, because one bad
-- chunk must not block the rest of its segment.
-- Stage 1 of routing: each book reduced to a handful of topical directions.
-- One centroid per book is too coarse -- a textbook spans many subjects -- so a
-- book is a small set of directions rather than one average. Rows live in a
-- per-model centroid file, contiguous per book, mirroring how chunks work.
CREATE TABLE IF NOT EXISTS book_centroids (
    model_id       INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    book_id        INTEGER NOT NULL REFERENCES books(id)  ON DELETE CASCADE,
    centroid_start INTEGER NOT NULL,
    centroid_count INTEGER NOT NULL,
    built_from     INTEGER NOT NULL,  -- chunks summarised, so staleness is visible
    PRIMARY KEY (model_id, book_id)
);

-- What has been done to this index, append-only. Not diagnostic logging -- for
-- a command-line tool the terminal is that, and `2> run.log` covers it. This is
-- provenance: a library kept for years accumulates surgery, and the only record
-- of it was whatever scrolled past at the time. Removing 113 duplicated books
-- and reclaiming 54,875 chunk ids left an index that afterwards looked like one
-- that had always been that shape.
--
-- It travels inside a backup, so someone restoring one can see what was done to
-- it before it reached them.
CREATE TABLE IF NOT EXISTS events (
    id     INTEGER PRIMARY KEY,
    at     TEXT NOT NULL,
    action TEXT NOT NULL,          -- add | remove | compact | route | ...
    detail TEXT NOT NULL           -- one line a person can read
);

-- One row per `dyp embed`, written when it stops. Embedding a library is a job
-- measured in days and run in pieces, so "how fast is this actually going" is a
-- question about the pieces together, not about the one in flight. It also
-- replaces a hardcoded 4.4 chunks/s in the time-remaining estimate: observed
-- rates on one machine in one day ranged from 1.4 to 6.3.
CREATE TABLE IF NOT EXISTS embed_runs (
    id         INTEGER PRIMARY KEY,
    model_id   INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    started_at TEXT    NOT NULL,
    seconds    REAL    NOT NULL,   -- monotonic, so it excludes machine sleep
    -- Wall clock over the same span. Two clocks rather than one because their
    -- difference is the only record of the machine having slept: monotonic
    -- alone cannot say when a run ended, and wall alone cannot say how much of
    -- the span was spent working. NULL on runs recorded before v12.
    wall_seconds REAL,
    embedded   INTEGER NOT NULL,
    -- Share of wall time this run spent working, 1.0 being flat out. The pause
    -- is between batches and proportional to the batch just done, so the
    -- working fraction *is* this number and a rate divided by it is the rate
    -- the machine would have managed unthrottled. NULL on runs before v15.
    duty       REAL,
    copied     INTEGER NOT NULL DEFAULT 0,
    failed     INTEGER NOT NULL DEFAULT 0,
    stopped    TEXT    NOT NULL    -- complete | time | limit | interrupted
);

-- What was asked, and what came back. Not needed to search, and nothing else
-- reads it: this is for the person, so a question worth asking twice does not
-- have to be reconstructed from memory, and so a good answer can be found
-- again. `detail` is JSON because the shape differs by what was switched on --
-- an expansion has three kinds of line, a summary has claims that verified and
-- claims that did not -- and inventing a column per option would mean a
-- migration every time an optional stage is added.
CREATE TABLE IF NOT EXISTS asked (
    id         INTEGER PRIMARY KEY,
    at         TEXT    NOT NULL,
    question   TEXT    NOT NULL,
    mode       TEXT    NOT NULL,
    ms         REAL    NOT NULL,
    detail     TEXT    NOT NULL     -- JSON: models, expansion, hits, summary
);

CREATE INDEX IF NOT EXISTS asked_by_time ON asked(at);

CREATE TABLE IF NOT EXISTS chunk_failures (
    model_id  INTEGER NOT NULL REFERENCES models(id)  ON DELETE CASCADE,
    chunk_id  INTEGER NOT NULL REFERENCES chunks(id)  ON DELETE CASCADE,
    reason    TEXT    NOT NULL,
    failed_at TEXT    NOT NULL,
    PRIMARY KEY (model_id, chunk_id)
);
"""


def data_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Where the index lives: `--data`, else $DYPRYS_DATA, else ./data."""
    if override is not None:
        return Path(override)
    env = os.environ.get("DYPRYS_DATA")
    return Path(env) if env else DEFAULT_DATA_DIR


def centroids_path(directory: Path, model_id: int) -> Path:
    """Stage 1's array: a few directions per book, one file per model."""
    return directory / f"centroids-{model_id:03d}.f32"


def vectors_path(directory: Path, model_id: int) -> Path:
    """One vector file per model, sized to the whole library.

    Chunk ids are unique across chunkings, so a single file per model holds
    vectors for every chunking without collision.  Untouched rows cost nothing:
    the file is sparse, so a second model is only as large as the part of the
    library actually embedded under it.
    """
    return directory / f"vectors-{model_id:03d}.f32"


def connect(directory: Path) -> sqlite3.Connection:
    """Open (creating if needed) the database under `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path(directory))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")   # embed writes while a query reads
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _check_schema_version(conn)
    # After the migration, not inside SCHEMA: on an older index the column this
    # indexes does not exist until _check_schema_version has added it. And it is
    # an index rather than a column constraint because ALTER TABLE ADD COLUMN
    # cannot carry UNIQUE -- declared on the column, a fresh index would get the
    # rule and every migrated one would quietly lack it.
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS models_by_alias ON models(alias)")
    conn.commit()
    return conn


def _check_schema_version(conn: sqlite3.Connection) -> None:
    found = get_meta(conn, "schema_version")
    if found is None:
        set_meta(conn, "schema_version", str(SCHEMA_VERSION))
    elif 4 <= int(found) < SCHEMA_VERSION:
        # v5 adds the lexical index, v6 the routing centroids, v7 a column
        # recording each model's vector format that defaults to the format every
        # older file already uses, v8 the weights provenance, v9 where those
        # weights were last seen and a nickname for them. Every step is a nullable
        # column or a default matching what older files already are, so an older
        # index keeps its vectors and its progress rather than being refused.
        _add_column(conn, "models", "quantisation", "TEXT NOT NULL DEFAULT 'fp32'")
        # v12 records wall-clock beside monotonic on an embed run, so a span
        # that includes machine sleep can be told from one that does not.
        _add_column(conn, "embed_runs", "wall_seconds", "REAL")
        # v14 binds a model to one chunking. NULL means unbound, which is what
        # every existing model is, and the first embed binds it -- so an index
        # with one chunking, which is nearly all of them, notices nothing.
        _add_column(conn, "models", "chunking_id", "INTEGER")
        # v15 records the duty a run was made at. Without it the rates of a
        # throttled run and a flat-out one are stored as though comparable, and
        # a median over both is a median of two different quantities. NULL
        # means "not recorded", which every older row is and which is read as
        # full speed -- the assumption those rows were already being used under.
        _add_column(conn, "embed_runs", "duty", "REAL")
        # v13 adds the `asked` table, which CREATE TABLE IF NOT EXISTS above has
        # already made; nothing to migrate, only the version to move.
        for column, spec in (("file_name", "TEXT"), ("file_bytes", "INTEGER"),
                             ("file_sha256", "TEXT"), ("source_uri", "TEXT"),
                             ("file_path", "TEXT"), ("alias", "TEXT")):
            _add_column(conn, "models", column, spec)
        set_meta(conn, "schema_version", str(SCHEMA_VERSION))
    elif int(found) != SCHEMA_VERSION:
        raise ValueError(
            f"index is schema v{found}, this build is v{SCHEMA_VERSION}. "
            f"Re-ingest into a fresh data directory."
        )


def _add_column(conn: sqlite3.Connection, table: str, column: str, spec: str) -> None:
    """Add a column if an older index does not have it yet."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {spec}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- derivations from the contiguity invariant -------------------------------


def locate(conn: sqlite3.Connection, chunk_id: int) -> sqlite3.Row | None:
    """Everything a result needs about `chunk_id`, found by range.

    The segment, its source file and the book it belongs to -- none of which is
    stored on the chunk row, so none of which can drift out of agreement with
    the id blocks.
    """
    return conn.execute(
        "SELECT seg.id AS segment_id, seg.chunk_start, seg.chunk_count, "
        "       seg.chunking_id, "
        "       src.id AS source_id, src.path, src.ordinal AS source_ordinal, "
        "       src.size_bytes, "
        "       b.id AS book_id, b.title, b.key "
        "FROM segments seg "
        "JOIN sources src ON src.id = seg.source_id "
        "JOIN books   b   ON b.id   = src.book_id "
        "WHERE seg.chunk_start <= ? AND seg.chunk_start + seg.chunk_count > ? "
        "ORDER BY seg.chunk_start DESC LIMIT 1",
        (chunk_id, chunk_id),
    ).fetchone()


def ordinal_in_segment(located: sqlite3.Row, chunk_id: int) -> int:
    """Position of `chunk_id` within its segment, counted from 0."""
    return chunk_id - located["chunk_start"]


def book_range(
    conn: sqlite3.Connection, book_id: int, chunking_id: int
) -> tuple[int, int] | None:
    """The contiguous chunk block a whole book occupies under one chunking.

    Routing may use this, or a single segment; the storage supports either
    without change, which is the §9 mitigation made cheap.
    """
    row = conn.execute(
        "SELECT MIN(seg.chunk_start) AS start, SUM(seg.chunk_count) AS n "
        "FROM segments seg JOIN sources src ON src.id = seg.source_id "
        "WHERE src.book_id = ? AND seg.chunking_id = ?",
        (book_id, chunking_id),
    ).fetchone()
    return (row["start"], row["n"]) if row and row["start"] is not None else None


# --- chunkings ---------------------------------------------------------------


def chunking_id(
    conn: sqlite3.Connection, chunker_version: int, target: int, overlap: int
) -> int:
    """Id for these chunk parameters, registering them on first sight."""
    key = (chunker_version, target, overlap)
    row = conn.execute(
        "SELECT id FROM chunkings WHERE chunker_version = ? AND target = ? "
        "AND overlap = ?",
        key,
    ).fetchone()
    if row is not None:
        return row["id"]
    return conn.execute(
        "INSERT INTO chunkings (chunker_version, target, overlap) VALUES (?, ?, ?)",
        key,
    ).lastrowid


# --- models and progress -----------------------------------------------------


def model_id(
    conn: sqlite3.Connection,
    name: str,
    dim: int,
    quantisation: str = "fp32",
    provenance: dict | None = None,
) -> int:
    """Id for `name`, registering it on first sight.  Dimension must not change.

    `provenance` records what the weights file was -- its name, size and full
    digest -- so someone holding only the vectors can tell whether a candidate
    download is the right one without embedding anything to find out.
    """
    row = conn.execute("SELECT id, dim FROM models WHERE name = ?", (name,)).fetchone()
    if row is not None:
        if row["dim"] != dim:
            raise ValueError(
                f"model {name!r} was registered with dim={row['dim']}, now given {dim}"
            )
        if provenance:
            # An index made before this was recorded learns it from any later
            # run that supplies the same weights.
            with conn:
                conn.execute(
                    "UPDATE models SET file_name = COALESCE(file_name, ?), "
                    "file_bytes = COALESCE(file_bytes, ?), "
                    "file_sha256 = COALESCE(file_sha256, ?) WHERE id = ?",
                    (provenance.get("file_name"), provenance.get("file_bytes"),
                     provenance.get("file_sha256"), row["id"]),
                )
        return row["id"]
    provenance = provenance or {}
    with conn:
        return conn.execute(
            "INSERT INTO models (name, dim, quantisation, file_name, file_bytes, "
            "file_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, dim, quantisation, provenance.get("file_name"),
             provenance.get("file_bytes"), provenance.get("file_sha256"), now()),
        ).lastrowid


def record_event(conn: sqlite3.Connection, action: str, detail: str) -> None:
    """Note something structural that was done to this index.

    Deliberately one line of prose rather than a payload to parse: this is read
    by a person asking why the index looks the way it does, months later.
    """
    with conn:
        conn.execute("INSERT INTO events (at, action, detail) VALUES (?, ?, ?)",
                     (now(), action, detail))


def record_run(conn: sqlite3.Connection, model_id: int, started: str,
               seconds: float, report, wall: float | None = None,
               duty: float | None = None) -> None:
    """Note what one embedding run achieved, once it has stopped.

    `seconds` is monotonic and `wall` is wall clock over the same span. Keeping
    both is the point: on a laptop that sleeps they diverge, and the difference
    is the only evidence in the index that it happened. With `seconds` alone a
    run cannot say when it ended -- a 30-minute run begun at 03:30 and finished
    at 05:10 looks identical to one that finished at 04:00.
    """
    with conn:
        conn.execute(
            "INSERT INTO embed_runs (model_id, started_at, seconds, wall_seconds, "
            "embedded, duty, copied, failed, stopped) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (model_id, started, seconds, wall, report.embedded, duty,
             report.copied, report.failed, report.stopped),
        )


def chunkings(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every way this library has been split, with how much uses each."""
    return conn.execute(
        "SELECT c.id, c.target, c.overlap, c.chunker_version, "
        "       COUNT(seg.id) AS segments, COALESCE(SUM(seg.chunk_count), 0) AS chunks "
        "FROM chunkings c LEFT JOIN segments seg ON seg.chunking_id = c.id "
        "GROUP BY c.id ORDER BY c.target DESC").fetchall()


def bound_chunking(conn: sqlite3.Connection, model_id: int) -> int | None:
    """Which chunking this model embeds, or None if it has not been bound."""
    row = conn.execute("SELECT chunking_id FROM models WHERE id = ?",
                       (model_id,)).fetchone()
    return row["chunking_id"] if row else None


def bind_chunking(conn: sqlite3.Connection, model_id: int, chunking_id: int) -> None:
    """Fix which chunking a model embeds.  Refuses to change an existing binding.

    A model's vectors are one population and every score in the system compares
    within it. Two granularities in one population means a search returns a
    passage and a piece of that same passage as separate results, which is not a
    thing a reader can be asked to interpret.
    """
    current = bound_chunking(conn, model_id)
    if current is not None and current != chunking_id:
        raise ValueError(
            f"this model already embeds chunking {current}; a model embeds one "
            f"chunking, so use a second model for chunking {chunking_id}")
    with conn:
        conn.execute("UPDATE models SET chunking_id = ? WHERE id = ?",
                     (chunking_id, model_id))


def record_question(conn: sqlite3.Connection, question: str, mode: str,
                    ms: float, detail: dict) -> int:
    """Keep what was asked and what came back.  Returns the row id."""
    import json

    with conn:
        cursor = conn.execute(
            "INSERT INTO asked (at, question, mode, ms, detail) VALUES (?, ?, ?, ?, ?)",
            (now(), question, mode, ms, json.dumps(detail, ensure_ascii=False)),
        )
    return cursor.lastrowid


def questions_asked(conn: sqlite3.Connection, limit: int = 20,
                    match: str | None = None) -> list[sqlite3.Row]:
    """Recent questions, newest first, optionally filtered by substring."""
    sql = "SELECT id, at, question, mode, ms, detail FROM asked"
    params: list = []
    if match:
        sql += " WHERE question LIKE ?"
        params.append(f"%{match}%")
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return conn.execute(sql, params).fetchall()


def forget_questions(conn: sqlite3.Connection, before: str | None = None,
                     which: int | None = None) -> int:
    """Delete stored questions.  Returns how many went."""
    with conn:
        if which is not None:
            cursor = conn.execute("DELETE FROM asked WHERE id = ?", (which,))
        elif before:
            cursor = conn.execute("DELETE FROM asked WHERE at < ?", (before,))
        else:
            cursor = conn.execute("DELETE FROM asked")
    return cursor.rowcount


def observed_rate(conn: sqlite3.Connection, model_id: int) -> float | None:
    """Median unthrottled chunks/s over recent runs, or None with no history.

    Median rather than mean: one run that spanned a laptop's idle sleep, or one
    that stopped two seconds after starting, should not move the estimate.

    Each run is divided by the duty it was made at first, because otherwise the
    median is taken over quantities that are not the same thing. `--duty` pauses
    between batches in proportion to the batch just done, so the share of time
    spent working *is* the duty and the division is exact rather than fitted --
    a run at 0.9 and a run at 1.0 differ by a factor the index knows. Left
    unnormalised, three real runs of one model read as 67.8, 69.5 and 93.7
    chunks/s and their median as 69.5, when the machine's rate was near enough
    constant and the spread was mostly the throttle being asked for.

    Returns the rate flat out, so an estimate built on it is what the work would
    take unthrottled. Multiply by a duty to predict a throttled run.
    """
    rates = []
    for row in conn.execute(
        "SELECT embedded, seconds, duty FROM embed_runs WHERE model_id = ? "
        "AND seconds > 30 AND embedded > 0 ORDER BY id DESC LIMIT 10",
        (model_id,),
    ):
        # NULL predates v15 and is read as full speed: those rows were already
        # being used that way, so this changes no existing estimate downwards.
        duty = row["duty"] if row["duty"] and 0 < row["duty"] <= 1 else 1.0
        rates.append(row["embedded"] / row["seconds"] / duty)
    if not rates:
        return None
    rates.sort()
    return rates[len(rates) // 2]


def remember_weights(conn: sqlite3.Connection, model_id: int, path: str | os.PathLike) -> None:
    """Note where these weights were opened from, so we need not be told again."""
    with conn:
        conn.execute("UPDATE models SET file_path = ? WHERE id = ?", (str(path), model_id))


def set_alias(conn: sqlite3.Connection, model_id: int, alias: str) -> None:
    """Give a model a short name to type.  `name` stays the weights identity."""
    with conn:
        conn.execute("UPDATE models SET alias = ? WHERE id = ?", (alias, model_id))


def find_model(conn: sqlite3.Connection, wanted: str) -> sqlite3.Row | None:
    """A model by alias, then by a unique substring of its name.

    Exact alias first: a nickname a person chose should never be ambiguous with
    part of a hash.
    """
    row = conn.execute("SELECT * FROM models WHERE alias = ?", (wanted,)).fetchone()
    if row is not None:
        return row
    matches = conn.execute(
        "SELECT * FROM models WHERE name LIKE '%' || ? || '%'", (wanted,)
    ).fetchall()
    return matches[0] if len(matches) == 1 else None


def sole_model(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """The only model in this index, if there is exactly one."""
    rows = conn.execute("SELECT * FROM models").fetchall()
    return rows[0] if len(rows) == 1 else None


def model_provenance(conn: sqlite3.Connection) -> list[dict]:
    """What each model's weights were, for anyone who has to obtain them."""
    return [
        {k: row[k] for k in row.keys()}
        for row in conn.execute(
            "SELECT name, dim, quantisation, file_name, file_bytes, file_sha256, "
            "source_uri FROM models ORDER BY id"
        )
    ]


def set_model_source(conn: sqlite3.Connection, model_id: int, uri: str) -> None:
    """Record where these weights can be obtained."""
    with conn:
        conn.execute("UPDATE models SET source_uri = ? WHERE id = ?", (uri, model_id))


def model_quantisation(conn: sqlite3.Connection, model_id: int) -> str:
    """How this model's vectors are stored.  The file does not say; this does."""
    row = conn.execute(
        "SELECT quantisation FROM models WHERE id = ?", (model_id,)
    ).fetchone()
    return (row["quantisation"] if row else None) or "fp32"


def embedded_prefix(conn: sqlite3.Connection, model: int, segment: int) -> int:
    """How many of `segment`'s chunks are embedded under `model`."""
    row = conn.execute(
        "SELECT n_embedded FROM segment_progress WHERE model_id = ? AND segment_id = ?",
        (model, segment),
    ).fetchone()
    return row["n_embedded"] if row else 0


def set_embedded_prefix(
    conn: sqlite3.Connection, model: int, segment: int, n: int
) -> None:
    conn.execute(
        "INSERT INTO segment_progress (model_id, segment_id, n_embedded) "
        "VALUES (?, ?, ?) ON CONFLICT(model_id, segment_id) "
        "DO UPDATE SET n_embedded = excluded.n_embedded",
        (model, segment, n),
    )


# --- meta --------------------------------------------------------------------


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
