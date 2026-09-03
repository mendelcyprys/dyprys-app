"""Step 01: walk source files and record their chunk boundaries.

No embeddings here.  What this step fixes -- and what everything downstream
inherits -- is the schema, the contiguity of a book's chunk ids, and enough
information to avoid re-embedding text that did not change.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from dyprys import db, lexical
from dyprys.text import span_hash
from dyprys.chunker import (
    CHUNKER_VERSION,
    OVERLAP_BYTES,
    TARGET_BYTES,
    chunk_bytes,
)

# What counts as a book by default. Deliberately narrow: extraction from PDF or
# EPUB happens upstream, and silently accepting a format we cannot chunk well is
# worse than refusing it. Widen it per run with --ext.
TEXT_SUFFIXES = (".txt",)


class NothingToIngest(Exception):
    """No file matched. Carries what was there, so the message can be useful."""

    def __init__(self, roots: list[Path], wanted: tuple[str, ...]):
        self.roots, self.wanted = roots, wanted
        super().__init__("no matching files")

    def found_instead(self, limit: int = 6) -> list[tuple[str, int]]:
        """The extensions that *are* present, commonest first."""
        seen: dict[str, int] = {}

        def note(child: Path) -> None:
            if child.suffix:
                seen[child.suffix.lower()] = seen.get(child.suffix.lower(), 0) + 1

        for root in self.roots:
            if root.is_dir():
                for child in root.rglob("*"):
                    if child.is_file():
                        note(child)
            elif root.is_file():
                # A root can be a named file now that those are filtered too.
                # Missing it reported ".pdf" as "no files with an extension".
                note(root)
        return sorted(seen.items(), key=lambda kv: -kv[1])[:limit]


@dataclass(frozen=True)
class BookResult:
    """What happened to one book."""

    key: Path
    title: str
    status: str  # "added" | "unchanged" | "moved" | "rechunked" | "updated"
    sources: int
    chunks: int
    carried: int = 0  # chunks whose vectors survive the edit


def ingest_paths(
    conn: sqlite3.Connection,
    paths: list[Path],
    target: int = TARGET_BYTES,
    overlap: int = OVERLAP_BYTES,
    chapters: bool = False,
    deep: bool = False,
    suffixes: tuple[str, ...] = TEXT_SUFFIXES,
) -> list[BookResult]:
    """Ingest `paths`.

    By default each text file is a book.  With `chapters=True` each given
    directory is one book and the text files inside it are its chapters, in
    filename order -- the shape an EPUB extraction leaves behind.

    `deep` re-hashes every file rather than trusting size and mtime.
    """
    found = _collect(paths, chapters, suffixes)
    if not found:
        raise NothingToIngest([Path(p) for p in paths], suffixes)
    return [
        ingest_book(conn, key, files, target=target, overlap=overlap, deep=deep)
        for key, files in found
    ]


def acceptable(path: Path, suffixes: tuple[str, ...] = TEXT_SUFFIXES) -> bool:
    """May this named file be ingested?

    A file with no extension passes: the filter exists to refuse formats we know
    we cannot chunk -- .pdf, .epub, .docx -- and an extensionless file makes no
    such claim. Walking a directory is stricter and still requires a matching
    suffix, because a rule loose enough for a deliberate name would sweep up
    LICENSE and Makefile.
    """
    return not path.suffix or path.suffix.lower() in suffixes


def _collect(
    paths: list[Path], chapters: bool, suffixes: tuple[str, ...] = TEXT_SUFFIXES
) -> list[tuple[Path, list[Path]]]:
    """Group input paths into (book key, source files), sorted for repeatability."""
    books: list[tuple[Path, list[Path]]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        path = path.resolve()
        if chapters:
            if not path.is_dir():
                raise NotADirectoryError(f"--chapters needs directories, got {path}")
            files = sorted(_text_files(path, suffixes))
            if files:
                books.append((path, files))
        elif path.is_dir():
            books.extend((f, [f]) for f in sorted(_text_files(path, suffixes)))
        elif acceptable(path, suffixes):
            books.append((path, [path]))
        # A named file of the wrong kind is dropped rather than ingested. It
        # used to go straight through: `dyp add book.pdf` chunked the binary,
        # and `dyp add library/*` bypassed --ext entirely, because the shell
        # expands a glob to files and only the directory walk was filtered.
    # The same path given twice must not process its books twice.
    seen: dict[Path, list[Path]] = {}
    for key, files in books:
        seen.setdefault(key, files)
    return sorted(seen.items(), key=lambda b: b[0])


def _text_files(directory: Path, suffixes: tuple[str, ...] = TEXT_SUFFIXES) -> list[Path]:
    return [
        child
        for child in directory.rglob("*")
        if child.is_file() and child.suffix.lower() in suffixes
    ]


def ingest_book(
    conn: sqlite3.Connection,
    key: Path,
    files: list[Path],
    target: int = TARGET_BYTES,
    overlap: int = OVERLAP_BYTES,
    deep: bool = False,
) -> BookResult:
    """Ingest one book from one or more source files.  Idempotent."""
    title = key.stem

    with conn:
        chunking = db.chunking_id(conn, CHUNKER_VERSION, target, overlap)
        book = conn.execute("SELECT * FROM books WHERE key = ?", (str(key),)).fetchone()

        if book is None:
            loaded = _load(files)
            hashes = [_hash(data) for _, data, _ in loaded]

            # The same bytes at a new path is a move, not a new book. Relocating
            # touches paths only: identical content chunks identically, so every
            # offset, every vector and all embedding progress stay valid.
            moved = _find_moved_book(conn, hashes)
            if moved is not None:
                _relocate(conn, moved, key, title, files)
                n = _segment_totals(conn, moved, chunking) or 0
                return BookResult(key, title, "moved", len(files), n)

            book_id = conn.execute(
                "INSERT INTO books (key, title, added_at) VALUES (?, ?, ?)",
                (str(key), title, db.now()),
            ).lastrowid
            _write_sources(conn, book_id, loaded, hashes)
            n, _ = _allocate(conn, book_id, chunking, loaded, target, overlap)
            return BookResult(key, title, "added", len(files), n)

        book_id = book["id"]
        stored = conn.execute(
            "SELECT path, size_bytes, mtime, content_hash FROM sources "
            "WHERE book_id = ? ORDER BY ordinal",
            (book_id,),
        ).fetchall()
        same_paths = [str(f) for f in files] == [r["path"] for r in stored]

        # Size and mtime cannot prove a file unchanged, but they can save reading
        # 25 GB to discover that nothing moved. `deep` forces the real check.
        if same_paths and not deep and all(map(_stat_matches, files, stored)):
            existing = _segment_totals(conn, book_id, chunking)
            if existing is not None:
                return BookResult(key, title, "unchanged", len(files), existing)

        loaded = _load(files)
        hashes = [_hash(data) for _, data, _ in loaded]
        unchanged = same_paths and hashes == [r["content_hash"] for r in stored]

        if unchanged:
            existing = _segment_totals(conn, book_id, chunking)
            if existing is not None:
                return BookResult(key, title, "unchanged", len(files), existing)
            # Same text, a chunking it has not been split by before. Add it
            # alongside the others; nothing existing is disturbed.
            n, _ = _allocate(conn, book_id, chunking, loaded, target, overlap)
            return BookResult(key, title, "rechunked", len(files), n)

        # Content moved on. Every chunking of this book is stale, and the book's
        # id block must stay contiguous, so the whole book is re-allocated at the
        # end of the id space. Salvage makes that cheap: chunks whose text
        # survived the edit carry their vectors across to their new ids.
        old = _reusable_chunks(conn, book_id)
        lexical.drop_chunks(conn, _live_chunk_ids(conn, book_id))
        conn.execute(
            "DELETE FROM segments WHERE source_id IN "
            "(SELECT id FROM sources WHERE book_id = ?)",
            (book_id,),
        )
        conn.execute("DELETE FROM sources WHERE book_id = ?", (book_id,))
        _write_sources(conn, book_id, loaded, hashes)
        n, carried = _allocate(
            conn, book_id, chunking, loaded, target, overlap, salvage=old
        )
        return BookResult(key, title, "updated", len(files), n, carried)


def _find_moved_book(conn, hashes: list[str]) -> int | None:
    """A book with exactly these source hashes whose old paths are all gone.

    Requiring the old paths to be absent is what separates a move from a copy.
    If the original is still on disk, this is a second copy and deserves its own
    book; only a vanished original means the library merely rearranged itself.
    """
    if not hashes:
        return None
    candidates = conn.execute(
        "SELECT DISTINCT book_id FROM sources WHERE content_hash = ? AND ordinal = 0",
        (hashes[0],),
    ).fetchall()

    for candidate in candidates:
        stored = conn.execute(
            "SELECT path, content_hash FROM sources WHERE book_id = ? ORDER BY ordinal",
            (candidate["book_id"],),
        ).fetchall()
        if [row["content_hash"] for row in stored] != hashes:
            continue
        if any(Path(row["path"]).exists() for row in stored):
            continue  # the original is still there: a copy, not a move
        return candidate["book_id"]
    return None


def _relocate(conn, book_id: int, key: Path, title: str, files: list[Path]) -> None:
    """Point an existing book at its new location.  Chunks are not touched."""
    conn.execute(
        "UPDATE books SET key = ?, title = ? WHERE id = ?", (str(key), title, book_id)
    )
    stored = conn.execute(
        "SELECT id FROM sources WHERE book_id = ? ORDER BY ordinal", (book_id,)
    ).fetchall()
    conn.executemany(
        "UPDATE sources SET path = ?, mtime = ? WHERE id = ?",
        [
            (str(path), path.stat().st_mtime, row["id"])
            for row, path in zip(stored, files)
        ],
    )


def _load(files: list[Path]) -> list[tuple[Path, bytes, float]]:
    """Read each file, sampling mtime *before* the read, never after.

    Sampled after, a file edited mid-read would be recorded with an mtime
    describing content we never saw, and the size/mtime fast path would call it
    unchanged forever.  Sampled before, the same race records a stale mtime, the
    fast path sees a mismatch, and the file is re-checked -- wrong in the safe
    direction.
    """
    loaded = []
    for path in files:
        mtime = path.stat().st_mtime
        loaded.append((path, path.read_bytes(), mtime))
    return loaded


def _stat_matches(path: Path, stored) -> bool:
    try:
        info = path.stat()
    except OSError:
        return False  # missing, unreadable, or on a volume that is not mounted
    return info.st_size == stored["size_bytes"] and info.st_mtime == stored["mtime"]


def _write_sources(conn, book_id: int, loaded, hashes) -> None:
    conn.executemany(
        "INSERT INTO sources (book_id, ordinal, path, size_bytes, mtime, "
        "content_hash, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (book_id, ordinal, str(path), len(data), mtime, digest, db.now())
            for ordinal, ((path, data, mtime), digest) in enumerate(zip(loaded, hashes))
        ],
    )


def _allocate(
    conn, book_id: int, chunking: int, loaded, target: int, overlap: int, salvage=None
) -> tuple[int, int]:
    """Give this book's sources one consecutive block of chunk ids.

    Allocated per book and subdivided per source, so a segment, a source and the
    whole book are each a contiguous slice of the vector array.
    """
    sources = conn.execute(
        "SELECT id, path FROM sources WHERE book_id = ? ORDER BY ordinal", (book_id,)
    ).fetchall()
    by_path = {row["path"]: row["id"] for row in sources}

    next_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()[0] + 1
    total = carried = 0
    for path, data, _ in loaded:
        spans = chunk_bytes(data, target=target, overlap=overlap)
        rows = [
            (next_id + i, offset, length, _chunk_hash(data, offset, length))
            for i, (offset, length) in enumerate(spans)
        ]
        conn.executemany(
            "INSERT INTO chunks (id, byte_offset, byte_length, content_hash) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        # The bytes are already here, so the lexical index costs no extra read.
        lexical.index_chunks(
            conn,
            [
                (new_id, data[offset : offset + length].decode("utf-8", "replace"))
                for new_id, offset, length, _ in rows
            ],
        )
        conn.execute(
            "INSERT INTO segments (source_id, chunking_id, chunk_start, chunk_count) "
            "VALUES (?, ?, ?, ?)",
            (by_path[str(path)], chunking, next_id, len(spans)),
        )
        if salvage:
            pairs = []
            for new_id, _, _, digest in rows:
                hit = salvage.get(digest)
                if hit is not None:
                    old_id, models = hit
                    pairs.extend((model, new_id, old_id) for model in models)
            conn.executemany(
                "INSERT OR IGNORE INTO chunk_carry "
                "(model_id, new_chunk_id, old_chunk_id) VALUES (?, ?, ?)",
                pairs,
            )
            carried += len({new_id for _, new_id, _ in pairs})
        next_id += len(spans)
        total += len(spans)
    return total, carried


def _reusable_chunks(conn, book_id: int) -> dict[int, tuple[int, list[int]]]:
    """{content hash: (chunk id, models that already hold its vector)}.

    Only chunks some model has actually embedded are worth carrying, and which
    models those are must be captured now -- the progress rows recording it are
    about to be deleted along with their segments.
    """
    prefixes = {
        (row["model_id"], row["segment_id"]): row["n_embedded"]
        for row in conn.execute(
            "SELECT model_id, segment_id, n_embedded FROM segment_progress"
        )
    }
    models = [row["id"] for row in conn.execute("SELECT id FROM models")]
    rows = conn.execute(
        "SELECT c.id, c.content_hash, seg.id AS seg_id, seg.chunk_start FROM chunks c "
        "JOIN segments seg ON c.id >= seg.chunk_start "
        "                 AND c.id < seg.chunk_start + seg.chunk_count "
        "JOIN sources src ON src.id = seg.source_id "
        "WHERE src.book_id = ?",
        (book_id,),
    ).fetchall()

    reusable: dict[int, tuple[int, list[int]]] = {}
    for row in rows:
        have = [
            model
            for model in models
            if row["id"] < row["chunk_start"] + prefixes.get((model, row["seg_id"]), 0)
        ]
        if have:
            reusable[row["content_hash"]] = (row["id"], have)
    return reusable


def _live_chunk_ids(conn, book_id: int) -> list[int]:
    """Every chunk currently reachable from this book's segments."""
    return [
        row[0]
        for row in conn.execute(
            "SELECT c.id FROM chunks c "
            "JOIN segments seg ON c.id >= seg.chunk_start "
            "                 AND c.id < seg.chunk_start + seg.chunk_count "
            "JOIN sources src ON src.id = seg.source_id WHERE src.book_id = ?",
            (book_id,),
        )
    ]


def _segment_totals(conn, book_id: int, chunking: int) -> int | None:
    row = conn.execute(
        "SELECT SUM(seg.chunk_count) AS n FROM segments seg "
        "JOIN sources src ON src.id = seg.source_id "
        "WHERE src.book_id = ? AND seg.chunking_id = ?",
        (book_id, chunking),
    ).fetchone()
    return row["n"] if row and row["n"] is not None else None


def _hash(data: bytes) -> str:
    """SHA-256 of a source file's bytes, hex encoded."""
    return hashlib.sha256(data).hexdigest()


def _chunk_hash(data: bytes, offset: int, length: int) -> int:
    """The chunk signature: first 8 bytes of its SHA-256, as a signed integer.

    Identical text gives an identical hash, which is what lets an edited file
    keep the vectors of the passages the edit did not touch -- and what lets a
    query verify that the passage it is about to show is the one that was
    embedded.  Shared with `text.span_hash` so the two can never disagree.
    """
    return span_hash(data[offset : offset + length])
