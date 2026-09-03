"""What is in the index, and what it would cost to remove.

Read-only inspection, plus the one removal that is genuinely clean. Books cannot
be dropped yet: their chunk ids sit inside a dense range that other books'
offsets depend on, so removing one is compaction, not deletion. A *model* is a
different matter -- its vectors live in their own file and nothing else points at
them -- which is exactly the property per-model files were chosen for.
"""

from __future__ import annotations

import sqlite3

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from dyprys import db


def _disk(path: Path) -> tuple[int, int]:
    """(apparent, actually allocated) bytes, which differ for a sparse file."""
    if not path.exists():
        return 0, 0
    info = path.stat()
    return info.st_size, info.st_blocks * 512


@dataclass
class ModelInfo:
    id: int
    name: str
    dim: int
    quantisation: str
    embedded: int
    live_chunks: int
    failures: int
    carries: int
    centroid_books: int
    stale_books: int
    vectors_apparent: int
    vectors_actual: int
    centroids_actual: int
    alias: str | None = None      # the short thing you can type for --model
    file_path: str | None = None  # where the weights were last opened from

    @property
    def coverage(self) -> float:
        return self.embedded / self.live_chunks if self.live_chunks else 0.0

    @property
    def bytes_on_disk(self) -> int:
        return self.vectors_actual + self.centroids_actual


def models(conn: sqlite3.Connection, directory: Path) -> list[ModelInfo]:
    from dyprys.routing import stale_books

    live = conn.execute("SELECT COALESCE(SUM(chunk_count), 0) FROM segments").fetchone()[0]
    out = []
    for row in conn.execute(
        "SELECT id, name, dim, quantisation, alias, file_path, chunking_id "
        "FROM models ORDER BY id"
    ):
        # Against the chunking this model is bound to, not the library. A model
        # embeds one chunking, so dividing its progress by every chunk in the
        # library is the same mistake `dyp watch` made -- a per-model numerator
        # over a library-wide denominator, which reads as under 100% for a model
        # that is finished and can exceed it when another model is further on.
        mine = live
        if row["chunking_id"] is not None:
            mine = conn.execute(
                "SELECT COALESCE(SUM(chunk_count), 0) FROM segments WHERE chunking_id = ?",
                (row["chunking_id"],)).fetchone()[0]
        one = lambda sql: conn.execute(sql, (row["id"],)).fetchone()[0]
        apparent, actual = _disk(db.vectors_path(directory, row["id"]))
        out.append(
            ModelInfo(
                id=row["id"],
                name=row["name"],
                dim=row["dim"],
                quantisation=row["quantisation"] or "fp32",
                embedded=one("SELECT COALESCE(SUM(n_embedded),0) FROM segment_progress WHERE model_id=?"),
                live_chunks=mine,
                failures=one("SELECT COUNT(*) FROM chunk_failures WHERE model_id=?"),
                carries=one("SELECT COUNT(*) FROM chunk_carry WHERE model_id=?"),
                centroid_books=one("SELECT COUNT(*) FROM book_centroids WHERE model_id=?"),
                stale_books=stale_books(conn, row["id"]),
                vectors_apparent=apparent,
                vectors_actual=actual,
                centroids_actual=_disk(db.centroids_path(directory, row["id"]))[1],
                alias=row["alias"],
                file_path=row["file_path"],
            )
        )
    return out


def drop_model(conn: sqlite3.Connection, directory: Path, model_id: int) -> list[Path]:
    """Remove a model, its progress, and its files.  Chunks are untouched.

    Everything that depends on a model cascades from the `models` row, so the
    library itself -- books, chunks, boundaries, the lexical index -- survives
    intact. That is the whole point of keeping vectors in per-model files.
    """
    removed = []
    for path in (db.vectors_path(directory, model_id), db.centroids_path(directory, model_id)):
        if path.exists():
            path.unlink()
            removed.append(path)
    with conn:
        conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
    return removed


@dataclass
class SourceInfo:
    ordinal: int
    path: str
    size_bytes: int
    present: bool


@dataclass
class ChunkingInfo:
    id: int
    target: int
    overlap: int
    chunks: int


@dataclass
class BookInfo:
    id: int
    title: str
    key: str
    sources: list[SourceInfo] = field(default_factory=list)
    chunkings: list[ChunkingInfo] = field(default_factory=list)
    lexical: int = 0
    per_model: dict[str, int] = field(default_factory=dict)

    @property
    def chunks(self) -> int:
        return sum(c.chunks for c in self.chunkings)


def books(conn: sqlite3.Connection, pattern: str | None = None) -> list[BookInfo]:
    """Every book, or those matching `pattern`, with what covers it."""
    from dyprys.search import scope_books

    wanted = scope_books(conn, pattern) if pattern else None

    # Everything grouped in one pass each, rather than a handful of queries per
    # book. Unfiltered, `dyp books` used to take 8 minutes on 3,453 books and
    # looked like a hang: the lexical count alone was 139.5 ms a book, because
    # it joined the whole 284,627-row FTS index against segments on a *range*
    # and no index can serve that. 3,453 x 284,627 is a billion comparisons.
    rows = conn.execute("SELECT id, title, key FROM books ORDER BY title").fetchall()
    keep = [r for r in rows if wanted is None or r["id"] in wanted]
    ids = {r["id"] for r in keep}
    books_by_id = {r["id"]: BookInfo(id=r["id"], title=r["title"], key=r["key"]) for r in keep}

    for src in conn.execute(
        "SELECT book_id, ordinal, path, size_bytes FROM sources ORDER BY book_id, ordinal"
    ):
        book = books_by_id.get(src["book_id"])
        if book is not None:
            book.sources.append(SourceInfo(src["ordinal"], src["path"], src["size_bytes"],
                                           Path(src["path"]).exists()))

    for ch in conn.execute(
        "SELECT src.book_id AS book_id, ch.id, ch.target, ch.overlap, "
        "       SUM(seg.chunk_count) AS n "
        "FROM segments seg JOIN chunkings ch ON ch.id = seg.chunking_id "
        "JOIN sources src ON src.id = seg.source_id "
        "GROUP BY src.book_id, ch.id ORDER BY src.book_id, ch.id"
    ):
        book = books_by_id.get(ch["book_id"])
        if book is not None:
            book.chunkings.append(ChunkingInfo(ch["id"], ch["target"], ch["overlap"], ch["n"]))

    names = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM models ORDER BY id")}
    for book in books_by_id.values():
        for name in names.values():
            book.per_model[name] = 0
    for row in conn.execute(
        "SELECT src.book_id AS book_id, p.model_id, SUM(p.n_embedded) AS done "
        "FROM segment_progress p JOIN segments seg ON seg.id = p.segment_id "
        "JOIN sources src ON src.id = seg.source_id GROUP BY src.book_id, p.model_id"
    ):
        book = books_by_id.get(row["book_id"])
        if book is not None and row["model_id"] in names:
            book.per_model[names[row["model_id"]]] = row["done"]

    # A segment's chunks are a contiguous id block and the FTS rowid *is* the
    # chunk id, so this is a rowid range scan the index can serve -- which is
    # the contiguity invariant paying for itself.
    for seg in conn.execute(
        "SELECT src.book_id AS book_id, seg.chunk_start, seg.chunk_count "
        "FROM segments seg JOIN sources src ON src.id = seg.source_id"
    ).fetchall():
        book = books_by_id.get(seg["book_id"])
        if book is None:
            continue
        book.lexical += conn.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE rowid >= ? AND rowid < ?",
            (seg["chunk_start"], seg["chunk_start"] + seg["chunk_count"]),
        ).fetchone()[0]

    return [books_by_id[r["id"]] for r in keep]


def chunkings(conn: sqlite3.Connection) -> list[ChunkingInfo]:
    return [
        ChunkingInfo(row["id"], row["target"], row["overlap"], row["n"] or 0)
        for row in conn.execute(
            "SELECT ch.id, ch.target, ch.overlap, SUM(seg.chunk_count) AS n "
            "FROM chunkings ch LEFT JOIN segments seg ON seg.chunking_id = ch.id "
            "GROUP BY ch.id ORDER BY ch.id"
        )
    ]


def quantise_model(
    conn: sqlite3.Connection, directory: Path, model_id: int, progress=None
) -> tuple[int, int]:
    """Rewrite a model's fp32 vector file as int8.  Returns (before, after) bytes.

    Written to a new file and swapped in, rather than converted in place: the
    output is a different width from the input, so there is no ordering that
    makes an in-place rewrite safe, and a crash half way would leave a file that
    reads without error and returns nonsense.
    """
    from dyprys.vectors import FP32, INT8, VectorStore

    row = conn.execute("SELECT dim, quantisation FROM models WHERE id = ?", (model_id,)).fetchone()
    if row is None:
        raise ValueError(f"no model {model_id}")
    if (row["quantisation"] or FP32) != FP32:
        raise ValueError("that model is already quantised")

    source = db.vectors_path(directory, model_id)
    if not source.exists():
        raise ValueError("that model has no vector file yet")

    dim = row["dim"]
    rows = source.stat().st_size // (dim * 4)
    before = source.stat().st_blocks * 512
    staging = source.with_suffix(".int8.partial")
    staging.unlink(missing_ok=True)

    old = VectorStore(source, dim=dim, rows=rows, quantisation=FP32)
    new = VectorStore(staging, dim=dim, rows=max(rows, 1), quantisation=INT8)
    try:
        block = 8192
        for start in range(0, rows, block):
            span = min(block, rows - start)
            new.write_many(start + 1, old.slice(start + 1, span))
            if progress:
                progress(start + span, rows)
        new.flush()
    finally:
        old.close()
        new.close()

    after = staging.stat().st_blocks * 512
    # The database row and the file must change together, or the next reader
    # opens one format expecting the other.
    with conn:
        conn.execute("UPDATE models SET quantisation = ? WHERE id = ?", (INT8, model_id))
        staging.replace(source)
    return before, after


def truncate_model(
    conn: sqlite3.Connection, directory: Path, model_id: int, dim: int, progress=None
) -> tuple[int, int]:
    """Keep only the first `dim` components of a model's vectors.

    **Only sound for a model trained with Matryoshka representation learning**,
    where the dimensions are ordered by importance so a prefix is still a usable
    embedding. EmbeddingGemma is; most models are not, and on one that is not
    this quietly destroys the index. There is no way to check from the weights,
    so it asks.

    Measured on 117 books / 53,567 chunks, against the full 768:

        dims   store   recall@1   recall@5   paired p vs 768
         768   41.4MB   77/110     97/110    --
         512   27.6MB   75/110     96/110    0.727  -- no difference
         256   13.9MB   67/110     98/110    0.053  -- borderline, probably real

    **Repeated on 3,453 books / 284,627 chunks, 5.3x larger, and it holds:**

        dims   store   flat r@1   flat r@5   routed r@1   paired p vs 768
         768   220MB    78/110     99/110      68/110     --
         512   147MB    74/110     97/110      74/110     0.508
         256    74MB    70/110     97/110      65/110     0.454

    Neither loss reaches significance at either scale, and both point the same
    way and by about the same amount -- 512 costs two to four questions at
    recall@1, 256 costs eight to ten. recall@5 barely moves at all (99 -> 97 ->
    97) and routing recall not at all (109 -> 110 -> 107), so what truncation
    costs is ranking precision rather than retrieval: the shape of a cheap first
    stage with something over the top of it.

    So 512 is a third of the storage for nothing measurable, at both scales.

    Rewritten to a new file and swapped in, for the same reason quantisation is:
    the output is a different width from the input.
    """
    from dyprys.vectors import VectorStore, bytes_per_row

    row = conn.execute("SELECT dim, quantisation FROM models WHERE id = ?",
                       (model_id,)).fetchone()
    if row is None:
        raise ValueError(f"no model {model_id}")
    if dim >= row["dim"]:
        raise ValueError(f"that model is already {row['dim']} dimensions")

    source = db.vectors_path(directory, model_id)
    if not source.exists():
        raise ValueError("that model has no vector file yet")
    quantisation = row["quantisation"] or "fp32"
    # From the same function the store itself uses. This line used to compute
    # `dim * 4 + 4` for int8, which is the fp32 width plus a scale rather than
    # the int8 width -- so on an int8 index it found a quarter of the rows and
    # silently dropped the other three quarters, irreversibly. An int8 row is
    # one byte per dimension and a float32 scale.
    rows = source.stat().st_size // bytes_per_row(quantisation, row["dim"])
    before = source.stat().st_blocks * 512

    staging = source.with_suffix(".trunc.partial")
    staging.unlink(missing_ok=True)
    old_store = VectorStore(source, dim=row["dim"], rows=rows, quantisation=quantisation)
    new_store = VectorStore(staging, dim=dim, rows=max(rows, 1), quantisation=quantisation)
    try:
        block = 4096
        for start in range(0, rows, block):
            span = min(block, rows - start)
            kept = old_store.slice(start + 1, span)[:, :dim]
            # Renormalise: a prefix of a unit vector is not a unit vector, and
            # every score in this system is a dot product that assumes one.
            norms = np.linalg.norm(kept, axis=1, keepdims=True)
            new_store.write_many(start + 1, np.divide(
                kept, norms, out=np.zeros_like(kept), where=norms > 0))
            if progress:
                progress(start + span, rows)
        new_store.flush()
    finally:
        old_store.close()
        new_store.close()

    after = staging.stat().st_blocks * 512
    with conn:
        conn.execute("UPDATE models SET dim = ? WHERE id = ?", (dim, model_id))
        # The centroids are in the old width and describe the old space.
        conn.execute("DELETE FROM book_centroids WHERE model_id = ?", (model_id,))
        db.centroids_path(directory, model_id).unlink(missing_ok=True)
        staging.replace(source)
    return before, after


def relocate(conn: sqlite3.Connection, old: str, new: str) -> tuple[int, int]:
    """Rewrite a path prefix across the whole index.  Returns (books, sources).

    Individual books that move are recognised by content hash at ingest, but a
    whole library moving -- a different disk, a different machine, a restored
    backup -- would otherwise be N separate rediscoveries. Chunk offsets are
    into the bytes of a file, so they survive the move untouched; only the
    strings that say where the file is need to change.
    """
    old, new = str(old).rstrip("/"), str(new).rstrip("/")
    # Matched by substring, not by LIKE. `_` and `%` are LIKE wildcards, and
    # `my_books` is an ordinary directory name -- it silently matched
    # `myXbooks`, redirecting a library that was never named. Escaping would fix
    # that but not the second problem: a prefix is not a path, so
    # `/Volumes/my_books` also matched `/Volumes/my_booksXTRA`. The boundary has
    # to be a separator, or the whole string.
    under = f"{old}/"
    with conn:
        books = conn.execute(
            "UPDATE books SET key = ? || substr(key, ?) "
            "WHERE key = ? OR substr(key, 1, ?) = ?",
            (new, len(old) + 1, old, len(under), under),
        ).rowcount
        sources = conn.execute(
            "UPDATE sources SET path = ? || substr(path, ?) "
            "WHERE path = ? OR substr(path, 1, ?) = ?",
            (new, len(old) + 1, old, len(under), under),
        ).rowcount
    return books, sources


def duplicate_titles(conn: sqlite3.Connection, limit: int = 10) -> list[tuple[str, int]]:
    """Titles held by more than one book, commonest first.

    Content hashing catches the same *bytes* arriving twice, which is what a
    moved or re-added file looks like. It cannot catch the same *book* arriving
    twice from two extractions: run a PDF through a different converter and
    every byte differs, so every chunk hash differs, and the library quietly
    holds the work twice.

    That is not hypothetical. Building the 2,341-book corpus, 113 books were
    extracted a second time from the PDFs the first 117 came from — 54,875
    chunks, 40% of the outstanding embedding, silently duplicated. A title
    collision would have said so in a second. It is a warning and not a refusal
    because two books can legitimately share a title.
    """
    return [
        (row["title"], row["n"])
        for row in conn.execute(
            "SELECT title, COUNT(*) AS n FROM books GROUP BY title "
            "HAVING n > 1 ORDER BY n DESC, title LIMIT ?",
            (limit,),
        )
    ]


def unreadable_sources(conn: sqlite3.Connection, limit: int = 5) -> list[str]:
    """Source paths that are not on disk — what a bad relocation looks like."""
    missing = []
    for row in conn.execute("SELECT path FROM sources ORDER BY path"):
        if not Path(row["path"]).exists():
            missing.append(row["path"])
            if len(missing) >= limit:
                break
    return missing
