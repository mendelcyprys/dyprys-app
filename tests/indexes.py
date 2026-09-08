"""A small, fully embedded index without a GGUF on the machine.

`test_cli.py`, `test_expand.py` and `test_rerank.py` each open-code the same
five steps -- ingest, register a model, mark every segment embedded, write one
stub vector per chunk, backfill BM25 -- and the service and API tests need it
several times more. It lives here once so a schema change breaks one place
rather than six, and so a test that is *about* the service layer does not open
with twenty lines that are not.

The vectors are `StubEmbedder`'s, keyed off the chunk id exactly as the existing
tests key them. They carry no meaning, which is the point: they are
deterministic, and nothing here is measuring retrieval quality.
"""

from __future__ import annotations

from pathlib import Path

from dyprys import db
from dyprys.embed import store_for
from dyprys.ingest import ingest_paths
from dyprys.lexical import backfill
from tests.conftest import DIM, StubEmbedder

# One paragraph in `write_books` carries this and nothing else does. A verbatim
# search for it has exactly one right answer, which is what makes it usable as
# a fixed point either side of a refactor.
SENTINEL = "the corpuscle of Meissner answers only to itself"


def write_books(root: Path, count: int = 2, paragraphs: int = 30) -> list[Path]:
    """Books with enough paragraphs to chunk, and one unique phrase in book 0."""
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for n in range(count):
        body = [f"Paragraph {i} of book {n}, concerning neurons and synapses. " * 9
                for i in range(paragraphs)]
        if n == 0:
            body[paragraphs // 2] = SENTINEL + ". " + body[paragraphs // 2]
        path = root / f"book{n}.txt"
        path.write_text("\n\n".join(body), encoding="utf-8")
        written.append(path)
    return written


def embedded_index(directory: Path, books: list[Path], *, name: str = "stub",
                   embedder: StubEmbedder | None = None):
    """Open an index at `directory` holding `books`, embedded and searchable.

    Returns `(conn, embedder, model_id, store)`. Call it twice with different
    `name`s against one directory to get the multi-model index that `--model`
    ambiguity is about.
    """
    embedder = embedder or StubEmbedder()
    conn = db.connect(directory)
    ingest_paths(conn, books)
    model_id = db.model_id(conn, name, embedder.dim)
    store = store_for(conn, directory, model_id, embedder.dim)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model_id, seg["id"], seg["chunk_count"])
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    for chunk_id in range(1, total + 1):
        store.write(chunk_id, embedder.embed_query(f"chunk {chunk_id}"))
    store.flush()
    backfill(conn)
    return conn, embedder, model_id, store


def add_model(conn, directory: Path, name: str, embedder: StubEmbedder | None = None):
    """A second model over the same chunks, so `--model` has to be said."""
    embedder = embedder or StubEmbedder()
    model_id = db.model_id(conn, name, embedder.dim)
    store = store_for(conn, directory, model_id, embedder.dim)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model_id, seg["id"], seg["chunk_count"])
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    for chunk_id in range(1, total + 1):
        store.write(chunk_id, embedder.embed_query(f"chunk {chunk_id}"))
    store.flush()
    return model_id, store
