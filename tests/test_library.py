"""Inspecting the index, and the one removal that is clean."""

import numpy as np
import pytest

from dyprys import db
from dyprys.embed import store_for
from dyprys.ingest import ingest_paths
from dyprys.library import books, chunkings, drop_model, models

DIM = 8


@pytest.fixture
def stocked(conn, tmp_path, library):
    """A library with a model, some progress, and a second chunking."""
    ingest_paths(conn, [library], target=1200, overlap=200)
    model = db.model_id(conn, "stub-model@abc", DIM)
    store = store_for(conn, tmp_path / "index", model, DIM)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    for chunk_id in range(1, conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] + 1):
        v = np.zeros(DIM, dtype=np.float32); v[chunk_id % DIM] = 1.0
        store.write(chunk_id, v)
    store.flush()
    return model, tmp_path / "index"


# --- models ------------------------------------------------------------------


def test_a_model_reports_its_coverage_and_disk(conn, stocked):
    model, directory = stocked
    info = models(conn, directory)[0]

    assert info.name == "stub-model@abc"
    assert info.coverage == 1.0
    assert info.bytes_on_disk > 0


def test_disk_reports_what_is_allocated_not_what_is_addressable(conn, tmp_path, library):
    """Sparse files are the reason a second model is affordable."""
    model = db.model_id(conn, "m", DIM)
    store_for(conn, tmp_path / "index", model, DIM)   # sized, nothing written
    info = models(conn, tmp_path / "index")[0]

    assert info.vectors_apparent >= info.vectors_actual


def test_an_unrouted_model_says_so(conn, stocked):
    model, directory = stocked
    assert models(conn, directory)[0].centroid_books == 0


def test_dropping_a_model_leaves_the_library_intact(conn, stocked):
    """The property per-model files were chosen for."""
    model, directory = stocked
    chunks_before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    lexical_before = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

    removed = drop_model(conn, directory, model)

    assert removed and all(not p.exists() for p in removed)
    assert conn.execute("SELECT COUNT(*) FROM models").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == chunks_before
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == lexical_before


def test_dropping_a_model_takes_its_progress_with_it(conn, stocked):
    model, directory = stocked
    assert conn.execute("SELECT COUNT(*) FROM segment_progress").fetchone()[0] > 0

    drop_model(conn, directory, model)

    assert conn.execute("SELECT COUNT(*) FROM segment_progress").fetchone()[0] == 0


def test_one_model_can_be_dropped_without_touching_another(conn, stocked):
    model, directory = stocked
    other = db.model_id(conn, "keep-me", DIM)
    store_for(conn, directory, other, DIM)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, other, seg["id"], 3)

    drop_model(conn, directory, model)

    survivors = models(conn, directory)
    assert [m.name for m in survivors] == ["keep-me"]
    assert survivors[0].embedded > 0


# --- books -------------------------------------------------------------------


def test_every_book_is_listed_with_what_covers_it(conn, stocked):
    found = books(conn)

    assert len(found) == 2
    for book in found:
        assert book.sources
        assert book.chunks > 0
        assert book.lexical == book.chunks          # ingest indexes inline
        assert sum(book.per_model.values()) > 0


def test_a_pattern_narrows_to_one_book(conn, stocked):
    assert [b.title for b in books(conn, "book1")] == ["book1"]
    assert books(conn, "nothing-like-this") == []


def test_both_chunkings_of_a_book_are_shown(conn, stocked):
    """One file can carry two chunkings; the view must not hide one."""
    book = books(conn, "book0")[0]

    assert len(book.chunkings) == 2
    assert {c.target for c in book.chunkings} == {3600, 1200}
    assert book.chunks == sum(c.chunks for c in book.chunkings)


def test_progress_is_measured_against_the_chunking_the_model_embeds(
    conn, tmp_path, library
):
    """A model bound to one chunking is finished when *that* one is done.

    Divided by the book's chunks across every chunking instead, a model that
    has embedded all of its own reads as part-way for ever, and can never
    reach "all" in a library split more than one way.
    """
    ingest_paths(conn, [library], target=1200, overlap=200)
    coarse, fine = (r["id"] for r in conn.execute("SELECT id FROM chunkings ORDER BY id"))
    model = db.model_id(conn, "stub-model@abc", DIM)
    db.bind_chunking(conn, model, coarse)
    with conn:
        for seg in conn.execute(
            "SELECT id, chunk_count FROM segments WHERE chunking_id = ?", (coarse,)
        ).fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])

    book = books(conn, "book0")[0]
    mine = next(c.chunks for c in book.chunkings if c.id == coarse)

    assert book.live_for("stub-model@abc") == mine
    assert book.live_for("stub-model@abc") < book.chunks, "the fixture must be split two ways"
    assert book.per_model["stub-model@abc"] == mine, "every chunk of its own chunking"
    # The property the display depends on: done >= live, so it renders "all".
    assert book.per_model["stub-model@abc"] >= book.live_for("stub-model@abc")


def test_an_unbound_model_falls_back_to_the_whole_book(conn, stocked):
    """Bound to nothing (pre-v14, or never embedded), the book total is right."""
    model, _ = stocked
    with conn:
        conn.execute("UPDATE models SET chunking_id = NULL WHERE id = ?", (model,))

    book = books(conn, "book0")[0]
    assert book.live_for("stub-model@abc") == book.chunks


def test_a_library_without_notes_reports_none(conn, tmp_path):
    """Absence is the normal case and must not be an error."""
    from dyprys.library import notes_path

    assert notes_path(tmp_path) is None


def test_notes_are_found_and_NOTES_wins_over_README(conn, tmp_path):
    """A corpus may already have a README for people; NOTES.md is for searchers."""
    from dyprys.library import notes_path

    (tmp_path / "README.md").write_text("what this collection is", encoding="utf-8")
    assert notes_path(tmp_path).name == "README.md"

    (tmp_path / "NOTES.md").write_text("-c cuts along the shelf, not the topic",
                                       encoding="utf-8")
    assert notes_path(tmp_path).name == "NOTES.md"


def test_a_directory_named_like_a_notes_file_is_not_one(conn, tmp_path):
    """`is_file`, not `exists` — a directory called NOTES.md cannot be read."""
    from dyprys.library import notes_path

    (tmp_path / "NOTES.md").mkdir()
    assert notes_path(tmp_path) is None


def test_a_missing_source_file_is_flagged(conn, stocked, library):
    (library / "book0.txt").unlink()

    book = books(conn, "book0")[0]
    assert any(not s.present for s in book.sources)
    assert all(s.present for s in books(conn, "book1")[0].sources)


def test_chunkings_are_listed_with_their_share(conn, stocked):
    found = chunkings(conn)
    total = conn.execute("SELECT COALESCE(SUM(chunk_count), 0) FROM segments").fetchone()[0]

    assert len(found) == 2
    assert sum(c.chunks for c in found) == total


# --- Matryoshka truncation ---------------------------------------------------


def test_truncating_keeps_the_prefix_and_renormalises(conn, tmp_path):
    """A prefix of a unit vector is not a unit vector, and every score here is a
    dot product that assumes one."""
    import numpy as np
    from dyprys.embed import store_for
    from dyprys.library import truncate_model
    from dyprys.vectors import VectorStore

    model = db.model_id(conn, "mrl", 16)
    with conn:
        conn.execute("INSERT INTO chunks (id, byte_offset, byte_length, content_hash) "
                     "VALUES (1, 0, 1, 0), (2, 0, 1, 0)")
    store = store_for(conn, tmp_path, model, 16)
    raw = np.random.default_rng(0).normal(size=(2, 16)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    store.write_many(1, raw)
    store.close()

    truncate_model(conn, tmp_path, model, 8)

    assert conn.execute("SELECT dim FROM models WHERE id=?", (model,)).fetchone()["dim"] == 8
    after = VectorStore(db.vectors_path(tmp_path, model), dim=8, rows=2)
    for i in range(2):
        got = after.read(i + 1)
        want = raw[i][:8] / np.linalg.norm(raw[i][:8])
        assert np.allclose(np.linalg.norm(got), 1.0, atol=1e-3)
        assert np.allclose(got, want, atol=2e-2)


def test_truncating_drops_the_routing_profile(conn, tmp_path):
    """Centroids are in the old width and describe the old space."""
    import numpy as np
    from dyprys.embed import store_for
    from dyprys.library import truncate_model

    model = db.model_id(conn, "mrl", 16)
    with conn:
        conn.execute("INSERT INTO chunks (id, byte_offset, byte_length, content_hash) "
                     "VALUES (1, 0, 1, 0)")
        conn.execute("INSERT INTO books (id, key, title, added_at) VALUES (1,'k','t','n')")
        conn.execute("INSERT INTO book_centroids (model_id, book_id, centroid_start, "
                     "centroid_count, built_from) VALUES (?, 1, 1, 4, 1)", (model,))
    store = store_for(conn, tmp_path, model, 16)
    store.write_many(1, np.ones((1, 16), dtype=np.float32) / 4)
    store.close()
    db.centroids_path(tmp_path, model).write_bytes(b"x" * 64)

    truncate_model(conn, tmp_path, model, 8)

    assert conn.execute("SELECT COUNT(*) FROM book_centroids").fetchone()[0] == 0
    assert not db.centroids_path(tmp_path, model).exists()


def test_truncating_upwards_is_refused(conn, tmp_path):
    from dyprys.library import truncate_model

    model = db.model_id(conn, "mrl", 16)
    with conn:
        conn.execute("INSERT INTO chunks (id, byte_offset, byte_length, content_hash) "
                     "VALUES (1, 0, 1, 0)")
    from dyprys.embed import store_for
    store_for(conn, tmp_path, model, 16).close()

    with pytest.raises(ValueError, match="already"):
        truncate_model(conn, tmp_path, model, 32)


# --- listing every book must not be quadratic --------------------------------


def test_listing_reports_the_lexical_count_per_book(conn, tmp_path, library):
    """The count that made `dyp books` look like a hang, pinned for correctness.

    It joined the whole FTS index against segments on a *range* once per book:
    139.5 ms a book, 8 minutes over 3,453 books. Counted per segment as a rowid
    range instead, which the index can serve, because a segment's chunks are a
    contiguous block and the FTS rowid is the chunk id.
    """
    from dyprys.lexical import drop_chunks

    listed = books(conn)
    assert listed and all(b.lexical == sum(c.chunks for c in b.chunkings) for b in listed)

    victim = listed[0]
    start = conn.execute(
        "SELECT seg.chunk_start FROM segments seg JOIN sources src ON src.id = seg.source_id "
        "WHERE src.book_id = ? ORDER BY seg.chunk_start", (victim.id,)).fetchone()["chunk_start"]
    with conn:
        drop_chunks(conn, [start, start + 1])

    again = {b.id: b for b in books(conn)}
    assert again[victim.id].lexical == victim.lexical - 2
    for other in listed[1:]:
        assert again[other.id].lexical == other.lexical, "another book's count moved"


def test_the_work_per_book_does_not_grow_with_the_library(conn, tmp_path, library):
    """The shape that made it quadratic, measured as work rather than queries.

    The old version ran the *same number* of statements per book, so counting
    statements cannot see the bug: the cost was inside one of them, a scan of
    the whole FTS index once per book. SQLite's progress handler fires every N
    virtual-machine instructions, which does see it.
    """
    from dyprys.ingest import ingest_paths

    def work():
        ticks = [0]
        def tick():
            ticks[0] += 1
            return 0
        conn.set_progress_handler(tick, 50)
        try:
            listed = books(conn)
        finally:
            conn.set_progress_handler(None, 0)
        return ticks[0], len(listed)

    small_work, small_books = work()

    more = tmp_path / "more"
    more.mkdir()
    for n in range(12):
        (more / f"extra{n}.txt").write_text(
            "\n\n".join(f"Para {i} of extra {n}. " * 12 for i in range(8)), encoding="utf-8")
    ingest_paths(conn, [more])

    big_work, big_books = work()

    assert big_books > small_books * 3, "the fixture did not actually grow"
    small_rate = small_work / small_books
    big_rate = big_work / big_books
    assert big_rate <= small_rate * 1.5, (
        f"{small_rate:.0f} units of work per book at {small_books} books, "
        f"{big_rate:.0f} at {big_books} — per-book work must not rise with the library"
    )


def test_filtering_still_selects_the_same_books(conn, tmp_path, library):
    everything = {b.id for b in books(conn)}
    one = books(conn, "book0")

    assert len(one) == 1
    assert one[0].id in everything
    assert books(conn, "nothing-matches-this") == []


def test_truncating_an_int8_model_keeps_every_vector(conn, tmp_path):
    """It used to keep a quarter of them, irreversibly.

    The row count was computed as `dim * 4 + 4` — the fp32 width plus a scale —
    when an int8 row is one byte per dimension plus a float32 scale. On a
    284,627-vector index that found 71,434 rows and silently discarded 213,193.
    """
    import numpy as np

    from dyprys import db as _db
    from dyprys.library import truncate_model
    from dyprys.vectors import VectorStore, bytes_per_row

    model = _db.model_id(conn, "matryoshka@aaaaaaaaaaaa", 64, "int8")
    path = _db.vectors_path(tmp_path, model)
    store = VectorStore(path, dim=64, rows=500, quantisation="int8")
    rng = np.random.default_rng(0)
    for row in range(1, 501):
        vector = rng.normal(size=64).astype(np.float32)
        store.write(row, vector / np.linalg.norm(vector))
    store.flush()
    store.close()

    truncate_model(conn, tmp_path, model, 32)

    assert path.stat().st_size == 500 * bytes_per_row("int8", 32)
    kept = VectorStore(path, dim=32, rows=500, quantisation="int8")
    assert abs(float(np.linalg.norm(kept.read(500))) - 1.0) < 0.02, \
        "the last vector was never written"


def test_a_truncated_index_can_still_be_queried(conn, tmp_path):
    """The saving is worthless if nothing can open the index afterwards.

    `--truncate` shortened the stored vectors and nothing shortened the query,
    so the model row said 512, the embedder said 768, and every command refused.
    """
    import numpy as np

    from dyprys.embedder import Truncated

    class Wide:
        name, dim = "wide@aaaaaaaaaaaa", 8

        def embed_query(self, text):
            return np.arange(8, dtype=np.float32) / np.linalg.norm(np.arange(8))

        def embed_documents(self, texts):
            return np.stack([self.embed_query(t) for t in texts])

        def provenance(self):
            return {}

    cut = Truncated(Wide(), 4)
    query = cut.embed_query("anything")

    assert query.shape == (4,)
    assert abs(float(np.linalg.norm(query)) - 1.0) < 1e-5, \
        "a prefix of a unit vector is not a unit vector"
    assert cut.embed_documents(["a", "b"]).shape == (2, 4)
