"""Schema guarantees: multiple models, per-model progress, and derivations."""

import pytest

from dyprys import db
from dyprys.ingest import ingest_paths


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "index")
    yield connection
    connection.close()


@pytest.fixture
def library(conn, tmp_path):
    """Three ingested books, so there is something to track progress against."""
    for n in range(3):
        (tmp_path / f"book{n}.txt").write_text(
            "\n\n".join(f"Paragraph {i} of book {n}. " * 15 for i in range(80)),
            encoding="utf-8",
        )
    ingest_paths(conn, [tmp_path])
    return conn


def segments(conn):
    return [r["id"] for r in conn.execute("SELECT id FROM segments ORDER BY id")]


GEMMA = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf"
OTHER = "hf:some-org/other-embed-model/other-Q8_0.gguf"


def test_registering_a_model_is_idempotent(conn):
    first = db.model_id(conn, GEMMA, 768)
    assert db.model_id(conn, GEMMA, 768) == first


def test_two_models_coexist_with_separate_vector_files(conn, tmp_path):
    a = db.model_id(conn, GEMMA, 768)
    b = db.model_id(conn, OTHER, 1024)

    assert a != b
    assert db.vectors_path(tmp_path, a) != db.vectors_path(tmp_path, b)
    assert conn.execute("SELECT COUNT(*) FROM models").fetchone()[0] == 2


def test_a_model_cannot_change_dimension(conn):
    db.model_id(conn, GEMMA, 768)
    with pytest.raises(ValueError, match="dim=768"):
        db.model_id(conn, GEMMA, 1024)


def test_progress_is_tracked_per_model_independently(library):
    """The case that matters: half-embedded under one model, untouched under another."""
    conn = library
    fast, slow = db.model_id(conn, GEMMA, 768), db.model_id(conn, OTHER, 1024)
    segs = segments(conn)

    with conn:
        for seg in segs:  # fully embedded under the first model
            count = conn.execute(
                "SELECT chunk_count FROM segments WHERE id = ?", (seg,)
            ).fetchone()[0]
            db.set_embedded_prefix(conn, fast, seg, count)
        db.set_embedded_prefix(conn, slow, segs[0], 5)  # barely started on the second

    assert db.embedded_prefix(conn, slow, segs[0]) == 5
    assert db.embedded_prefix(conn, slow, segs[1]) == 0
    assert db.embedded_prefix(conn, fast, segs[1]) > 0


def test_two_chunkings_of_one_book_embed_independently(library, tmp_path):
    """Model A may want 900-token chunks and model B 512, over the same text."""
    conn = library
    ingest_paths(conn, [tmp_path], target=1200, overlap=200)  # a second chunking
    big, small = (
        r["id"] for r in conn.execute("SELECT id FROM chunkings ORDER BY id")
    )
    a, b = db.model_id(conn, GEMMA, 768), db.model_id(conn, OTHER, 1024)

    seg_big = conn.execute(
        "SELECT id, chunk_count FROM segments WHERE chunking_id = ? LIMIT 1", (big,)
    ).fetchone()
    seg_small = conn.execute(
        "SELECT id, chunk_count FROM segments WHERE chunking_id = ? LIMIT 1", (small,)
    ).fetchone()
    with conn:
        db.set_embedded_prefix(conn, a, seg_big["id"], seg_big["chunk_count"])
        db.set_embedded_prefix(conn, b, seg_small["id"], seg_small["chunk_count"])

    # each model is complete on its own chunking and absent from the other
    assert db.embedded_prefix(conn, a, seg_big["id"]) == seg_big["chunk_count"]
    assert db.embedded_prefix(conn, a, seg_small["id"]) == 0
    assert db.embedded_prefix(conn, b, seg_small["id"]) == seg_small["chunk_count"]
    assert db.embedded_prefix(conn, b, seg_big["id"]) == 0


def test_an_interrupted_segment_resumes_at_the_right_chunk(library):
    conn = library
    model = db.model_id(conn, GEMMA, 768)
    seg = conn.execute("SELECT * FROM segments ORDER BY id LIMIT 1").fetchone()

    with conn:
        db.set_embedded_prefix(conn, model, seg["id"], 7)

    next_chunk = seg["chunk_start"] + db.embedded_prefix(conn, model, seg["id"])
    assert next_chunk == seg["chunk_start"] + 7
    assert next_chunk < seg["chunk_start"] + seg["chunk_count"]


def test_dropping_a_model_drops_only_its_progress(library):
    conn = library
    keep, drop = db.model_id(conn, GEMMA, 768), db.model_id(conn, OTHER, 1024)
    seg = segments(conn)[0]
    with conn:
        db.set_embedded_prefix(conn, keep, seg, 4)
        db.set_embedded_prefix(conn, drop, seg, 9)

    with conn:
        conn.execute("DELETE FROM models WHERE id = ?", (drop,))

    assert db.embedded_prefix(conn, keep, seg) == 4
    assert conn.execute("SELECT COUNT(*) FROM segment_progress").fetchone()[0] == 1
    # chunks are untouched -- they belong to the library, not to any model
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] > 0


def test_a_failed_chunk_is_recorded_against_its_model(library):
    conn = library
    model = db.model_id(conn, GEMMA, 768)
    with conn:
        conn.execute(
            "INSERT INTO chunk_failures (model_id, chunk_id, reason, failed_at) "
            "VALUES (?, 1, 'tokeniser overflow', '2026-08-30')",
            (model,),
        )
    assert conn.execute("SELECT COUNT(*) FROM chunk_failures").fetchone()[0] == 1


def test_an_index_from_an_older_schema_is_refused(tmp_path):
    conn = db.connect(tmp_path / "index")
    with conn:
        db.set_meta(conn, "schema_version", "1")
    conn.close()

    with pytest.raises(ValueError, match="schema v1"):
        db.connect(tmp_path / "index")


def test_a_v8_index_gains_the_new_columns_without_losing_anything(tmp_path):
    """Migrations are additive: an older index opens and keeps its work.

    Verified on the live 3,453-book index mid-embed — schema 8 to 9 with 99,612
    chunks of progress untouched — but pinned here so it stays true.
    """
    directory = tmp_path / "old"
    conn = db.connect(directory)
    model = db.model_id(conn, "gemma@aaaaaaaaaaaa", 768)
    with conn:
        conn.execute("INSERT INTO books (id, key, title, added_at) VALUES (1,'k','t','n')")
        conn.execute("INSERT INTO sources (id, book_id, ordinal, path, size_bytes, mtime, "
                     "content_hash, ingested_at) VALUES (1,1,0,'p',1,1,'h','n')")
        conn.execute("INSERT INTO chunkings (id, chunker_version, target, overlap) "
                     "VALUES (1,1,3600,0)")
        conn.execute("INSERT INTO segments (id, source_id, chunking_id, chunk_start, "
                     "chunk_count) VALUES (1,1,1,1,42)")
        db.set_embedded_prefix(conn, model, 1, 42)
        # Pretend this index predates the two newest columns.
        conn.execute("DROP INDEX IF EXISTS models_by_alias")
        conn.execute("ALTER TABLE models DROP COLUMN file_path")
        conn.execute("ALTER TABLE models DROP COLUMN alias")
        db.set_meta(conn, "schema_version", "8")
    conn.close()

    conn = db.connect(directory)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(models)")}
        assert {"file_path", "alias"} <= columns
        assert db.get_meta(conn, "schema_version") == str(db.SCHEMA_VERSION)
        assert conn.execute(
            "SELECT COALESCE(SUM(n_embedded), 0) FROM segment_progress").fetchone()[0] == 42
        assert conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 1
    finally:
        conn.close()


def test_structural_changes_are_recorded_for_later(conn):
    """An index kept for years accumulates surgery; the only record of it used
    to be whatever scrolled past at the time."""
    db.record_event(conn, "remove", "113 duplicated books, 54,875 chunks reclaimable")
    db.record_event(conn, "compact", "reclaimed 54,875 chunk ids")

    rows = conn.execute("SELECT action, detail FROM events ORDER BY id").fetchall()

    assert [r["action"] for r in rows] == ["remove", "compact"]
    assert "54,875" in rows[1]["detail"]
    assert conn.execute("SELECT at FROM events LIMIT 1").fetchone()["at"]


def test_the_journal_survives_a_backup_and_restore(tmp_path):
    """It is provenance, so it has to travel with the index it describes."""
    from dyprys.backup import restore, write

    source = tmp_path / "src"
    conn = db.connect(source)
    db.record_event(conn, "compact", "reclaimed 54,875 chunk ids")
    conn.commit()
    archive = tmp_path / "out.tar.gz"
    write(conn, source, archive, include_sources=False)
    conn.close()

    restore(archive, tmp_path / "restored")

    back = db.connect(tmp_path / "restored")
    try:
        row = back.execute("SELECT action, detail FROM events").fetchone()
        assert row["action"] == "compact" and "54,875" in row["detail"]
    finally:
        back.close()


def _build_index(directory):
    """A real index in `directory`; returns its chunk count."""
    directory.mkdir(parents=True, exist_ok=True)
    book = directory / "b.txt"
    book.write_text("Paragraph about neurons. " * 300, encoding="utf-8")
    c = db.connect(directory)
    ingest_paths(c, [book])
    n = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    c.close()
    return n


def test_an_index_is_found_by_its_tables_whatever_it_is_named(tmp_path):
    """The database is recognised by content, not filename — so a renamed index,
    or one a fork or sibling tool wrote, opens in place with no reference to any
    other project's name."""
    src = tmp_path / "built"
    n = _build_index(src)
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    (src / db.DB_FILENAME).rename(moved / "some-old-backup.sqlite")   # arbitrary name

    assert db.index_exists(moved)
    assert db.db_path(moved).name == "some-old-backup.sqlite"
    reopened = db.connect(moved)
    assert reopened.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == n
    reopened.close()


def test_a_stray_non_index_sqlite_is_not_mistaken_for_one(tmp_path):
    """The guess is checked: a random .sqlite lacking our tables is ignored, so it
    is never opened as though it were an index."""
    import sqlite3

    junk = sqlite3.connect(tmp_path / "notes.sqlite")
    junk.execute("CREATE TABLE todo (id INTEGER, task TEXT)")
    junk.commit(); junk.close()

    assert not db.index_exists(tmp_path)
    assert db.db_path(tmp_path).name == db.DB_FILENAME   # would create ours, not adopt the junk


def test_a_present_dyprys_db_wins_without_a_search(tmp_path):
    _build_index(tmp_path)
    assert db.db_path(tmp_path).name == db.DB_FILENAME


def test_two_indexes_in_one_directory_is_refused_clearly(tmp_path):
    """Ambiguity is an error, not a silent pick of the wrong one."""
    a = tmp_path / "a"; _build_index(a)
    b = tmp_path / "b"; _build_index(b)
    (a / db.DB_FILENAME).rename(tmp_path / "one.sqlite")
    (b / db.DB_FILENAME).rename(tmp_path / "two.sqlite")

    with pytest.raises(ValueError, match="more than one index"):
        db.db_path(tmp_path)


def test_a_new_index_is_created_under_our_own_name(tmp_path):
    assert not db.index_exists(tmp_path)
    db.connect(tmp_path).close()
    assert (tmp_path / db.DB_FILENAME).exists()
    assert db.db_path(tmp_path).name == db.DB_FILENAME
