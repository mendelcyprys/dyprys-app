"""Ingest writes the invariants the rest of the system relies on."""

import os
from pathlib import Path

import pytest

from dyprys import db
from dyprys.ingest import ingest_paths
from dyprys.text import read_span


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "index")
    yield connection
    connection.close()


def write_book(directory, name, paragraphs=120, tag=""):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n\n".join(f"Paragraph {n} of {name}.{tag} " * 15 for n in range(paragraphs)),
        encoding="utf-8",
    )
    return path


def pretend_embedded(conn, name="test-model", dim=768):
    """Mark everything currently ingested as embedded under one model.

    Salvage only offers a carry when some model actually holds the old vector,
    so an un-embedded library has nothing to save -- correctly.
    """
    model = db.model_id(conn, name, dim)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    return model


def all_chunks(conn):
    return conn.execute(
        "SELECT id, byte_offset, byte_length, content_hash FROM chunks ORDER BY id"
    ).fetchall()


# --- the basics --------------------------------------------------------------


def test_a_book_becomes_chunks(conn, tmp_path):
    result = ingest_paths(conn, [write_book(tmp_path, "book.txt")])[0]

    assert result.status == "added"
    assert result.chunks > 1
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == result.chunks


def test_offsets_read_back_as_the_original_text(conn, tmp_path):
    """The load-bearing claim of 'text lives in the source files'."""
    path = write_book(tmp_path, "book.txt")
    ingest_paths(conn, [path])

    source = path.read_text(encoding="utf-8")
    for row in all_chunks(conn):
        passage = read_span(path, row["byte_offset"], row["byte_length"])
        assert passage in source
        assert passage.strip() == passage


def test_re_ingesting_unchanged_files_does_nothing(conn, tmp_path):
    path = write_book(tmp_path, "book.txt")
    first = ingest_paths(conn, [path])[0]
    before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    second = ingest_paths(conn, [path])[0]

    assert second.status == "unchanged"
    assert second.chunks == first.chunks
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == before
    assert conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 1


def test_directories_are_walked_and_non_text_ignored(conn, tmp_path):
    library = tmp_path / "library"
    write_book(library, "one.txt")
    write_book(library / "nested", "two.txt")
    (library / "cover.jpg").write_bytes(b"\xff\xd8\xff")

    results = ingest_paths(conn, [library])

    assert sorted(r.title for r in results) == ["one", "two"]


def test_a_missing_path_is_an_error(conn, tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_paths(conn, [tmp_path / "nope"])


# --- contiguity --------------------------------------------------------------


def test_chunk_ids_are_dense_across_the_whole_index(conn, tmp_path):
    """Row index into the vector file is id - 1, so gaps would be holes."""
    ingest_paths(conn, [write_book(tmp_path, f"book{n}.txt") for n in range(3)])

    ids = [row["id"] for row in all_chunks(conn)]
    assert ids == list(range(1, len(ids) + 1))


def test_a_chunks_book_and_source_are_derived_not_stored(conn, tmp_path):
    paths = [write_book(tmp_path, f"book{n}.txt", paragraphs=40 + n) for n in range(4)]
    ingest_paths(conn, paths)

    for seg in conn.execute("SELECT * FROM segments").fetchall():
        for ordinal in (0, seg["chunk_count"] - 1):
            chunk_id = seg["chunk_start"] + ordinal
            found = db.locate(conn, chunk_id)
            assert found["segment_id"] == seg["id"]
            assert db.ordinal_in_segment(found, chunk_id) == ordinal


def test_no_segment_claims_a_chunk_past_the_end(conn, tmp_path):
    ingest_paths(conn, [write_book(tmp_path, "book.txt")])
    last = conn.execute("SELECT MAX(id) FROM chunks").fetchone()[0]

    assert db.locate(conn, last) is not None
    assert db.locate(conn, last + 1) is None


# --- chapters: one book, many source files -----------------------------------


def test_a_book_can_span_many_chapter_files(conn, tmp_path):
    """The EPUB shape: one work, one file per chapter."""
    book = tmp_path / "Some Long Book"
    for n in range(6):
        write_book(book, f"chapter{n:02d}.txt", paragraphs=30)

    result = ingest_paths(conn, [book], chapters=True)[0]

    assert result.status == "added"
    assert result.sources == 6
    assert conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 6
    assert result.title == "Some Long Book"


def test_chapters_keep_reading_order_in_the_vector_array(conn, tmp_path):
    book = tmp_path / "Ordered Book"
    for n in range(5):
        write_book(book, f"chapter{n:02d}.txt", paragraphs=25)
    ingest_paths(conn, [book], chapters=True)

    rows = conn.execute(
        "SELECT src.ordinal, seg.chunk_start FROM segments seg "
        "JOIN sources src ON src.id = seg.source_id ORDER BY src.ordinal"
    ).fetchall()
    assert [r["ordinal"] for r in rows] == [0, 1, 2, 3, 4]
    starts = [r["chunk_start"] for r in rows]
    assert starts == sorted(starts)  # chapter 2 follows chapter 1 in the array


def test_a_whole_multi_chapter_book_is_one_contiguous_slice(conn, tmp_path):
    """Routing may work per book or per chapter; both must be a plain slice."""
    for name in ("Book A", "Book B"):
        for n in range(4):
            write_book(tmp_path / name, f"chapter{n:02d}.txt", paragraphs=20)
    ingest_paths(conn, [tmp_path / "Book A", tmp_path / "Book B"], chapters=True)

    chunking = conn.execute("SELECT id FROM chunkings").fetchone()["id"]
    seen = []
    for book in conn.execute("SELECT id FROM books ORDER BY id"):
        start, count = db.book_range(conn, book["id"], chunking)
        ids = [
            r["id"]
            for r in conn.execute(
                "SELECT c.id FROM chunks c JOIN segments seg "
                "  ON c.id >= seg.chunk_start "
                " AND c.id < seg.chunk_start + seg.chunk_count "
                "JOIN sources src ON src.id = seg.source_id "
                "WHERE src.book_id = ? ORDER BY c.id",
                (book["id"],),
            )
        ]
        assert ids == list(range(start, start + count))  # no holes, no interleaving
        seen.append((start, count))
    assert seen[0][0] + seen[0][1] == seen[1][0]  # and the books do not overlap


def test_chapters_needs_a_directory(conn, tmp_path):
    path = write_book(tmp_path, "loose.txt")
    with pytest.raises(NotADirectoryError):
        ingest_paths(conn, [path], chapters=True)


# --- several chunkings of the same text --------------------------------------


def test_one_file_can_carry_several_chunkings_at_once(conn, tmp_path):
    """Models differ in context length, so they differ in chunk budget."""
    path = write_book(tmp_path, "book.txt")
    big = ingest_paths(conn, [path], target=3600, overlap=0)[0]
    small = ingest_paths(conn, [path], target=1200, overlap=200)[0]

    assert big.status == "added"
    assert small.status == "rechunked"
    assert small.chunks > big.chunks
    assert conn.execute("SELECT COUNT(*) FROM chunkings").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 1


def test_each_chunking_is_its_own_contiguous_block(conn, tmp_path):
    path = write_book(tmp_path, "book.txt")
    ingest_paths(conn, [path], target=3600)
    ingest_paths(conn, [path], target=1200)

    blocks = conn.execute(
        "SELECT chunking_id, chunk_start, chunk_count FROM segments ORDER BY chunk_start"
    ).fetchall()
    assert len(blocks) == 2
    first, second = blocks
    assert first["chunk_start"] + first["chunk_count"] == second["chunk_start"]
    assert first["chunking_id"] != second["chunking_id"]


def test_adding_a_chunking_leaves_the_other_untouched(conn, tmp_path):
    path = write_book(tmp_path, "book.txt")
    ingest_paths(conn, [path], target=3600)
    before = conn.execute(
        "SELECT id, byte_offset, byte_length FROM chunks ORDER BY id"
    ).fetchall()

    ingest_paths(conn, [path], target=1200)

    after = conn.execute(
        "SELECT id, byte_offset, byte_length FROM chunks WHERE id <= ? ORDER BY id",
        (len(before),),
    ).fetchall()
    assert [tuple(r) for r in before] == [tuple(r) for r in after]


def test_the_same_chunking_is_not_duplicated(conn, tmp_path):
    path = write_book(tmp_path, "book.txt")
    ingest_paths(conn, [path], target=1200, overlap=200)
    ingest_paths(conn, [path], target=1200, overlap=200)

    assert conn.execute("SELECT COUNT(*) FROM chunkings").fetchone()[0] == 1


# --- salvage across an edit --------------------------------------------------


def test_appending_to_a_book_keeps_almost_every_vector(conn, tmp_path):
    """The common case for a growing library: a new chapter on the end."""
    path = write_book(tmp_path, "book.txt", paragraphs=200)
    first = ingest_paths(conn, [path])[0]
    pretend_embedded(conn)

    path.write_text(
        path.read_text(encoding="utf-8") + "\n\n" + "A newly added tail. " * 400,
        encoding="utf-8",
    )
    second = ingest_paths(conn, [path])[0]

    assert second.status == "updated"
    assert second.chunks > first.chunks
    assert second.carried >= first.chunks * 0.9


def test_an_edit_that_does_not_move_text_keeps_almost_every_vector(conn, tmp_path):
    """A typo fix shifts nothing, so boundaries land in the same places."""
    path = write_book(tmp_path, "book.txt", paragraphs=200)
    ingest_paths(conn, [path])
    pretend_embedded(conn)

    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("Paragraph 100 of", "Paragraph 1OO of", 1), "utf-8")
    second = ingest_paths(conn, [path])[0]

    assert second.carried >= second.chunks * 0.9


def test_an_edit_that_moves_text_keeps_the_part_before_it(conn, tmp_path):
    """Boundaries are anchored to offsets, so a length change desynchronises
    everything after it. Salvage recovers the prefix; the tail is re-embedded.

    This is the one real limit of salvage, and it is a property of the chunker,
    not of the storage: content-defined boundaries would resynchronise within a
    chunk or two of the edit. Swapping in such a chunker is a new CHUNKER_VERSION
    and therefore a new row in `chunkings` -- not a migration.
    """
    path = write_book(tmp_path, "book.txt", paragraphs=200)
    ingest_paths(conn, [path])
    pretend_embedded(conn)

    paragraphs = path.read_text(encoding="utf-8").split("\n\n")
    paragraphs[180] = "Short."  # a big length change, 90% of the way in
    path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    late = ingest_paths(conn, [path])[0]

    assert 0.7 <= late.carried / late.chunks < 1.0


def test_carried_chunks_point_at_the_old_ids(conn, tmp_path):
    path = write_book(tmp_path, "book.txt", paragraphs=60)
    ingest_paths(conn, [path])
    pretend_embedded(conn)
    old = {r["content_hash"]: r["id"] for r in all_chunks(conn)}

    text = path.read_text(encoding="utf-8").replace("Paragraph 0 of", "CHANGED HERE", 1)
    path.write_text(text, encoding="utf-8")
    ingest_paths(conn, [path])

    carries = conn.execute("SELECT * FROM chunk_carry").fetchall()
    assert carries
    for carry in carries:
        new_hash = conn.execute(
            "SELECT content_hash FROM chunks WHERE id = ?", (carry["new_chunk_id"],)
        ).fetchone()["content_hash"]
        # a carry is only ever offered when the text is byte-identical
        assert old[new_hash] == carry["old_chunk_id"]


def test_a_rewritten_book_salvages_nothing(conn, tmp_path):
    path = write_book(tmp_path, "book.txt", paragraphs=60)
    ingest_paths(conn, [path])
    pretend_embedded(conn)

    write_book(tmp_path, "book.txt", paragraphs=60, tag=" ENTIRELY DIFFERENT.")
    second = ingest_paths(conn, [path])[0]

    assert second.status == "updated"
    assert second.carried == 0


def test_an_updated_book_stays_contiguous(conn, tmp_path):
    """Re-allocation is what keeps the invariant unconditional."""
    a = write_book(tmp_path, "a.txt", paragraphs=40)
    b = write_book(tmp_path, "b.txt", paragraphs=40)
    ingest_paths(conn, [a, b])

    write_book(tmp_path, "a.txt", paragraphs=90)  # a grows
    ingest_paths(conn, [a])

    chunking = conn.execute("SELECT id FROM chunkings").fetchone()["id"]
    for book in conn.execute("SELECT id FROM books"):
        start, count = db.book_range(conn, book["id"], chunking)
        found = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE id >= ? AND id < ?", (start, start + count)
        ).fetchone()[0]
        assert found == count


def test_offsets_are_correct_after_an_edit(conn, tmp_path):
    """Salvage must never leave a chunk pointing into the old file layout."""
    path = write_book(tmp_path, "book.txt", paragraphs=80)
    ingest_paths(conn, [path])

    text = path.read_text(encoding="utf-8")
    path.write_text("A new opening paragraph.\n\n" + text, encoding="utf-8")
    ingest_paths(conn, [path])

    source = path.read_text(encoding="utf-8")
    live = conn.execute(
        "SELECT c.byte_offset, c.byte_length FROM chunks c "
        "JOIN segments seg ON c.id >= seg.chunk_start "
        "                 AND c.id < seg.chunk_start + seg.chunk_count"
    ).fetchall()
    assert live
    for row in live:
        assert read_span(path, row["byte_offset"], row["byte_length"]) in source


# --- text fidelity -----------------------------------------------------------


def test_overlapping_chunks_still_read_back_exactly(conn, tmp_path):
    """Overlap is never consulted at read time -- offset and length suffice."""
    path = write_book(tmp_path, "book.txt")
    ingest_paths(conn, [path], target=1200, overlap=400)

    source = path.read_text(encoding="utf-8")
    rows = all_chunks(conn)
    for row in rows:
        assert read_span(path, row["byte_offset"], row["byte_length"]) in source
    assert any(
        b["byte_offset"] < a["byte_offset"] + a["byte_length"]
        for a, b in zip(rows, rows[1:])
    )


def test_non_ascii_text_survives_the_round_trip(conn, tmp_path):
    path = tmp_path / "greek.txt"
    path.write_text("\n\n".join(f"Νευρώνας {n} — σῆμα. " * 20 for n in range(60)), "utf-8")
    ingest_paths(conn, [path])

    source = path.read_text(encoding="utf-8")
    for row in all_chunks(conn):
        assert read_span(path, row["byte_offset"], row["byte_length"]) in source


def test_nothing_is_carried_when_nothing_was_embedded(conn, tmp_path):
    """A carry claims a vector exists. With no model, none does."""
    path = write_book(tmp_path, "book.txt", paragraphs=60)
    ingest_paths(conn, [path])

    path.write_text(path.read_text(encoding="utf-8") + "\n\nA tail.\n", encoding="utf-8")
    result = ingest_paths(conn, [path])[0]

    assert result.status == "updated"
    assert result.carried == 0
    assert conn.execute("SELECT COUNT(*) FROM chunk_carry").fetchone()[0] == 0


def test_a_carry_is_recorded_per_model_that_holds_the_vector(conn, tmp_path):
    """Two models, one fully embedded and one not: only the first gets carries."""
    path = write_book(tmp_path, "book.txt", paragraphs=80)
    ingest_paths(conn, [path])
    done = pretend_embedded(conn, "finished-model")
    partial = db.model_id(conn, "barely-started-model", 768)
    with conn:
        seg = conn.execute("SELECT id FROM segments").fetchone()["id"]
        db.set_embedded_prefix(conn, partial, seg, 3)  # only the first 3 chunks

    path.write_text(path.read_text(encoding="utf-8") + "\n\nA tail.\n", encoding="utf-8")
    ingest_paths(conn, [path])

    by_model = dict(
        conn.execute("SELECT model_id, COUNT(*) FROM chunk_carry GROUP BY model_id")
    )
    assert by_model[done] > by_model.get(partial, 0)
    assert by_model.get(partial, 0) <= 3


def test_an_unchanged_file_is_not_reread(conn, tmp_path, monkeypatch):
    """Size and mtime spare a 25 GB library from being hashed to learn nothing."""
    path = write_book(tmp_path, "book.txt")
    ingest_paths(conn, [path])

    def explode(*args, **kwargs):
        raise AssertionError("file was read despite unchanged size and mtime")

    monkeypatch.setattr(Path, "read_bytes", explode)
    assert ingest_paths(conn, [path])[0].status == "unchanged"


def test_deep_forces_the_real_check(conn, tmp_path):
    """A file edited without changing size or mtime is still caught by --deep."""
    path = write_book(tmp_path, "book.txt", paragraphs=60)
    ingest_paths(conn, [path])
    before = path.stat()

    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("Paragraph 7 of", "Paragraph 8 of", 1), encoding="utf-8")
    os.utime(path, (before.st_atime, before.st_mtime))  # disguise the edit

    assert ingest_paths(conn, [path])[0].status == "unchanged"   # fast path fooled
    assert ingest_paths(conn, [path], deep=True)[0].status == "updated"


# --- moving files ------------------------------------------------------------


def test_a_moved_file_keeps_its_chunks_and_its_embeddings(conn, tmp_path):
    """Rearranging a library must not cost a single embedding."""
    old = tmp_path / "old" / "book.txt"
    write_book(old.parent, "book.txt", paragraphs=80)
    ingest_paths(conn, [old])
    model = pretend_embedded(conn)
    before = [tuple(r) for r in all_chunks(conn)]
    embedded_before = conn.execute(
        "SELECT SUM(n_embedded) FROM segment_progress WHERE model_id = ?", (model,)
    ).fetchone()[0]

    new = tmp_path / "shelf" / "renamed.txt"
    new.parent.mkdir()
    old.rename(new)
    result = ingest_paths(conn, [new])[0]

    assert result.status == "moved"
    assert conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 1
    assert [tuple(r) for r in all_chunks(conn)] == before  # not one chunk rewritten
    assert conn.execute(
        "SELECT SUM(n_embedded) FROM segment_progress WHERE model_id = ?", (model,)
    ).fetchone()[0] == embedded_before
    assert conn.execute("SELECT COUNT(*) FROM chunk_carry").fetchone()[0] == 0


def test_a_moved_book_reads_from_its_new_path(conn, tmp_path):
    old = tmp_path / "old" / "book.txt"
    write_book(old.parent, "book.txt", paragraphs=60)
    ingest_paths(conn, [old])

    new = tmp_path / "new" / "book.txt"
    new.parent.mkdir()
    old.rename(new)
    ingest_paths(conn, [new])

    source = new.read_text(encoding="utf-8")
    for row in all_chunks(conn):
        assert read_span(new, row["byte_offset"], row["byte_length"]) in source
    assert conn.execute("SELECT path FROM sources").fetchone()["path"] == str(new)


def test_a_moved_multi_chapter_book_is_relocated_whole(conn, tmp_path):
    old = tmp_path / "Some Book"
    for n in range(5):
        write_book(old, f"chapter{n:02d}.txt", paragraphs=25)
    ingest_paths(conn, [old], chapters=True)
    before = [tuple(r) for r in all_chunks(conn)]

    new = tmp_path / "Shelf" / "Some Book"
    new.parent.mkdir()
    old.rename(new)
    result = ingest_paths(conn, [new], chapters=True)[0]

    assert result.status == "moved"
    assert result.sources == 5
    assert [tuple(r) for r in all_chunks(conn)] == before
    assert conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 1


def test_a_copy_is_a_new_book_not_a_move(conn, tmp_path):
    """If the original is still there, this is a second copy."""
    original = write_book(tmp_path, "book.txt", paragraphs=50)
    ingest_paths(conn, [original])

    duplicate = tmp_path / "elsewhere" / "book.txt"
    duplicate.parent.mkdir()
    duplicate.write_bytes(original.read_bytes())
    result = ingest_paths(conn, [duplicate])[0]

    assert result.status == "added"
    assert conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 2


def test_a_file_that_moved_and_changed_is_not_treated_as_a_move(conn, tmp_path):
    old = tmp_path / "old" / "book.txt"
    write_book(old.parent, "book.txt", paragraphs=50)
    ingest_paths(conn, [old])

    new = tmp_path / "new" / "book.txt"
    new.parent.mkdir()
    old.rename(new)
    new.write_text(new.read_text(encoding="utf-8") + "\n\nA tail.\n", encoding="utf-8")

    assert ingest_paths(conn, [new])[0].status == "added"


def test_the_same_path_given_twice_is_processed_once(conn, tmp_path):
    library = tmp_path / "library"
    write_book(library, "one.txt")
    write_book(library, "two.txt")

    results = ingest_paths(conn, [library, library])

    assert len(results) == 2
    assert all(r.status == "added" for r in results)


def test_mtime_is_recorded_from_before_the_read(conn, tmp_path):
    """Recorded after the read, a mid-read edit would look permanently unchanged."""
    path = write_book(tmp_path, "book.txt")
    ingest_paths(conn, [path])

    recorded = conn.execute("SELECT mtime FROM sources").fetchone()["mtime"]
    assert recorded <= path.stat().st_mtime
