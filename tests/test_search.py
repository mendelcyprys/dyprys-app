"""Flat search: exact, and honest about what has actually been embedded."""

import numpy as np
import pytest

from dyprys import db
from dyprys.embed import store_for
from dyprys.search import _span_of, embedded_ranges, flat_search, resolve
from dyprys.text import CHANGED, SHIFTED

DIM = 8


@pytest.fixture
def searchable(conn, tmp_path, library):
    """A model with every chunk embedded, and vectors we chose ourselves."""
    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "index", model, DIM)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    # Every chunk gets its own direction, so ranking is fully determined and no
    # two chunks can tie for first place.
    rng = np.random.default_rng(0)
    for chunk_id in range(1, total + 1):
        vector = rng.normal(size=DIM).astype(np.float32)
        store.write(chunk_id, vector / np.linalg.norm(vector))
    store.flush()
    return model, store, total


def test_it_returns_the_nearest_chunks_best_first(conn, searchable):
    model, store, total = searchable
    query = store.read(9).copy()

    hits = flat_search(conn, store, query, model, k=5)

    assert hits[0].chunk_id == 9
    assert np.isclose(hits[0].score, 1.0, atol=1e-6)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_k_is_respected(conn, searchable):
    model, store, _ = searchable
    assert len(flat_search(conn, store, store.read(3).copy(), model, k=3)) == 3


def test_unembedded_chunks_are_never_returned(conn, searchable):
    """A half-embedded library must be searched exactly, not approximately."""
    model, store, total = searchable
    segment = conn.execute("SELECT id, chunk_start FROM segments ORDER BY id").fetchone()
    with conn:
        db.set_embedded_prefix(conn, model, segment["id"], 3)

    reachable = {c for start, end in embedded_ranges(conn, model) for c in range(start, end)}
    hits = flat_search(conn, store, store.read(1).copy(), model, k=10)

    assert all(h.chunk_id in reachable for h in hits)
    assert segment["chunk_start"] + 5 not in reachable


def test_nothing_embedded_means_no_hits(conn, tmp_path, library):
    model = db.model_id(conn, "fresh", DIM)
    store = store_for(conn, tmp_path / "index", model, DIM)

    assert flat_search(conn, store, np.ones(DIM, dtype=np.float32), model, k=5) == []


def test_hits_resolve_to_book_and_passage(conn, searchable):
    model, store, _ = searchable
    hits = flat_search(conn, store, store.read(4).copy(), model, k=2)

    passages = resolve(conn, hits)
    assert len(passages) == 2
    assert passages[0].chunk_id == hits[0].chunk_id
    assert passages[0].title.startswith("book")
    assert passages[0].text and passages[0].text.strip() == passages[0].text


def test_a_missing_source_resolves_to_no_text_rather_than_raising(conn, searchable, library):
    """The vectors are still good; only the passage cannot be shown."""
    model, store, _ = searchable
    (library / "book0.txt").unlink()

    passages = resolve(conn, flat_search(conn, store, store.read(2).copy(), model, k=5))

    assert any(p.text is None for p in passages)
    assert all(p.score is not None for p in passages)


def test_search_reads_one_contiguous_slice_per_segment(conn, searchable):
    """Ranges are what make stage 2 a matrix multiply rather than a gather."""
    model, store, total = searchable
    ranges = embedded_ranges(conn, model)

    assert ranges == sorted(ranges)
    assert sum(end - start for start, end in ranges) == total


# --- scoping a search to part of the library ---------------------------------


def test_a_pattern_selects_books_by_title(conn, library):
    from dyprys.search import scope_books
    assert len(scope_books(conn, "book0")) == 1
    assert len(scope_books(conn, "book")) == 2


def test_scoping_is_case_insensitive(conn, library):
    from dyprys.search import scope_books
    assert scope_books(conn, "BOOK0") == scope_books(conn, "book0")


def test_a_glob_selects_a_set(conn, library):
    from dyprys.search import scope_books
    assert len(scope_books(conn, "book*")) == 2
    assert scope_books(conn, "nothing*") == set()


def test_a_scoped_search_returns_only_that_book(conn, searchable, library):
    """The same restriction stage 2 of routing applies, reached from a flag."""
    from dyprys.search import scope_books
    model, store, _ = searchable
    wanted = scope_books(conn, "book1")

    hits = flat_search(conn, store, store.read(1).copy(), model, k=10, book_ids=wanted)

    for hit in hits:
        assert db.locate(conn, hit.chunk_id)["book_id"] in wanted


def test_scoping_reports_the_fraction_of_corpus_scanned(conn, searchable, library):
    """The cost side of every quality claim, computed the same way routing will."""
    from dyprys.search import scanned_fraction, scope_books
    model, store, _ = searchable

    assert scanned_fraction(conn, model) == 1.0
    half = scanned_fraction(conn, model, scope_books(conn, "book0"))
    assert 0 < half < 1
    assert scanned_fraction(conn, model, scope_books(conn, "book")) == 1.0


def test_a_named_book_can_be_scoped_to_by_that_name(conn, searchable):
    """A name you can see but not say would be a name in name only.

    `label` is a display name someone chose, kept beside the title ingest
    derived rather than over it. The point of calling a book something is to be
    able to ask a question of it by that name, so the matcher takes the label
    alongside the title and the key -- and the key stays identity either way,
    because a label is no more unique than a title.
    """
    from dyprys.search import scope

    with conn:
        conn.execute("UPDATE books SET label = ? WHERE title = ?", ("Bartleby", "book0"))

    matched, missed = scope(conn, "bartleby")
    named = conn.execute("SELECT id FROM books WHERE title = 'book0'").fetchone()["id"]

    assert missed == [], "the label matched, so nothing was missed"
    assert matched == {named}
    # And the derived title still matches, because it was not overwritten.
    assert scope(conn, "book0")[0] == {named}


def test_scoping_to_nothing_finds_nothing_rather_than_everything(conn, searchable):
    """A pattern that matches no book must not silently widen to the library."""
    model, store, _ = searchable
    assert flat_search(conn, store, store.read(1).copy(), model, k=5, book_ids=set()) == []


def test_a_scoped_search_agrees_with_the_unscoped_one_on_that_book(conn, searchable, library):
    """Restricting must not change ranking within the restriction."""
    from dyprys.search import scope_books
    model, store, _ = searchable
    wanted = scope_books(conn, "book1")
    query = store.read(2).copy()

    scoped = flat_search(conn, store, query, model, k=50, book_ids=wanted)
    everything = flat_search(conn, store, query, model, k=50)
    from_that_book = [h for h in everything if db.locate(conn, h.chunk_id)["book_id"] in wanted]

    assert [h.chunk_id for h in scoped][:5] == [h.chunk_id for h in from_that_book][:5]


# --- range merging -----------------------------------------------------------


def make_segments(conn, model, count, per=100, embedded=None):
    """`count` consecutive fully-embedded segments, one book each."""
    chunking = db.chunking_id(conn, 1, 3600, 0)
    chunk_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()[0] + 1
    ids = []
    with conn:
        for n in range(count):
            book = conn.execute(
                "INSERT INTO books (key, title, added_at) VALUES (?, ?, ?)",
                (f"/synthetic/{n}", f"synthetic{n}", "x"),
            ).lastrowid
            source = conn.execute(
                "INSERT INTO sources (book_id, ordinal, path, size_bytes, mtime, "
                "content_hash, ingested_at) VALUES (?, 0, ?, 1, 1.0, 'h', 'x')",
                (book, f"/synthetic/{n}.txt"),
            ).lastrowid
            segment = conn.execute(
                "INSERT INTO segments (source_id, chunking_id, chunk_start, chunk_count) "
                "VALUES (?, ?, ?, ?)",
                (source, chunking, chunk_id, per),
            ).lastrowid
            db.set_embedded_prefix(conn, model, segment, embedded if embedded else per)
            ids.append(segment)
            chunk_id += per
    return ids


def test_touching_ranges_become_one(conn):
    """A fully embedded library is one span, however many books it holds."""
    model = db.model_id(conn, "m", DIM)
    make_segments(conn, model, 500)

    assert len(embedded_ranges(conn, model)) == 1


def test_a_partly_embedded_segment_still_splits_the_span(conn):
    """Merging must not paper over a gap that really exists."""
    model = db.model_id(conn, "m", DIM)
    segments = make_segments(conn, model, 10)
    with conn:
        db.set_embedded_prefix(conn, model, segments[4], 40)

    ranges = embedded_ranges(conn, model)
    assert len(ranges) == 2
    covered = sum(end - start for start, end in ranges)
    assert covered == 9 * 100 + 40  # exactly what is embedded, no more


def test_bm25_survives_a_library_of_thousands_of_segments(conn):
    """Unmerged this built one SQL OR-clause per segment and SQLite refused it."""
    from dyprys.lexical import search_bm25
    model = db.model_id(conn, "m", DIM)
    make_segments(conn, model, 2000)

    assert search_bm25(conn, "some words here", k=5, model_id=model) == []  # no index rows, but no error


def test_merging_does_not_invent_coverage(conn):
    model = db.model_id(conn, "m", DIM)
    make_segments(conn, model, 6, per=50, embedded=50)
    assert sum(e - s for s, e in embedded_ranges(conn, model)) == 300


def test_an_edit_above_a_passage_still_shows_it(conn, searchable, library):
    """The file grew at the front; the passage did not move relative to its text.

    `resolve` derives the displacement from the size recorded at ingest, so
    nothing outside it has to know the file was touched. Without that, every
    passage in an edited book disappears from results while its bytes sit intact
    a few hundred bytes further on.
    """
    model, store, _ = searchable
    book = library / "book0.txt"
    book.write_text("A newly written opening.\n\n" + book.read_text(), encoding="utf-8")

    passages = resolve(conn, flat_search(conn, store, store.read(2).copy(), model, k=5))
    moved = [p for p in passages if p.path == str(book)]

    assert moved, "the edited book should still be in the results"
    assert all(p.state == SHIFTED and p.text for p in moved)
    assert all(p.text in book.read_text(encoding="utf-8") for p in moved)


def test_a_rewritten_passage_is_reported_rather_than_approximated(conn, searchable, library):
    """A size change is a guess; the hash is what decides.

    Here the file is edited *and* stays the same length, so the shift is zero
    and there is nowhere to look. The passage must come back empty, not come
    back as whatever now sits at those bytes.
    """
    model, store, _ = searchable
    book = library / "book0.txt"
    original = book.read_text(encoding="utf-8")
    book.write_text("Z" * len(original), encoding="utf-8")

    passages = resolve(conn, flat_search(conn, store, store.read(2).copy(), model, k=5))
    edited = [p for p in passages if p.path == str(book)]

    assert edited
    assert all(p.state == CHANGED and p.text is None for p in edited)


def test_duplicate_suppression_still_works_on_an_edited_book(conn, searchable, library):
    """Suppression used to switch itself off silently when a file was edited.

    It read the recorded offset strictly, so every passage in an edited book came
    back unreadable, and an unreadable passage is kept on the grounds that it
    cannot be compared -- for results display was showing in full. Both halves
    now look for a passage the same way.
    """
    model, store, _ = searchable
    book = library / "book0.txt"
    body = book.read_text(encoding="utf-8")
    book.write_text("A newly written opening.\n\n" + body, encoding="utf-8")

    hits = flat_search(conn, store, store.read(2).copy(), model, k=10)
    shifts: dict[int, int] = {}
    rows = [
        (
            db.locate(conn, h.chunk_id),
            conn.execute(
                "SELECT byte_offset, byte_length, content_hash FROM chunks WHERE id = ?",
                (h.chunk_id,),
            ).fetchone(),
        )
        for h in hits
    ]
    readable = [_span_of(loc, row, shifts).text for loc, row in rows]

    assert any(t is not None for t in readable), "the edited book must still read"
    assert all(t is not None for t in readable), "every passage moved by one constant"


def test_a_passage_carries_the_offset_its_text_was_read_from(conn, searchable):
    """A citation should name a byte in a file, not only a chunk id.

    And it must be where the text was *found*, not where it was recorded — an
    edited file moves passages, and a citation pointing at the old offset would
    send a reader to different words.
    """
    model, store, _ = searchable
    row = conn.execute("SELECT byte_offset FROM chunks WHERE id = 2").fetchone()

    passage = resolve(conn, flat_search(conn, store, store.read(2).copy(), model, k=1))[0]

    assert passage.chunk_id == 2
    assert passage.offset == row["byte_offset"]
