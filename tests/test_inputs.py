"""What dyprys accepts, and what it does when a library moves."""

import pytest

from dyprys import db
from dyprys.ingest import NothingToIngest, ingest_paths
from dyprys.library import relocate, unreadable_sources


def write(directory, name, body="Paragraph about neurons. " * 400):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# --- the input contract ------------------------------------------------------


def test_only_txt_is_taken_by_default(conn, tmp_path):
    write(tmp_path / "lib", "kept.txt")
    write(tmp_path / "lib", "skipped.md")

    results = ingest_paths(conn, [tmp_path / "lib"])

    assert [r.title for r in results] == ["kept"]


def test_other_extensions_can_be_asked_for(conn, tmp_path):
    write(tmp_path / "lib", "one.md")
    write(tmp_path / "lib", "two.markdown")

    results = ingest_paths(conn, [tmp_path / "lib"], suffixes=(".md", ".markdown"))

    assert sorted(r.title for r in results) == ["one", "two"]


def test_matching_nothing_raises_rather_than_quietly_succeeding(conn, tmp_path):
    """The first command a new user runs must not do nothing and report success."""
    write(tmp_path / "lib", "book.md")

    with pytest.raises(NothingToIngest):
        ingest_paths(conn, [tmp_path / "lib"])


def test_the_refusal_says_what_was_there_instead(conn, tmp_path):
    """So the fix is obvious without reading the source."""
    write(tmp_path / "lib", "a.md")
    write(tmp_path / "lib", "b.md")
    write(tmp_path / "lib", "c.epub")

    with pytest.raises(NothingToIngest) as raised:
        ingest_paths(conn, [tmp_path / "lib"])

    found = dict(raised.value.found_instead())
    assert found[".md"] == 2
    assert found[".epub"] == 1


def test_an_empty_directory_reports_nothing_rather_than_guessing(conn, tmp_path):
    (tmp_path / "lib").mkdir()
    with pytest.raises(NothingToIngest) as raised:
        ingest_paths(conn, [tmp_path / "lib"])
    assert raised.value.found_instead() == []


# --- the library moves -------------------------------------------------------


@pytest.fixture
def moved(conn, tmp_path):
    old = tmp_path / "before"
    for n in range(3):
        write(old, f"book{n}.txt")
    ingest_paths(conn, [old])
    new = tmp_path / "after"
    old.rename(new)
    return old, new


def test_a_moved_library_is_all_missing_until_relocated(conn, moved):
    old, new = moved
    assert len(unreadable_sources(conn, limit=99)) == 3


def test_relocating_rewrites_every_path(conn, moved):
    old, new = moved
    books, sources = relocate(conn, str(old), str(new))

    assert (books, sources) == (3, 3)
    assert unreadable_sources(conn) == []


def test_relocating_does_not_touch_the_chunks(conn, moved):
    """Offsets are into a file's bytes; moving the file changes none of them."""
    old, new = moved
    before = [tuple(r) for r in conn.execute(
        "SELECT id, byte_offset, byte_length, content_hash FROM chunks ORDER BY id")]

    relocate(conn, str(old), str(new))

    after = [tuple(r) for r in conn.execute(
        "SELECT id, byte_offset, byte_length, content_hash FROM chunks ORDER BY id")]
    assert after == before


def test_passages_read_again_after_relocating(conn, moved):
    from dyprys.text import read_span
    old, new = moved
    relocate(conn, str(old), str(new))

    for row in conn.execute("SELECT id, byte_offset, byte_length, content_hash FROM chunks"):
        located = db.locate(conn, row["id"])
        assert read_span(located["path"], row["byte_offset"], row["byte_length"],
                         row["content_hash"]) is not None


def test_a_prefix_that_matches_nothing_changes_nothing(conn, moved):
    old, new = moved
    assert relocate(conn, "/somewhere/else", str(new)) == (0, 0)


def test_relocating_is_idempotent(conn, moved):
    old, new = moved
    relocate(conn, str(old), str(new))
    again = relocate(conn, str(old), str(new))

    assert again == (0, 0)          # nothing still carries the old prefix
    assert unreadable_sources(conn) == []


# --- what relocate must not touch --------------------------------------------


def _record(conn, *paths):
    """Books and sources at literal paths, without going near the filesystem."""
    with conn:
        for key in paths:
            book = conn.execute(
                "INSERT INTO books (key, title, added_at) VALUES (?, ?, ?)",
                (key, "t", db.now()),
            ).lastrowid
            conn.execute(
                "INSERT INTO sources (book_id, ordinal, path, size_bytes, mtime, "
                "content_hash, ingested_at) VALUES (?, 0, ?, 0, 0, 'h', ?)",
                (book, key, db.now()),
            )


def test_relocating_leaves_a_library_whose_name_merely_resembles_it(conn):
    """`_` is a LIKE wildcard, and `my_books` is an ordinary directory name.

    It matched `myXbooks` -- a different library, on a different disk, redirected
    to a path its files are not at. `restore` goes through this too, so a backup
    could quietly repoint an index you already had.
    """
    _record(conn, "/Volumes/my_books/a.txt", "/Volumes/myXbooks/b.txt")

    books, sources = relocate(conn, "/Volumes/my_books", "/mnt/lib")

    assert (books, sources) == (1, 1)
    paths = [r["path"] for r in conn.execute("SELECT path FROM sources ORDER BY path")]
    assert paths == ["/Volumes/myXbooks/b.txt", "/mnt/lib/a.txt"]


def test_relocating_stops_at_a_path_component(conn):
    """A prefix is not a path: `/x/my_books` must not match `/x/my_booksXTRA`."""
    _record(conn, "/x/my_books/a.txt", "/x/my_booksXTRA/c.txt")

    books, sources = relocate(conn, "/x/my_books", "/mnt/lib")

    assert (books, sources) == (1, 1)
    paths = [r["path"] for r in conn.execute("SELECT path FROM sources ORDER BY path")]
    assert paths == ["/mnt/lib/a.txt", "/x/my_booksXTRA/c.txt"]


def test_relocating_moves_the_prefix_itself_and_everything_under_it(conn):
    _record(conn, "/x/lib", "/x/lib/a.txt", "/x/lib/deep/nested/e.txt")

    assert relocate(conn, "/x/lib", "/mnt/new") == (3, 3)
    assert [r["path"] for r in conn.execute("SELECT path FROM sources ORDER BY path")] == [
        "/mnt/new", "/mnt/new/a.txt", "/mnt/new/deep/nested/e.txt",
    ]


# --- a named file is filtered like a walked one ------------------------------


def test_naming_a_file_does_not_bypass_the_extension_filter(conn, tmp_path):
    """`dyp add book.pdf` used to chunk the binary and call it a book.

    The filter only ran when walking a directory, so `dyp add library/*` -- which
    the shell expands to files -- bypassed --ext entirely.
    """
    binary = tmp_path / "paper.pdf"
    binary.write_bytes(b"%PDF-1.4\n" + bytes(range(256)) * 40)

    with pytest.raises(NothingToIngest) as refused:
        ingest_paths(conn, [binary])

    assert refused.value.found_instead() == [(".pdf", 1)]


def test_a_named_file_of_the_right_kind_still_goes_in(conn, tmp_path):
    path = write(tmp_path, "book.txt")
    assert ingest_paths(conn, [path])[0].status == "added"


def test_a_named_file_with_no_extension_is_accepted(conn, tmp_path):
    """The filter refuses formats we cannot chunk; no extension claims nothing."""
    path = write(tmp_path, "NOTES")
    assert ingest_paths(conn, [path])[0].status == "added"


def test_a_glob_ingests_what_it_can_and_refuses_the_rest(conn, tmp_path):
    good = write(tmp_path, "book.txt")
    bad = tmp_path / "paper.pdf"
    bad.write_bytes(b"%PDF-1.4\nbinary")

    results = ingest_paths(conn, [good, bad])       # as the shell expands lib/*

    assert [r.title for r in results] == ["book"]


# --- the same book from two extractions --------------------------------------


def test_a_second_extraction_of_one_book_is_flagged_by_title(conn, tmp_path):
    """Content hashing cannot see this, and the cost of missing it is real.

    Run a PDF through a different converter and every byte differs, so every
    chunk hash differs and the library holds the work twice. Building the
    2,341-book corpus this happened to 113 books — 54,875 chunks, 40% of the
    outstanding embedding — and nothing said a word.
    """
    from dyprys.library import duplicate_titles

    first = tmp_path / "one"
    second = tmp_path / "two"
    write(first, "Kandel.txt", "Paragraph about neurons. " * 400)
    # Same book, a different extraction: different spacing, different bytes.
    write(second, "Kandel.txt", "Paragraph  about  neurons. " * 400)
    ingest_paths(conn, [first])
    ingest_paths(conn, [second])

    assert conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 2
    assert duplicate_titles(conn) == [("Kandel", 2)]


def test_the_same_bytes_at_a_new_path_is_still_a_move_not_a_duplicate(conn, tmp_path):
    """The hash path must keep working: identical bytes are one book, relocated."""
    from dyprys.library import duplicate_titles

    first = tmp_path / "one"
    second = tmp_path / "two"
    body = "Paragraph about neurons. " * 400
    path = write(first, "Kandel.txt", body)
    ingest_paths(conn, [first])

    path.unlink()                      # the original is gone: a move
    write(second, "Kandel.txt", body)
    result = ingest_paths(conn, [second])[0]

    assert result.status == "moved"
    assert conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 1
    assert duplicate_titles(conn) == []


def test_a_library_with_no_repeats_reports_none(conn, library):
    from dyprys.library import duplicate_titles

    assert duplicate_titles(conn) == []
