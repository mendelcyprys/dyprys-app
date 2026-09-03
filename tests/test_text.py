"""Reading a passage back: soft failure, and an exact trust guarantee."""

import pytest

from dyprys import db
from dyprys.ingest import ingest_paths
from dyprys.text import (
    CHANGED,
    EXACT,
    MISSING,
    SHIFTED,
    locate_span,
    read_span,
    span_hash,
)


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "index")
    yield connection
    connection.close()


@pytest.fixture
def book(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text(
        "\n\n".join(f"Paragraph {n} about neurons and synapses. " * 15 for n in range(60)),
        encoding="utf-8",
    )
    return path


def first_chunk(conn):
    return conn.execute(
        "SELECT byte_offset, byte_length, content_hash FROM chunks ORDER BY id"
    ).fetchone()


def test_a_good_span_reads_back(conn, book):
    ingest_paths(conn, [book])
    row = first_chunk(conn)

    passage = read_span(book, row["byte_offset"], row["byte_length"])
    assert passage
    assert passage in book.read_text(encoding="utf-8")


def test_the_stored_hash_verifies_the_passage(conn, book):
    ingest_paths(conn, [book])
    row = first_chunk(conn)

    passage = read_span(
        book, row["byte_offset"], row["byte_length"], expect_hash=row["content_hash"]
    )
    assert passage is not None


def test_a_deleted_source_yields_none_rather_than_raising(conn, book):
    """One book on an unmounted drive must not kill a whole query."""
    ingest_paths(conn, [book])
    row = first_chunk(conn)
    book.unlink()

    assert read_span(book, row["byte_offset"], row["byte_length"]) is None


def test_a_truncated_source_yields_none(conn, book):
    ingest_paths(conn, [book])
    row = conn.execute(
        "SELECT byte_offset, byte_length FROM chunks ORDER BY id DESC LIMIT 1"
    ).fetchone()
    book.write_text("much shorter now", encoding="utf-8")

    assert read_span(book, row["byte_offset"], row["byte_length"]) is None


def test_an_edit_that_keeps_the_length_is_caught_by_the_hash(conn, book):
    """The case nothing cheaper can catch: same size, different bytes."""
    ingest_paths(conn, [book])
    row = first_chunk(conn)
    text = book.read_text(encoding="utf-8")
    book.write_text(text.replace("neurons", "NEURONS", 1), encoding="utf-8")

    # length is unchanged, so without the hash a stale passage would be shown
    assert read_span(book, row["byte_offset"], row["byte_length"]) is not None
    assert read_span(
        book, row["byte_offset"], row["byte_length"], expect_hash=row["content_hash"]
    ) is None


def test_the_query_side_and_ingest_side_hashes_agree(conn, book):
    """One function, so the two can never drift apart."""
    ingest_paths(conn, [book])
    row = first_chunk(conn)
    raw = book.read_bytes()[row["byte_offset"] : row["byte_offset"] + row["byte_length"]]

    assert span_hash(raw) == row["content_hash"]


# --- finding a passage again after the file was edited ------------------------


def last_chunk(conn):
    return conn.execute(
        "SELECT byte_offset, byte_length, content_hash FROM chunks ORDER BY id DESC LIMIT 1"
    ).fetchone()


def test_an_untouched_file_is_found_exactly(conn, book):
    ingest_paths(conn, [book])
    row = first_chunk(conn)

    span = locate_span(book, row["byte_offset"], row["byte_length"], row["content_hash"])
    assert span.state == EXACT
    assert span.offset == row["byte_offset"]
    assert span.text


def test_an_insertion_at_the_front_relocates_every_later_passage(conn, book):
    """The case that motivated this: an edit above a passage, not to it.

    Without the shift the passage is unprovable and vanishes from results, even
    though its bytes are sitting in the file untouched a few hundred bytes on.
    """
    ingest_paths(conn, [book])
    row = last_chunk(conn)
    inserted = "A newly written opening paragraph.\n\n"
    original = book.read_text(encoding="utf-8")
    book.write_text(inserted + original, encoding="utf-8")
    shift = len(inserted.encode("utf-8"))

    lost = locate_span(book, row["byte_offset"], row["byte_length"], row["content_hash"])
    assert lost.state == CHANGED and lost.text is None

    found = locate_span(
        book, row["byte_offset"], row["byte_length"], row["content_hash"], shift
    )
    assert found.state == SHIFTED
    assert found.offset == row["byte_offset"] + shift
    assert found.text in book.read_text(encoding="utf-8")


def test_a_wrong_shift_reports_changed_rather_than_the_wrong_passage(conn, book):
    """A hash match is proof; a guess that does not verify shows nothing.

    This is what stops the feature from being worse than no feature: two edits
    give two different shifts, and the half this guess does not cover must come
    back empty rather than come back plausible.
    """
    ingest_paths(conn, [book])
    row = last_chunk(conn)
    original = book.read_text(encoding="utf-8")
    book.write_text("12345" + original, encoding="utf-8")

    span = locate_span(
        book, row["byte_offset"], row["byte_length"], row["content_hash"], shift=999
    )
    assert span.state == CHANGED
    assert span.text is None


def test_a_missing_file_is_distinguished_from_an_edited_one(conn, book):
    """Three outcomes, three different things for the reader to do."""
    ingest_paths(conn, [book])
    row = first_chunk(conn)
    original = book.read_text(encoding="utf-8")

    book.write_text(original.replace("neurons", "NEURONS", 1), encoding="utf-8")
    assert locate_span(
        book, row["byte_offset"], row["byte_length"], row["content_hash"]
    ).state == CHANGED

    book.unlink()
    assert locate_span(
        book, row["byte_offset"], row["byte_length"], row["content_hash"]
    ).state == MISSING


def test_shifting_is_never_consulted_when_the_recorded_offset_is_right(conn, book):
    """An intact file must not be searched twice, nor answer from the wrong place.

    A pathological shift is passed; the exact hit has to win before it is tried.
    """
    ingest_paths(conn, [book])
    row = first_chunk(conn)

    span = locate_span(
        book, row["byte_offset"], row["byte_length"], row["content_hash"], shift=4096
    )
    assert span.state == EXACT
    assert span.offset == row["byte_offset"]


def test_relocating_without_a_hash_is_refused(conn, book):
    """The shift is a guess, and only the hash can settle which guess is right.

    Allowed through, it returned the bytes at the stale offset labelled EXACT --
    a claim of proof in the one case where no proof was available.
    """
    ingest_paths(conn, [book])
    row = first_chunk(conn)

    with pytest.raises(ValueError):
        locate_span(book, row["byte_offset"], row["byte_length"], None, shift=50)


# --- reading a passage with its surroundings ---------------------------------


def test_a_window_starts_and_ends_on_a_sentence(tmp_path):
    """A chunk boundary is a byte budget, so it lands mid-sentence often.

    The first summary this project produced began "neuropeptides, and
    neurosteroids, as well as..." because that is where the chunk started and
    the model copied the fragment it was handed.
    """
    from dyprys.text import read_window

    body = ("First sentence here. Second sentence here. THE CHUNK BEGINS "
            "mid-thought and runs on. Fourth sentence here. Fifth here.")
    path = tmp_path / "b.txt"
    path.write_text(body, encoding="utf-8")
    start = body.index("mid-thought")

    text, at = read_window(path, start, len("mid-thought and runs on."),
                           before=60, after=60)

    assert not text.startswith("mid-thought"), "the window did not widen"
    assert "mid-thought and runs on." in text
    assert at < start
    assert body[at:at + len(text)] == text, "the offset does not name the text"


def test_a_window_keeps_as_much_context_as_it_can(tmp_path):
    """The *first* clean edge in the leading context, not the last.

    Taking the last one trims the context away entirely, which is what the
    first version did — it returned a window starting at the chunk.
    """
    from dyprys.text import read_window

    body = "Alpha one. Beta two. Gamma three. TARGET here. Delta four."
    path = tmp_path / "b.txt"
    path.write_text(body, encoding="utf-8")
    start = body.index("TARGET")

    text, _ = read_window(path, start, len("TARGET here."), before=40, after=20)

    assert "Beta two." in text, "context was trimmed to nothing"


def test_a_window_at_the_start_of_a_file_does_not_run_off_it(tmp_path):
    from dyprys.text import read_window

    path = tmp_path / "b.txt"
    path.write_text("Right at the beginning. And more after.", encoding="utf-8")

    text, at = read_window(path, 0, 23, before=500, after=100)

    assert at == 0 and text.startswith("Right at the beginning")


def test_a_window_never_splits_a_character(tmp_path):
    """Arbitrary byte offsets land inside multi-byte characters."""
    from dyprys.text import read_window

    body = "Naïve café. Достаточно текста здесь. Le café est prêt. Ende."
    path = tmp_path / "b.txt"
    path.write_text(body, encoding="utf-8")
    raw = body.encode()
    start = raw.index("Le café".encode())

    text, at = read_window(path, start, len("Le café est prêt.".encode()),
                           before=7, after=7)

    assert "�" not in text
    assert body.encode()[at:at + len(text.encode())].decode() == text


def test_a_missing_file_gives_no_window(tmp_path):
    from dyprys.text import read_window

    assert read_window(tmp_path / "gone.txt", 0, 10) is None
