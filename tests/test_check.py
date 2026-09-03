"""`dyp check` answers what drifted and what work remains."""

import os

import pytest

from dyprys import db
from dyprys.check import survey
from dyprys.ingest import ingest_paths


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "index")
    yield connection
    connection.close()


@pytest.fixture
def library(conn, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    for n in range(3):
        (lib / f"book{n}.txt").write_text(
            "\n\n".join(f"Paragraph {i} of book {n}. " * 15 for i in range(80)), "utf-8"
        )
    ingest_paths(conn, [lib])
    return lib


def embed_everything(conn, name="test-model", dim=768):
    model = db.model_id(conn, name, dim)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    return model


def test_a_fresh_index_is_clean(conn, library):
    report = survey(conn)

    assert report.drift.clean
    assert report.drift.intact == 3
    assert report.dead_chunks == 0
    assert report.models == []


def test_a_deleted_file_is_reported_missing(conn, library):
    (library / "book1.txt").unlink()

    report = survey(conn)

    assert len(report.drift.missing) == 1
    assert report.drift.missing[0].endswith("book1.txt")
    assert report.drift.intact == 2


def test_an_edited_file_is_reported_changed(conn, library):
    path = library / "book2.txt"
    path.write_text(path.read_text(encoding="utf-8") + "\n\nA new tail.\n", "utf-8")

    report = survey(conn)

    assert [p.split("/")[-1] for p in report.drift.changed] == ["book2.txt"]


def test_a_disguised_edit_needs_the_deep_check(conn, library):
    """Same size and mtime, different bytes: only re-hashing finds it."""
    path = library / "book0.txt"
    before = path.stat()
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("Paragraph 4 of", "Paragraph 5 of", 1), "utf-8")
    os.utime(path, (before.st_atime, before.st_mtime))

    assert survey(conn).drift.clean
    assert not survey(conn, deep=True).drift.clean


def test_outstanding_work_is_split_into_copying_and_embedding(conn, library):
    """The distinction that matters: one is free, the other is hours."""
    embed_everything(conn)
    path = library / "book0.txt"
    path.write_text(path.read_text(encoding="utf-8") + "\n\nA new tail.\n", "utf-8")
    ingest_paths(conn, [library])

    work = survey(conn).models[0]
    assert work.to_copy > 0        # most of the edited book survives
    assert work.to_embed > 0       # but not all of it
    assert work.outstanding == work.to_copy + work.to_embed


def test_an_untouched_index_reports_no_work(conn, library):
    embed_everything(conn)

    work = survey(conn).models[0]
    assert work.outstanding == 0
    assert work.failed == 0


def test_each_model_reports_its_own_backlog(conn, library):
    done = embed_everything(conn, "finished-model")
    fresh = db.model_id(conn, "new-model", 1024)

    by_name = {m.name: m for m in survey(conn).models}
    assert by_name["finished-model"].outstanding == 0
    assert by_name["new-model"].to_embed == survey(conn).live_chunks
    assert by_name["new-model"].embedded == 0
    assert done != fresh


def test_superseded_chunks_are_counted_as_reclaimable(conn, library):
    before = survey(conn)
    path = library / "book1.txt"
    path.write_text(path.read_text(encoding="utf-8") + "\n\nA new tail.\n", "utf-8")
    ingest_paths(conn, [library])

    after = survey(conn)
    assert after.dead_chunks > 0
    assert after.live_chunks >= before.live_chunks


def test_failures_surface_per_model(conn, library):
    model = embed_everything(conn)
    with conn:
        conn.execute(
            "INSERT INTO chunk_failures (model_id, chunk_id, reason, failed_at) "
            "VALUES (?, 1, 'tokeniser overflow', '2026-08-30')",
            (model,),
        )

    assert survey(conn).models[0].failed == 1


# --- text that is not words --------------------------------------------------


def _book(conn, tmp_path, name, body):
    from dyprys.ingest import ingest_paths
    lib = tmp_path / name
    lib.mkdir()
    (lib / f"{name}.txt").write_text(body, encoding="utf-8")
    ingest_paths(conn, [lib])


def test_a_pdf_whose_font_map_defeated_the_extractor_is_flagged(conn, tmp_path):
    """Real example: pdftotext returned this for two Springer books.

    Embedding it costs exactly what real text costs and it can never match a
    query, so it is worth knowing before spending the GPU rather than after.
    """
    glued = ("IY]MW][XPI[MJ]\\QVLQNNMZMV\\ZMOQWV[WN\\PMXZW\\MQVQVWZLMZ\\WLM\\MK\\[XMKQ "
             "UMUJZIVMQV\\MZIK\\QWV[QN\\PMaIZM\\WWKK]ZQN\\PMTIJMT[_MZMJ]ZQML\\PMa_W]TL ") * 40
    _book(conn, tmp_path, "broken", glued)

    flagged = [g.title for g in survey(conn).garbled]

    assert flagged == ["broken"]


def test_ordinary_prose_is_not_flagged(conn, tmp_path):
    _book(conn, tmp_path, "prose",
          "The cat sat upon the mat and considered the afternoon. " * 200)
    assert survey(conn).garbled == []


def test_a_book_that_is_a_table_of_numbers_is_not_flagged(conn, tmp_path):
    """Gutenberg holds 'Number e to one million places' and the 1990 US Census.

    Counting every token flagged both. They are legitimate books that happen to
    contain no prose, and telling someone to delete them is worse than not
    looking — so only tokens that are mostly letters count as words.
    """
    digits = " ".join("2718281828459045235360287471352662497757" for _ in range(400))
    _book(conn, tmp_path, "million_digits", digits)

    assert survey(conn).garbled == []


def test_non_english_prose_is_not_flagged(conn, tmp_path):
    """A word-length test must not become a test for English."""
    latin = ("Gallia est omnis divisa in partes tres quarum unam incolunt Belgae "
             "aliam Aquitani tertiam qui ipsorum lingua Celtae nostra Galli appellantur ") * 60
    _book(conn, tmp_path, "latin", latin)

    assert survey(conn).garbled == []


def test_only_scans_the_books_it_is_given(conn, tmp_path):
    """`dyp add` asks about the books it just ingested, not the whole library.

    At 3,453 books a full scan is 2.9s; on an incremental add that would be
    paid every time to re-learn what the previous run already reported.
    """
    from dyprys.check import garbled_books

    glued = ("IY]MW][XPI[MJ]\\QVLQNNMZMV\\ZMOQWV[WN\\PMXZW\\MQVQVWZLMZ\\WLM\\MK\\[XMKQ "
             "UMUJZIVMQV\\MZIK\\QWV[QN\\PMaIZM\\WWKK]ZQN\\PMTIJMT[_MZMJ]ZQML\\PMa_W]TL ") * 40
    _book(conn, tmp_path, "broken", glued)
    _book(conn, tmp_path, "fine", "The cat sat upon the mat this afternoon. " * 200)

    ids = {row["id"]: row["title"] for row in conn.execute("SELECT id, title FROM books")}
    broken_id = next(i for i, t in ids.items() if t == "broken")
    fine_id = next(i for i, t in ids.items() if t == "fine")

    assert [g.title for g in garbled_books(conn)] == ["broken"]
    assert [g.title for g in garbled_books(conn, only={broken_id})] == ["broken"]
    assert garbled_books(conn, only={fine_id}) == []
    assert garbled_books(conn, only=set()) == []


# --- reading the journal at the command line ---------------------------------


def test_ago_uses_the_coarsest_informative_unit():
    from datetime import datetime, timedelta, timezone

    from dyprys.cli import _ago

    def stamp(**kw):
        return (datetime.now(timezone.utc) - timedelta(**kw)).isoformat(timespec="seconds")

    assert _ago(stamp(seconds=5)).endswith("s ago")
    assert _ago(stamp(minutes=3)) == "3m ago"
    assert _ago(stamp(hours=5)) == "5h ago"
    assert _ago(stamp(days=2)) == "2d ago"


def test_ago_handles_a_clock_that_disagrees(): 
    """A timestamp from the future must read as something, not crash or lie."""
    from datetime import datetime, timedelta, timezone

    from dyprys.cli import _ago

    ahead = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(timespec="seconds")
    assert _ago(ahead) == "just now"
    assert _ago("not a timestamp") == ""


def test_naive_timestamps_are_read_as_utc():
    """db.now() is tz-aware, but an index written by an older build may not be."""
    from datetime import datetime, timedelta, timezone

    from dyprys.cli import _ago

    naive = (datetime.now(timezone.utc) - timedelta(minutes=4)).replace(
        tzinfo=None).isoformat(timespec="seconds")
    assert _ago(naive) == "4m ago"
