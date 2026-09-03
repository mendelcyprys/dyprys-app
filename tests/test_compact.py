"""Compaction closes the gaps without moving a single vector away from its text."""

import numpy as np
import pytest

from dyprys import db
from dyprys.compact import (
    CompactReport,
    Run,
    compact,
    interrupted,
    outstanding_carries,
    plan,
    remove_books,
)
from dyprys.embed import store_for
from dyprys.ingest import ingest_paths
from dyprys.lexical import indexed_count, search_bm25
from dyprys.search import embedded_ranges, flat_search
from dyprys.text import read_span

DIM = 8


def fill(conn, store, total):
    """Give every chunk a vector that identifies it, so a move is detectable."""
    for chunk_id in range(1, total + 1):
        v = np.zeros(DIM, dtype=np.float32)
        v[chunk_id % DIM] = 1.0
        v[(chunk_id * 3) % DIM] += 0.25
        store.write(chunk_id, v / np.linalg.norm(v))
    store.flush()


@pytest.fixture
def stocked(conn, tmp_path, library):
    """Three books embedded, then one edited so its old block is dead."""
    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "index", model, DIM)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    fill(conn, store, conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    return model, store, tmp_path / "index"


def make_dead(conn, library, store, model):
    """Edit a book so its original chunk block is superseded."""
    book = library / "book0.txt"
    book.write_text(book.read_text(encoding="utf-8") + "\n\nA tail.\n", encoding="utf-8")
    ingest_paths(conn, [library])
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    store.grow(total)
    with conn:
        conn.execute("DELETE FROM chunk_carry")          # consumed, as embed would
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    fill(conn, store, total)


# --- the plan ----------------------------------------------------------------


def test_a_clean_index_has_nothing_to_reclaim(conn, stocked):
    assert plan(conn).dead == 0
    assert not plan(conn).worth_doing


def test_an_edit_leaves_dead_ids_behind(conn, stocked, library):
    model, store, _ = stocked
    make_dead(conn, library, store, model)

    shape = plan(conn)
    assert shape.dead > 0
    assert shape.live == conn.execute(
        "SELECT COALESCE(SUM(chunk_count), 0) FROM segments"
    ).fetchone()[0]


def test_the_plan_only_ever_moves_ids_downwards(conn, stocked, library):
    """The property that makes an in-place rewrite safe."""
    model, store, _ = stocked
    make_dead(conn, library, store, model)

    for run in plan(conn).runs:
        assert run.new_start <= run.old_start


def test_runs_do_not_overlap_after_the_move(conn, stocked, library):
    model, store, _ = stocked
    make_dead(conn, library, store, model)

    runs = sorted(plan(conn).runs, key=lambda r: r.new_start)
    cursor = 1
    for run in runs:
        assert run.new_start == cursor
        cursor += run.count


# --- doing it ----------------------------------------------------------------


def test_compaction_reclaims_the_dead_ids(conn, stocked, library):
    model, store, directory = stocked
    make_dead(conn, library, store, model)
    before = plan(conn)

    report = compact(conn, directory)

    assert report.reclaimed == before.dead
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == before.live
    assert plan(conn).dead == 0


def test_every_vector_follows_its_own_chunk(conn, stocked, library):
    """The one thing compaction must not get wrong."""
    model, store, directory = stocked
    make_dead(conn, library, store, model)

    # remember, per live chunk, the text and the vector that belong together
    before = {}
    for seg in conn.execute("SELECT chunk_start, chunk_count FROM segments").fetchall():
        for cid in range(seg["chunk_start"], seg["chunk_start"] + seg["chunk_count"]):
            row = conn.execute(
                "SELECT byte_offset, byte_length, content_hash FROM chunks WHERE id = ?", (cid,)
            ).fetchone()
            before[row["content_hash"]] = store.read(cid).copy()

    compact(conn, directory)

    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    after = store_for(conn, directory, model, DIM)
    assert len(before) >= total * 0.9
    checked = 0
    for row in conn.execute("SELECT id, content_hash FROM chunks"):
        if row["content_hash"] in before:
            assert np.allclose(after.read(row["id"]), before[row["content_hash"]])
            checked += 1
    assert checked > 0


def test_chunk_ids_are_dense_again(conn, stocked, library):
    model, store, directory = stocked
    make_dead(conn, library, store, model)
    compact(conn, directory)

    ids = [r["id"] for r in conn.execute("SELECT id FROM chunks ORDER BY id")]
    assert ids == list(range(1, len(ids) + 1))


def test_offsets_still_read_the_right_text(conn, stocked, library):
    model, store, directory = stocked
    make_dead(conn, library, store, model)
    compact(conn, directory)

    for row in conn.execute("SELECT id, byte_offset, byte_length, content_hash FROM chunks"):
        located = db.locate(conn, row["id"])
        assert located is not None
        text = read_span(located["path"], row["byte_offset"], row["byte_length"],
                         row["content_hash"])
        assert text is not None      # the hash still verifies after renumbering


def test_the_lexical_index_is_rebuilt_and_searchable(conn, stocked, library):
    model, store, directory = stocked
    make_dead(conn, library, store, model)
    compact(conn, directory)

    live = conn.execute("SELECT COALESCE(SUM(chunk_count), 0) FROM segments").fetchone()[0]
    assert indexed_count(conn) == live
    assert search_bm25(conn, "neurons paragraph", k=3)


def test_search_still_works_afterwards(conn, stocked, library):
    model, store, directory = stocked
    make_dead(conn, library, store, model)
    compact(conn, directory)

    after = store_for(conn, directory, model, DIM)
    hits = flat_search(conn, after, after.read(3).copy(), model, k=5)
    assert hits
    assert all(db.locate(conn, h.chunk_id) is not None for h in hits)


def test_the_vector_file_shrinks(conn, stocked, library):
    model, store, directory = stocked
    make_dead(conn, library, store, model)
    path = db.vectors_path(directory, model)
    before = path.stat().st_size

    compact(conn, directory)

    assert path.stat().st_size < before


# --- refusals and resumption -------------------------------------------------


def test_it_refuses_while_carries_are_outstanding(conn, stocked, library):
    """Carries point at rows compaction is about to move."""
    model, store, _ = stocked
    book = library / "book0.txt"
    book.write_text(book.read_text(encoding="utf-8") + "\n\nA tail.\n", encoding="utf-8")
    ingest_paths(conn, [library])
    assert outstanding_carries(conn) > 0

    with pytest.raises(RuntimeError, match="carries"):
        compact(conn, store.path.parent)


def test_compacting_a_clean_index_changes_nothing(conn, stocked):
    model, store, directory = stocked
    before = [tuple(r) for r in conn.execute("SELECT id, byte_offset FROM chunks ORDER BY id")]

    report = compact(conn, directory)

    assert report.reclaimed == 0
    assert [tuple(r) for r in conn.execute("SELECT id, byte_offset FROM chunks ORDER BY id")] == before


def test_an_interrupted_compaction_is_visible_and_resumable(conn, stocked, library):
    model, store, directory = stocked
    make_dead(conn, library, store, model)
    expected = plan(conn)

    boom = {"n": 0}
    def explode(*a):
        boom["n"] += 1
        if boom["n"] > 1:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        compact(conn, directory, progress=explode)

    assert interrupted(conn)

    report = compact(conn, directory)
    assert report.resumed
    assert not interrupted(conn)
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == expected.live


# --- removing a book ---------------------------------------------------------


def test_removing_a_book_leaves_its_chunks_dead(conn, stocked):
    model, store, _ = stocked
    victim = conn.execute("SELECT id FROM books ORDER BY id LIMIT 1").fetchone()["id"]
    total_before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    remove_books(conn, {victim})

    assert conn.execute("SELECT COUNT(*) FROM books WHERE id = ?", (victim,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == total_before
    assert plan(conn).dead > 0     # its ids are now reclaimable


def test_removing_then_compacting_frees_the_space(conn, stocked):
    model, store, directory = stocked
    victim = conn.execute("SELECT id FROM books ORDER BY id LIMIT 1").fetchone()
    doomed = conn.execute(
        "SELECT COALESCE(SUM(seg.chunk_count), 0) FROM segments seg "
        "JOIN sources src ON src.id = seg.source_id WHERE src.book_id = ?", (victim["id"],)
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    remove_books(conn, {victim["id"]})
    report = compact(conn, directory)

    assert report.reclaimed == doomed
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == total - doomed


def test_the_surviving_books_are_untouched(conn, stocked, library):
    model, store, directory = stocked
    victim = conn.execute("SELECT id, title FROM books ORDER BY id LIMIT 1").fetchone()
    survivor = conn.execute(
        "SELECT id, title FROM books WHERE id != ? LIMIT 1", (victim["id"],)
    ).fetchone()
    kept = {
        r["content_hash"]: store.read(r["id"]).copy()
        for r in conn.execute(
            "SELECT c.id, c.content_hash FROM chunks c JOIN segments seg "
            "  ON c.id >= seg.chunk_start AND c.id < seg.chunk_start + seg.chunk_count "
            "JOIN sources src ON src.id = seg.source_id WHERE src.book_id = ?",
            (survivor["id"],),
        )
    }

    remove_books(conn, {victim["id"]})
    compact(conn, directory)

    after = store_for(conn, directory, model, DIM)
    for row in conn.execute("SELECT id, content_hash FROM chunks"):
        if row["content_hash"] in kept:
            assert np.allclose(after.read(row["id"]), kept[row["content_hash"]])
    assert conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 1


def test_removing_nothing_is_a_no_op(conn, stocked):
    assert remove_books(conn, set()) == 0


# --- surviving an interruption -----------------------------------------------
#
# A run longer than the gap it closes writes over rows it has still to read, so
# repeating it reads what the first pass already shifted. The resume used to
# restart the current model at run 0, which silently attached vectors to the
# wrong passages.


def overlapping(conn, tmp_path, library):
    """One tiny book between two big ones, then remove the tiny one.

    A small gap is the dangerous one: every later segment is longer than it, so
    every later run overruns its own source. Removing a *large* book leaves a
    gap nothing overruns, which is why the shape of the fixture matters here.
    """
    (library / "aaa_tiny.txt").write_text("One short paragraph.\n", encoding="utf-8")
    for name in ("bbb", "ccc"):
        (library / f"{name}_big.txt").write_text(
            "\n\n".join(f"Para {i} of {name}, on synapses. " * 12 for i in range(50)),
            encoding="utf-8",
        )
    ingest_paths(conn, [library])
    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "index", model, DIM)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    fill(conn, store, conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    store.close()
    tiny = conn.execute("SELECT id FROM books WHERE key LIKE '%aaa_tiny%'").fetchone()
    remove_books(conn, {tiny["id"]})
    return model


def vectors_by_text(conn, tmp_path, model):
    """Every live chunk's vector, keyed by the passage it is supposed to describe.

    Keyed by text and not by id on purpose: compaction renumbers, so an id-keyed
    comparison would pass while every vector pointed at the wrong passage.
    """
    store = store_for(conn, tmp_path / "index", model, DIM)
    found = {}
    for seg in conn.execute(
        "SELECT seg.chunk_start, seg.chunk_count, src.path FROM segments seg "
        "JOIN sources src ON src.id = seg.source_id ORDER BY seg.chunk_start"
    ):
        for chunk_id in range(seg["chunk_start"], seg["chunk_start"] + seg["chunk_count"]):
            row = conn.execute(
                "SELECT byte_offset, byte_length, content_hash FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
            text = read_span(seg["path"], row["byte_offset"], row["byte_length"],
                             row["content_hash"])
            found[text] = store.read(chunk_id).copy()
    store.close()
    return found


def test_a_small_removal_makes_every_later_run_overrun_its_source(conn, tmp_path, library):
    overlapping(conn, tmp_path, library)
    shape = plan(conn)
    moving = [r for r in shape.runs if r.moves]

    assert moving, "the fixture must actually move something"
    assert all(r.self_overlaps for r in moving), (
        "this fixture exists to exercise the unrepeatable move"
    )


class _Interruption:
    """Raise on the Nth vector write, checkpoint or index rebuild."""

    HOOKS = [
        ("dyprys.vectors", "VectorStore", "move_rows"),
        ("dyprys.vectors", "VectorStore", "write_raw"),
        ("dyprys.vectors", "VectorStore", "truncate"),
        ("dyprys.compact", None, "_save_state"),
        ("dyprys.lexical", None, "backfill"),
    ]

    def __init__(self, at):
        self.at, self.seen = at, 0

    def arm(self, monkeypatch):
        import importlib

        for module_name, class_name, attribute in self.HOOKS:
            module = importlib.import_module(module_name)
            owner = getattr(module, class_name) if class_name else module
            real = getattr(owner, attribute)

            def hooked(*args, _real=real, **kwargs):
                out = _real(*args, **kwargs)
                self.seen += 1
                if self.seen == self.at:
                    raise KeyboardInterrupt(f"power cut after operation {self.at}")
                return out

            monkeypatch.setattr(owner, attribute, hooked)


@pytest.mark.parametrize("at", range(1, 40))
def test_an_interruption_anywhere_resumes_without_moving_a_vector(
    conn, tmp_path, library, monkeypatch, at
):
    """Kill it after the Nth step, resume, and check every vector against its text.

    The sweep is the point: this failed at four of these forty positions before,
    with up to nine of twenty-two vectors describing the wrong passage, and
    `dyp compact` reported success either way.
    """
    model = overlapping(conn, tmp_path, library)
    before = vectors_by_text(conn, tmp_path, model)

    _Interruption(at).arm(monkeypatch)
    try:
        compact(conn, tmp_path / "index")
    except KeyboardInterrupt:
        pass
    monkeypatch.undo()

    if interrupted(conn):
        compact(conn, tmp_path / "index")

    after = vectors_by_text(conn, tmp_path, model)
    assert set(before) == set(after), f"interrupted at {at}: the passages changed"
    wrong = [t for t in before if not np.allclose(before[t], after[t], atol=1e-6)]
    assert not wrong, f"interrupted at {at}: {len(wrong)}/{len(before)} vectors wrong"


def test_the_staging_file_does_not_outlive_the_compaction(conn, tmp_path, library):
    from dyprys.compact import _scratch_path

    model = overlapping(conn, tmp_path, library)
    compact(conn, tmp_path / "index")

    assert not _scratch_path(tmp_path / "index", model).exists()
    assert not list((tmp_path / "index").glob("*.scratch"))


def test_unstaging_the_same_run_twice_changes_nothing(tmp_path):
    """Why staging works: the scratch copy can be replayed, the source cannot."""
    from dyprys.vectors import VectorStore

    store = VectorStore(tmp_path / "v.f32", dim=DIM, rows=40)
    for row in range(1, 41):
        store.write(row, np.full(DIM, row, dtype=np.float32))
    run = Run(segment_id=1, old_start=14, new_start=13, count=10)
    assert run.self_overlaps
    expected = store.slice(run.old_start, run.count).copy()

    scratch = VectorStore(tmp_path / "s.f32", dim=DIM, rows=run.count)
    scratch.write_raw(1, store.read_raw(run.old_start, run.count))
    for _ in range(3):
        store.write_raw(run.new_start, scratch.read_raw(1, run.count))
        assert np.allclose(store.slice(run.new_start, run.count), expected)
