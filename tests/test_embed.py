"""Embedding is resumable, idempotent, and never recomputes finished work."""

import numpy as np
import pytest

from dyprys import db
from dyprys.embed import embed_pending, store_for
from dyprys.ingest import ingest_paths
from tests.conftest import DIM, StubEmbedder


@pytest.fixture
def setup(conn, tmp_path, library):
    embedder = StubEmbedder()
    model = db.model_id(conn, embedder.name, DIM)
    store = store_for(conn, tmp_path / "index", model, DIM)
    return embedder, model, store


def total_chunks(conn):
    return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def embedded(conn, model):
    return conn.execute(
        "SELECT COALESCE(SUM(n_embedded), 0) FROM segment_progress WHERE model_id = ?",
        (model,),
    ).fetchone()[0]


# --- the basics --------------------------------------------------------------


def test_it_embeds_everything_outstanding(conn, setup):
    embedder, model, store = setup

    report = embed_pending(conn, store, embedder, model)

    assert report.embedded == total_chunks(conn)
    assert embedded(conn, model) == total_chunks(conn)


def test_vectors_are_unit_length(conn, setup):
    embedder, model, store = setup
    embed_pending(conn, store, embedder, model)

    for chunk_id in (1, 2, total_chunks(conn)):
        assert np.isclose(np.linalg.norm(store.read(chunk_id)), 1.0, atol=1e-6)


def test_a_second_run_does_nothing(conn, setup):
    """Idempotent: the expensive work is never repeated."""
    embedder, model, store = setup
    embed_pending(conn, store, embedder, model)
    first_pass = len(embedder.seen)

    again = embed_pending(conn, store, embedder, model)

    assert again.embedded == 0
    assert len(embedder.seen) == first_pass  # the model was not called once more


# --- resumption --------------------------------------------------------------


def test_a_limited_run_stops_where_told(conn, setup):
    embedder, model, store = setup

    report = embed_pending(conn, store, embedder, model, limit=10, batch_size=5)

    assert report.embedded == 10
    assert embedded(conn, model) == 10


def test_resuming_continues_and_never_recomputes(conn, setup):
    """The property the corpus depends on: interruption costs nothing."""
    embedder, model, store = setup
    total = total_chunks(conn)

    while embedded(conn, model) < total:
        embed_pending(conn, store, embedder, model, limit=7, batch_size=3)

    assert embedded(conn, model) == total
    # every chunk embedded exactly once across all the interrupted runs
    assert len(embedder.seen) == total
    assert len(set(embedder.seen)) == total


def test_the_prefix_never_claims_a_vector_that_was_not_written(conn, setup):
    """Progress is only ever recorded behind a flushed vector."""
    embedder, model, store = setup
    embed_pending(conn, store, embedder, model, limit=12, batch_size=4)

    for segment in conn.execute("SELECT id, chunk_start FROM segments").fetchall():
        done = db.embedded_prefix(conn, model, segment["id"])
        for offset in range(done):
            vector = store.read(segment["chunk_start"] + offset)
            assert np.linalg.norm(vector) > 0  # a real vector, not untouched space


def test_progress_is_reported_against_the_real_total(conn, setup):
    embedder, model, store = setup
    seen = []
    embed_pending(conn, store, embedder, model, batch_size=5,
                  progress=lambda done, total, book: seen.append((done, total, book)))

    assert seen[-1][0] == total_chunks(conn)
    assert all(t == total_chunks(conn) for _, t, _b in seen)
    assert all(b.startswith("book") for *_ , b in seen)  # names the work in flight


# --- carries -----------------------------------------------------------------


def test_a_carried_chunk_is_copied_not_embedded(conn, setup, library):
    """Salvage has to actually save the model call, or it saves nothing."""
    embedder, model, store = setup
    embed_pending(conn, store, embedder, model)
    calls_before = len(embedder.seen)

    book = library / "book0.txt"
    book.write_text(book.read_text(encoding="utf-8") + "\n\nA short tail.\n", encoding="utf-8")
    ingest_paths(conn, [library])
    carries = conn.execute("SELECT COUNT(*) FROM chunk_carry").fetchone()[0]
    assert carries > 0

    store.grow(total_chunks(conn))
    report = embed_pending(conn, store, embedder, model)

    assert report.copied == carries
    assert report.embedded < carries  # the edit cost far less than the book
    assert len(embedder.seen) == calls_before + report.embedded


def test_a_copied_vector_matches_the_one_it_came_from(conn, setup, library):
    embedder, model, store = setup
    embed_pending(conn, store, embedder, model)

    book = library / "book0.txt"
    book.write_text(book.read_text(encoding="utf-8") + "\n\nA short tail.\n", encoding="utf-8")
    ingest_paths(conn, [library])
    pairs = conn.execute(
        "SELECT new_chunk_id, old_chunk_id FROM chunk_carry WHERE model_id = ?", (model,)
    ).fetchall()
    originals = {r["new_chunk_id"]: store.read(r["old_chunk_id"]).copy() for r in pairs}

    store.grow(total_chunks(conn))
    embed_pending(conn, store, embedder, model)

    for new_id, original in originals.items():
        assert np.allclose(store.read(new_id), original)


def test_consumed_carries_are_cleared(conn, setup, library):
    embedder, model, store = setup
    embed_pending(conn, store, embedder, model)
    book = library / "book0.txt"
    book.write_text(book.read_text(encoding="utf-8") + "\n\nA short tail.\n", encoding="utf-8")
    ingest_paths(conn, [library])

    store.grow(total_chunks(conn))
    embed_pending(conn, store, embedder, model)

    assert conn.execute("SELECT COUNT(*) FROM chunk_carry").fetchone()[0] == 0


# --- failure -----------------------------------------------------------------


def test_an_unreadable_source_is_recorded_and_stepped_over(conn, setup, library):
    """One bad chunk must not block the rest of its segment."""
    embedder, model, store = setup
    (library / "book0.txt").unlink()

    report = embed_pending(conn, store, embedder, model)

    assert report.failed > 0
    assert report.embedded > 0  # book1 still went through
    assert embedded(conn, model) == total_chunks(conn)
    assert conn.execute("SELECT COUNT(*) FROM chunk_failures").fetchone()[0] == report.failed


def test_a_failed_chunk_scores_zero_against_everything(conn, setup, library):
    embedder, model, store = setup
    (library / "book0.txt").unlink()
    embed_pending(conn, store, embedder, model)

    failed = conn.execute("SELECT chunk_id FROM chunk_failures LIMIT 1").fetchone()["chunk_id"]
    query = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
    assert float(store.read(failed) @ query) == 0.0


def test_a_source_edited_without_reindexing_is_caught(conn, setup, library):
    """The stored chunk hash is what stops a stale passage being embedded."""
    embedder, model, store = setup
    book = library / "book1.txt"
    text = book.read_text(encoding="utf-8")
    book.write_text(text.replace("Paragraph 0 of", "PARAGRAPH X of", 1), encoding="utf-8")

    report = embed_pending(conn, store, embedder, model)

    assert report.failed > 0


# --- several models ----------------------------------------------------------


def test_models_embed_into_separate_files_and_progress(conn, tmp_path, library):
    first, second = StubEmbedder(), StubEmbedder()
    a = db.model_id(conn, "model-a", DIM)
    b = db.model_id(conn, "model-b", DIM)
    store_a = store_for(conn, tmp_path / "index", a, DIM)
    store_b = store_for(conn, tmp_path / "index", b, DIM)

    embed_pending(conn, store_a, first, a)
    embed_pending(conn, store_b, second, b, limit=5)

    assert embedded(conn, a) == total_chunks(conn)
    assert embedded(conn, b) == 5
    assert store_a.path != store_b.path
    assert store_a.path.exists() and store_b.path.exists()


def test_one_model_being_complete_says_nothing_about_another(conn, tmp_path, library):
    embedder = StubEmbedder()
    a = db.model_id(conn, "model-a", DIM)
    b = db.model_id(conn, "model-b", DIM)
    embed_pending(conn, store_for(conn, tmp_path / "index", a, DIM), embedder, a)

    assert embedded(conn, a) == total_chunks(conn)
    assert embedded(conn, b) == 0


# --- session limits ----------------------------------------------------------


def test_a_time_limit_stops_cleanly_and_keeps_the_work(conn, setup):
    """qmd capped sessions and discarded partial documents, so resume runs never
    converged. A limit here bounds the wait, never the work retained."""
    embedder, model, store = setup

    report = embed_pending(conn, store, embedder, model, seconds=0.0, batch_size=4)

    assert report.stopped == "time"
    # whatever it managed is committed and will not be recomputed
    assert embedded(conn, model) == report.embedded + report.copied + report.failed
    again = embed_pending(conn, store, embedder, model)
    assert len(embedder.seen) == total_chunks(conn)  # each chunk embedded once


def test_a_chunk_limit_reports_why_it_stopped(conn, setup):
    embedder, model, store = setup
    report = embed_pending(conn, store, embedder, model, limit=8, batch_size=4)
    assert report.stopped == "limit"


def test_finishing_reports_completion(conn, setup):
    embedder, model, store = setup
    assert embed_pending(conn, store, embedder, model).stopped == "complete"


def test_a_generous_time_limit_does_not_truncate(conn, setup):
    embedder, model, store = setup
    report = embed_pending(conn, store, embedder, model, seconds=3600)
    assert report.stopped == "complete"
    assert embedded(conn, model) == total_chunks(conn)


def test_the_limit_never_leaves_a_count_ahead_of_the_vectors(conn, setup):
    """The property a session limit could plausibly break, checked directly."""
    embedder, model, store = setup
    embed_pending(conn, store, embedder, model, limit=9, batch_size=4)

    for seg in conn.execute("SELECT id, chunk_start FROM segments").fetchall():
        done = db.embedded_prefix(conn, model, seg["id"])
        for offset in range(done):
            assert np.linalg.norm(store.read(seg["chunk_start"] + offset)) > 0


def test_a_stop_request_finishes_the_batch_in_flight(conn, setup):
    """Ctrl-C must cost nothing already paid for."""
    embedder, model, store = setup
    stop = {"asked": False}

    def should_stop():
        # ask to stop as soon as any work has been committed
        if embedder.seen:
            stop["asked"] = True
        return stop["asked"]

    report = embed_pending(conn, store, embedder, model, batch_size=4,
                           should_stop=should_stop)

    assert report.stopped == "interrupted"
    assert report.embedded > 0
    # everything the model computed was committed, not discarded
    assert embedded(conn, model) == report.embedded
    for seg in conn.execute("SELECT id, chunk_start FROM segments").fetchall():
        for offset in range(db.embedded_prefix(conn, model, seg["id"])):
            assert np.linalg.norm(store.read(seg["chunk_start"] + offset)) > 0


def test_resuming_after_a_stop_request_recomputes_nothing(conn, setup):
    embedder, model, store = setup
    stop = {"asked": False}
    embed_pending(conn, store, embedder, model, batch_size=4,
                  should_stop=lambda: bool(embedder.seen) and not stop.update(asked=True))

    embed_pending(conn, store, embedder, model)

    assert embedded(conn, model) == total_chunks(conn)
    assert len(embedder.seen) == total_chunks(conn)
    assert len(set(embedder.seen)) == total_chunks(conn)


def test_stopping_before_any_work_reports_interrupted(conn, setup):
    embedder, model, store = setup
    report = embed_pending(conn, store, embedder, model, should_stop=lambda: True)
    assert report.stopped == "interrupted"
    assert report.embedded == 0


# --- what embedding has actually cost, run by run ----------------------------


class _Report:
    def __init__(self, embedded, copied=0, failed=0, stopped="time"):
        self.embedded, self.copied = embedded, copied
        self.failed, self.stopped = failed, stopped


def test_a_run_is_recorded_when_it_stops(conn):
    model = db.model_id(conn, "gemma@aaaaaaaaaaaa", 768)

    db.record_run(conn, model, db.now(), 1800.0, _Report(6462))

    row = conn.execute("SELECT * FROM embed_runs").fetchone()
    assert row["embedded"] == 6462 and row["seconds"] == 1800.0
    assert row["stopped"] == "time"


def test_the_estimate_comes_from_the_median_not_the_mean(conn):
    """One run that spanned a laptop's idle sleep must not move the estimate.

    That is not hypothetical: a 30-minute run reported 1.4 chunks/s against a
    true 5.9, because wall clock counted 49 minutes of sleep.
    """
    model = db.model_id(conn, "gemma@aaaaaaaaaaaa", 768)
    for embedded in (6462, 7519, 7433, 7136):
        db.record_run(conn, model, db.now(), 1800.0, _Report(embedded))
    db.record_run(conn, model, db.now(), 4767.0, _Report(6733))     # the slept run

    rate = db.observed_rate(conn, model)

    assert 3.5 < rate < 4.2, rate
    mean = sum(r["embedded"] for r in conn.execute("SELECT embedded FROM embed_runs")) / (
        sum(r["seconds"] for r in conn.execute("SELECT seconds FROM embed_runs")))
    assert rate > mean, "the median should be less dragged down than the mean"


def test_runs_too_short_to_mean_anything_are_ignored(conn):
    """A run stopped two seconds in says nothing about throughput."""
    model = db.model_id(conn, "gemma@aaaaaaaaaaaa", 768)
    db.record_run(conn, model, db.now(), 1800.0, _Report(7200))     # 4.0/s
    db.record_run(conn, model, db.now(), 2.0, _Report(60))          # 30/s, meaningless

    assert abs(db.observed_rate(conn, model) - 4.0) < 0.01


def test_with_no_history_there_is_no_rate(conn):
    model = db.model_id(conn, "gemma@aaaaaaaaaaaa", 768)
    assert db.observed_rate(conn, model) is None


def test_history_is_per_model(conn):
    a = db.model_id(conn, "one@aaaaaaaaaaaa", 768)
    b = db.model_id(conn, "two@bbbbbbbbbbbb", 384)
    db.record_run(conn, a, db.now(), 1800.0, _Report(7200))

    assert db.observed_rate(conn, a) == 4.0
    assert db.observed_rate(conn, b) is None


def test_dropping_a_model_takes_its_history(conn):
    model = db.model_id(conn, "gemma@aaaaaaaaaaaa", 768)
    db.record_run(conn, model, db.now(), 1800.0, _Report(7200))

    with conn:
        conn.execute("DELETE FROM models WHERE id = ?", (model,))

    assert conn.execute("SELECT COUNT(*) FROM embed_runs").fetchone()[0] == 0


def test_a_run_records_both_clocks_so_sleep_is_recoverable(conn):
    """Monotonic alone cannot say when a run ended.

    A run begun at 03:30 that worked for 79 minutes finished at 04:49 on a
    machine that stayed awake and at 13:00 on one that slept between. The index
    held only the first number, so `dyp history` could show a start and a
    duration that did not add up to anything.
    """
    from dyprys.cli import _asleep

    model = db.model_id(conn, "gemma@aaaaaaaaaaaa", 768)
    db.record_run(conn, model, db.now(), 4773.0, _Report(16588), wall=11000.0)

    row = conn.execute("SELECT * FROM embed_runs").fetchone()
    assert row["seconds"] == 4773.0 and row["wall_seconds"] == 11000.0
    assert _asleep(row) == pytest.approx(6227.0)


def test_a_run_that_did_not_sleep_reports_none(conn):
    """The two clocks never agree exactly; a few seconds is not a sleep."""
    from dyprys.cli import _asleep

    model = db.model_id(conn, "gemma@aaaaaaaaaaaa", 768)
    db.record_run(conn, model, db.now(), 1800.0, _Report(7200), wall=1800.4)

    assert _asleep(conn.execute("SELECT * FROM embed_runs").fetchone()) == 0.0


def test_runs_recorded_before_the_second_clock_existed_still_read(conn):
    """Older rows have no wall clock, and must not be read as never sleeping."""
    from dyprys.cli import _asleep

    model = db.model_id(conn, "gemma@aaaaaaaaaaaa", 768)
    db.record_run(conn, model, db.now(), 1800.0, _Report(7200))

    row = conn.execute("SELECT * FROM embed_runs").fetchone()
    assert row["wall_seconds"] is None
    assert _asleep(row) == 0.0


# --- embedding part of a library ---------------------------------------------


def test_embedding_can_be_scoped_to_some_books(conn, tmp_path, library):
    """At 3,453 books, "do this shelf first" is an ordinary thing to want."""
    from dyprys.embed import embed_pending

    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "ix", model, DIM)
    wanted = conn.execute("SELECT id FROM books ORDER BY id").fetchone()["id"]

    embed_pending(conn, store, StubEmbedder(), model, book_ids={wanted})

    embedded = conn.execute(
        "SELECT src.book_id, SUM(p.n_embedded) AS n FROM segment_progress p "
        "JOIN segments seg ON seg.id = p.segment_id "
        "JOIN sources src ON src.id = seg.source_id "
        "WHERE p.model_id = ? GROUP BY src.book_id", (model,)).fetchall()
    touched = {r["book_id"] for r in embedded if r["n"]}
    assert touched == {wanted}, "a scoped run embedded a book it was not asked for"


def test_a_scoped_run_reports_progress_against_its_own_scope(conn, tmp_path, library):
    """Otherwise a finished shelf reports itself as 4% of the library.

    The total and the progress must be narrowed together; scoping only the work
    leaves the run looking permanently unfinished.
    """
    from dyprys.embed import _outstanding, embed_pending

    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "ix", model, DIM)
    wanted = conn.execute("SELECT id FROM books ORDER BY id").fetchone()["id"]

    scoped_before = _outstanding(conn, model, {wanted})
    whole_before = _outstanding(conn, model)
    assert 0 < scoped_before < whole_before

    embed_pending(conn, store, StubEmbedder(), model, book_ids={wanted})

    assert _outstanding(conn, model, {wanted}) == 0
    assert _outstanding(conn, model) == whole_before - scoped_before


def test_an_empty_scope_embeds_nothing_rather_than_everything(conn, tmp_path, library):
    """A pattern matching no book must not silently widen to the whole library.

    The same rule search already follows: a typo that becomes a full run is the
    expensive kind of mistake here, measured in GPU-hours.
    """
    from dyprys.embed import embed_pending

    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "ix", model, DIM)

    report = embed_pending(conn, store, StubEmbedder(), model, book_ids=set())

    assert report.embedded == 0
    assert conn.execute(
        "SELECT COALESCE(SUM(n_embedded), 0) FROM segment_progress").fetchone()[0] == 0


# --- one model, one chunking --------------------------------------------------


def _two_chunkings(conn, tmp_path):
    """A book split two ways, as `dyp add --target N` twice produces."""
    from dyprys.ingest import ingest_paths

    book = tmp_path / "book.txt"
    book.write_text("\n\n".join(
        f"Paragraph {n} about neurons and synapses here. " * 6 for n in range(60)),
        encoding="utf-8")
    ingest_paths(conn, [book], target=3600)
    ingest_paths(conn, [book], target=900)
    return book


def test_a_model_embeds_one_chunking_only(conn, tmp_path):
    """Two granularities in one vector population is not a thing a reader can read.

    Without this, a second `dyp add --target N` silently doubled the embedding
    bill and returned a passage and a piece of that same passage as separate
    results.
    """
    from dyprys.embed import embed_pending

    _two_chunkings(conn, tmp_path)
    ways = db.chunkings(conn)
    assert len(ways) == 2, "the fixture should produce two chunkings"
    coarse = next(c for c in ways if c["target"] == 3600)

    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "ix", model, DIM)
    embed_pending(conn, store, StubEmbedder(), model, chunking_id=coarse["id"])

    touched = conn.execute(
        "SELECT DISTINCT seg.chunking_id FROM segment_progress p "
        "JOIN segments seg ON seg.id = p.segment_id "
        "WHERE p.model_id = ? AND p.n_embedded > 0", (model,)).fetchall()
    assert [r["chunking_id"] for r in touched] == [coarse["id"]]


def test_search_sees_only_the_chunking_its_model_embedded(conn, tmp_path):
    """No change was needed for this, and that is the point.

    `embedded_ranges` derives from segment_progress, so confining the *embed*
    confines the search for free.
    """
    from dyprys.embed import embed_pending
    from dyprys.search import embedded_ranges

    _two_chunkings(conn, tmp_path)
    coarse = next(c for c in db.chunkings(conn) if c["target"] == 3600)
    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "ix", model, DIM)
    embed_pending(conn, store, StubEmbedder(), model, chunking_id=coarse["id"])

    reachable = {c for lo, hi in embedded_ranges(conn, model) for c in range(lo, hi)}
    other = {r["id"] for r in conn.execute(
        "SELECT ch.id FROM chunks ch JOIN segments seg "
        "ON ch.id >= seg.chunk_start AND ch.id < seg.chunk_start + seg.chunk_count "
        "WHERE seg.chunking_id != ?", (coarse["id"],))}
    assert reachable and not (reachable & other), "search reached the other chunking"


def test_a_model_cannot_be_rebound_to_another_chunking(conn, tmp_path):
    """Its existing vectors are the other granularity; mixing them is the bug."""
    _two_chunkings(conn, tmp_path)
    ways = db.chunkings(conn)
    model = db.model_id(conn, "stub", DIM)
    db.bind_chunking(conn, model, ways[0]["id"])

    db.bind_chunking(conn, model, ways[0]["id"])          # same again is fine
    with pytest.raises(ValueError, match="already embeds"):
        db.bind_chunking(conn, model, ways[1]["id"])


def test_outstanding_work_is_counted_against_the_bound_chunking(conn, tmp_path):
    """It used to count the library, so a finished model was never finished.

    `dyp status` then said "embed the remaining 5 chunks" after every run, for
    chunks that model would never touch — a loop with no exit.
    """
    from dyprys.embed import _outstanding, embed_pending

    _two_chunkings(conn, tmp_path)
    coarse = next(c for c in db.chunkings(conn) if c["target"] == 3600)
    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "ix", model, DIM)
    embed_pending(conn, store, StubEmbedder(), model, chunking_id=coarse["id"])

    assert _outstanding(conn, model, chunking_id=coarse["id"]) == 0
    assert _outstanding(conn, model) > 0, "the other chunking is still unembedded"


def test_each_model_family_gets_its_own_task_prefix():
    """Prefixes are not interchangeable, and using the wrong one does not fail.

    It returns vectors, and they are worse — the shape of every prompt-format
    bug in this project: a wrong reranker template made reranking worse than no
    reranking, and a raw prompt made an expansion model hand back its
    instructions. Embedding nomic with EmbeddingGemma's prefix would have
    measured prefix mismatch and called it model quality.
    """
    from dyprys.embedder import prompts_for

    assert prompts_for("gemma-embedding") == (
        "title: none | text: {}", "task: search result | query: {}")
    assert prompts_for("nomic-bert") == ("search_document: {}", "search_query: {}")
    assert prompts_for("jina-bert-v2") == ("{}", "{}")


def test_an_unknown_architecture_gets_no_prefix():
    """Wrong for a model that wants one, but smaller than another family's."""
    from dyprys.embedder import prompts_for

    assert prompts_for("something-new-2027") == ("{}", "{}")
    assert prompts_for("") == ("{}", "{}")


def test_duty_pauses_in_proportion_to_the_work_just_done(conn, tmp_path, monkeypatch):
    """80% duty must cost 20%, not some number that depends on batch size.

    Sleeping a fixed amount between batches would make the ratio depend on how
    long a batch happens to take, which varies with the model and the chunk
    size. Sleeping in proportion to the batch just finished keeps it right.
    """
    from dyprys import embed as embed_mod
    from dyprys.embed import embed_pending

    library = tmp_path / "lib"
    library.mkdir()
    (library / "b.txt").write_text("\n\n".join(
        f"Paragraph {n} about neurons. " * 10 for n in range(40)), encoding="utf-8")
    from dyprys.ingest import ingest_paths
    ingest_paths(conn, [library])

    slept: list[float] = []
    monkeypatch.setattr(embed_mod.time, "sleep", slept.append)
    ticks = iter(range(0, 10_000))
    monkeypatch.setattr(embed_mod.time, "monotonic", lambda: next(ticks))

    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "ix", model, DIM)
    embed_pending(conn, store, StubEmbedder(), model, duty=0.8, batch_size=4)

    assert slept, "a duty below 1 should have paused"
    # Each batch "took" one tick, so each pause is 1 * (1 - 0.8) / 0.8 = 0.25.
    assert all(abs(s - 0.25) < 1e-9 for s in slept), slept


def test_full_duty_never_pauses(conn, tmp_path, monkeypatch):
    """The default must not add a single sleep call to a 33-day run."""
    from dyprys import embed as embed_mod
    from dyprys.embed import embed_pending
    from dyprys.ingest import ingest_paths

    library = tmp_path / "lib"
    library.mkdir()
    (library / "b.txt").write_text("\n\n".join(
        f"Paragraph {n} about neurons. " * 10 for n in range(20)), encoding="utf-8")
    ingest_paths(conn, [library])

    slept: list[float] = []
    monkeypatch.setattr(embed_mod.time, "sleep", slept.append)

    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "ix", model, DIM)
    embed_pending(conn, store, StubEmbedder(), model)

    assert slept == []


def test_a_scoped_run_reports_what_it_left_out(conn, tmp_path):
    """Finishing a scope looks exactly like finishing the library.

    Without this, `dyp embed -c neuro` completing 119 books of 3,453 printed
    "next: profile the books" — the advice for a finished library — and nothing
    said the other 3,334 were untouched.
    """
    from dyprys.embed import _outstanding, embed_pending
    from dyprys.ingest import ingest_paths
    from dyprys.search import scope_books

    library = tmp_path / "lib"
    library.mkdir()
    for name in ("alpha", "beta"):
        (library / f"{name}.txt").write_text("\n\n".join(
            f"Paragraph {n} about neurons. " * 10 for n in range(30)), encoding="utf-8")
    ingest_paths(conn, [library])

    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "ix", model, DIM)
    scope = scope_books(conn, "alpha")
    assert len(scope) == 1

    embed_pending(conn, store, StubEmbedder(), model, book_ids=scope)

    assert _outstanding(conn, model, scope) == 0, "the scope should be finished"
    left = _outstanding(conn, model) - _outstanding(conn, model, scope)
    assert left > 0, "the unscoped remainder is what the warning reports"
