"""The service layer: everything the CLI knows, said as values instead of prose.

These are the specification for `errors.py` and `service.py`, and they skip
until those land. What
they check is not that the code was moved -- `test_ask_pipeline_contract.py`
does that -- but the four things the move is *for*, each of which the current
`cli.py` gets wrong in a way that only matters once there is a second frontend:

  * a failure is a typed value, not a `print` and an exit code;
  * what a search learned about itself (its expansion, its summary, its
    warnings) is *returned*, not left in a module-level dict;
  * the history is written once, by whoever actually searched;
  * a path is served only if this index owns it.

The signatures asserted here are the intended ones. If the implementation
picks different ones, change them here in one place -- but changing an
assertion to match a behaviour is how a specification stops being one.
"""

from __future__ import annotations

import pytest

service = pytest.importorskip("dyprys.service", reason="dyprys.service does not exist yet")
errors = pytest.importorskip("dyprys.errors", reason="dyprys.errors does not exist yet")

from tests.indexes import SENTINEL, add_model, embedded_index, write_books  # noqa: E402


@pytest.fixture
def session(tmp_path):
    """A one-model index, open through whatever `open_session` turns out to be."""
    books = write_books(tmp_path / "lib")
    conn, embedder, model_id, store = embedded_index(tmp_path / "ix", books)
    conn.close()
    with service.open_session(data=tmp_path / "ix",
                              load_model=lambda *a, **k: (embedder, model_id, store)) as s:
        yield s


# --------------------------------------------------------------------------
# Failures are values
# --------------------------------------------------------------------------

def test_an_empty_query_raises_instead_of_printing(session):
    """Today this is `print(...); return 2`, which a server cannot translate.

    The refusal itself matters: an empty query embeds to a meaningless vector
    and matches no words, so hybrid search returns whatever the vector half
    drifts to -- noise presented as answers, at exit 0.
    """
    for blank in ("", "   ", "\t \n"):
        with pytest.raises(errors.EmptyQuery):
            service.search(session, blank, service.SearchOptions())


def test_an_unmatched_collection_names_the_pattern_that_matched_nothing(session):
    """`-c` typed wrong must not silently widen to the whole library.

    The message has to carry the pattern, because "no book matches" without it
    is unactionable in a UI where the pattern came from a dropdown.
    """
    with pytest.raises(errors.NoSuchBook) as caught:
        service.search(session, "neurons",
                       service.SearchOptions(collection="no-such-shelf"))
    assert "no-such-shelf" in caught.value.message


def test_a_multi_model_index_refuses_and_says_which_models(tmp_path):
    """The one error the API must return as data, not prose.

    CLAUDE.md's rule is that search refuses rather than guess which vectors to
    answer from. A web UI can only offer the picker that unblocks the user if
    the choices arrive as a list -- so `choices` is part of the contract, not
    a nicety of the message.
    """
    books = write_books(tmp_path / "lib")
    conn, embedder, first, _ = embedded_index(tmp_path / "ix", books, name="stub-alpha")
    add_model(conn, tmp_path / "ix", "stub-beta")
    conn.close()

    with pytest.raises(errors.ModelAmbiguous) as caught:
        with service.open_session(data=tmp_path / "ix") as s:
            service.search(s, "neurons", service.SearchOptions())

    assert caught.value.choices, "ModelAmbiguous must carry the models to choose from"
    assert {"stub-alpha", "stub-beta"} <= {str(c) for c in caught.value.choices}


def test_rerank_without_a_reranker_is_an_error_not_a_silent_fallback(session):
    """CLAUDE.md: bare `--rerank` errors rather than falling back to anything.

    The reranker is the one model the index does not remember, so this is the
    error users will actually hit -- and a fallback would silently return
    unreranked results while the caller believes they were rescored.
    """
    with pytest.raises(errors.OptionalModelMissing) as caught:
        service.search(session, "neurons",
                       service.SearchOptions(rerank=10, reranker=None))
    assert caught.value.role == "reranker"


def test_an_unknown_library_raises_rather_than_exiting_the_process(tmp_path):
    """`cli._where` raises SystemExit today. In a server that is a 500 at best."""
    with pytest.raises(errors.NoSuchLibrary):
        service.resolve_index(library="definitely-not-registered", data=None, env={})


def test_resolution_takes_the_environment_as_an_argument(tmp_path):
    """Explicit beats named beats default -- and none of it is read from os.environ.

    Passing `env` in makes the precedence testable and stops a server from
    answering according to the shell that launched it.
    """
    assert service.resolve_index(library=None, data=tmp_path / "ix", env={}) == tmp_path / "ix"
    assert service.resolve_index(
        library=None, data=tmp_path / "ix",
        env={"DYPRYS_DATA": "/somewhere/else"}) == tmp_path / "ix"


# --------------------------------------------------------------------------
# A search reports on itself
# --------------------------------------------------------------------------

def test_the_result_carries_its_own_expansion(session):
    """`cli._last_expansion` is a module-level dict, filled inside the pipeline
    and read back when the question is recorded.

    One process, two frontends, and a concurrency model that gives each
    library its own thread -- so two searches in flight write the same dict and
    the loser's expansion is recorded against the winner's question. Returning
    it is the fix; this asserts the fix rather than the dict.
    """
    class StubExpander:
        model = "stub-expander"

        def expand(self, query):
            from dyprys.expand import Expansion
            return Expansion(vector=[f"rephrased: {query}"])

    result = service.search(session, "neurons",
                            service.SearchOptions(expand="stub-expander",
                                                  expander=StubExpander()))

    assert result.expansion, "the expansion is part of the result, not a side effect"
    assert "rephrased: neurons" in str(result.expansion)


def test_two_searches_do_not_see_each_others_expansion(session):
    """The concrete failure the module-level dicts cause.

    Sequential here because it is deterministic; the threaded version is the
    same bug arriving nondeterministically. If this passes only because the
    second search overwrote the first, run them in the other order.
    """
    from dyprys.expand import Expansion

    class Fixed:
        def __init__(self, line):
            self.model, self._line = "stub-expander", line

        def expand(self, query):
            return Expansion(vector=[self._line])

    first = service.search(session, "neurons",
                           service.SearchOptions(expand="x", expander=Fixed("ALPHA")))
    second = service.search(session, "synapses",
                            service.SearchOptions(expand="x", expander=Fixed("BETA")))

    assert "ALPHA" in str(first.expansion) and "BETA" not in str(first.expansion)
    assert "BETA" in str(second.expansion) and "ALPHA" not in str(second.expansion)


def test_unreachable_books_are_a_warning_on_the_result(tmp_path):
    """CLAUDE.md calls this the one failure a reader cannot otherwise detect.

    A book embedded since the last `dyp route` has no profile, so `--route`
    cannot return it at any rank -- and the search still returns a full `k` and
    exits 0, so nothing in the output says part of the library was skipped.
    `cli._router` prints the warning and continues; over HTTP a print is a
    warning that does not exist, so it has to ride on the payload.
    """
    books = write_books(tmp_path / "lib")
    conn, embedder, model_id, store = embedded_index(tmp_path / "ix", books)
    conn.close()

    with service.open_session(
            data=tmp_path / "ix",
            load_model=lambda *a, **k: (embedder, model_id, store)) as s:
        with pytest.raises(errors.NoRoutingProfile):
            service.search(s, "neurons", service.SearchOptions(route=8))


def test_a_search_that_was_not_routed_warns_about_nothing(session):
    """`warnings` must be empty rather than absent, so a caller can always read it."""
    result = service.search(session, "neurons", service.SearchOptions())
    assert result.warnings == []


# --------------------------------------------------------------------------
# The history is written once, by whoever searched
# --------------------------------------------------------------------------

def test_a_search_records_the_question_exactly_once(session):
    """`dyp asked` is cross-session memory, and this makes it cross-*interface*.

    Two hazards, opposite directions. Moving the call into `search` while
    leaving `cli._ask`'s call in place records every terminal search twice.
    Moving it and forgetting the caller records none.
    """
    from dyprys import db

    before = len(db.questions_asked(session.conn, limit=50))
    service.search(session, SENTINEL, service.SearchOptions())
    after = db.questions_asked(session.conn, limit=50)

    assert len(after) == before + 1
    assert after[0]["question"] == SENTINEL


def test_the_json_shaped_search_is_recorded_too(session):
    """Today it is not, and that is the behaviour change to make deliberately.

    `cli._ask` returns from the `--json` branch before `record_question`, so
    every agent-driven search this project's own CLAUDE.md prescribes is absent
    from `dyp asked`. Recording inside `search` fixes it for both frontends at
    once -- but it *is* a change, and an anti-drift test that compares
    `dyp asked --json` across two runs will now see rows appear.
    """
    from dyprys import db

    before = len(db.questions_asked(session.conn, limit=50))
    service.search(session, "neurons", service.SearchOptions(full=True))
    assert len(db.questions_asked(session.conn, limit=50)) == before + 1


# --------------------------------------------------------------------------
# The reader pane
# --------------------------------------------------------------------------

def test_a_source_window_is_served_only_for_a_path_this_index_owns(session, tmp_path):
    """The endpoint takes a path from a query string. That is a file-read
    primitive unless it is bounded by the `sources` table, and no amount of
    prefix-checking is equivalent: symlinks, `..`, and a second library's files
    all pass a prefix test."""
    outsider = tmp_path / "secret.txt"
    outsider.write_text("not part of the corpus", encoding="utf-8")

    for path in (outsider, "/etc/passwd", "../../etc/passwd", tmp_path / "lib"):
        with pytest.raises(errors.NoSuchBook):
            service.source_window(session, str(path), 0, 400)


def test_a_source_window_reads_a_real_book(session):
    """The path a result already handed back must work, or the pane is useless."""
    row = session.conn.execute("SELECT path FROM sources LIMIT 1").fetchone()

    text, offset = service.source_window(session, row["path"], 0, 400)

    assert text and isinstance(text, str)
    assert offset >= 0


def test_a_source_window_span_is_capped(session):
    """An uncapped span turns one request into "send me this whole book"."""
    row = session.conn.execute("SELECT path FROM sources LIMIT 1").fetchone()

    text, _ = service.source_window(session, row["path"], 0, 10 ** 9)

    assert len(text.encode()) <= service.MAX_SPAN


def test_a_summariser_retry_does_not_rewrite_what_the_search_returned(session, monkeypatch):
    """The retry is a second search, and a second search rebinds the pipeline.

    `_pipeline` attaches `why`, `cosine` and `expansion` to the callable and
    overwrites all three on every call. The summariser's retry calls it again --
    with the model's own rephrasing, over different books -- so anything that
    reads those attributes *after* summarising describes the rephrasing's
    results while pointing at the passages the caller was handed.

    Found by diffing real output: rank 3's cosine moved 0.15 to 0.20 while the
    passage printed above it did not change. Nothing failed, and the citation
    still looked right.
    """
    from dyprys import summarise as summarise_mod
    from dyprys.summarise import NO_ANSWER

    said = []

    def talk(model, prompt, *a, **k):
        # Refuse, offer other words, refuse again -- the shape that makes the
        # retry run and then give up, so the first search's results stand.
        said.append(prompt)
        return SENTINEL if len(said) == 2 else NO_ANSWER

    monkeypatch.setattr(summarise_mod, "ask_ollama", talk)

    result = service.search(session, "neurons and synapses",
                            service.SearchOptions(k=3, summarise="stub-summariser"))

    assert len(said) == 3, "the retry did not run, so this proves nothing"
    assert result.answer.failed == SENTINEL
    for passage in result.passages:
        assert passage.chunk_id in result.why, (
            f"chunk {passage.chunk_id} was returned with provenance belonging "
            f"to a different search")
        assert passage.chunk_id in result.cosine
