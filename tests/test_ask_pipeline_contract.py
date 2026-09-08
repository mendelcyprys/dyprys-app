"""What the retrieval core must still do after it is moved out of `cli`.

The service-layer refactor moves `_searcher` from `cli.py` into
`service._pipeline` with its body
"substantially unchanged". That claim is the whole basis for believing the
refactor is safe, and a diff cannot check it: `git diff -M` reads a move as a
move whether or not a line was dropped on the way.

So these tests name the pipeline through a function that finds it in either
home, and assert the properties `_ask`, `_eval` and `--json` actually depend
on. They pass before the move and must pass after it; if they only pass before,
the move changed something.

Everything here is deterministic -- `StubEmbedder` hashes its input, and book 0
holds one phrase that appears nowhere else -- so nothing depends on a GGUF, a
GPU, or a real corpus.
"""

from __future__ import annotations

import pytest

from tests.indexes import SENTINEL, embedded_index, write_books


def pipeline():
    """The retrieval core, wherever it currently lives.

    Named in both homes on purpose. A characterization test for a move has to
    run on both sides of it, which means it cannot import only one.
    """
    try:
        from dyprys.service import _pipeline
        return _pipeline
    except ImportError:
        from dyprys.cli import _searcher
        return _searcher


@pytest.fixture
def index(tmp_path):
    books = write_books(tmp_path / "lib")
    conn, embedder, model_id, store = embedded_index(tmp_path / "ix", books)
    yield conn, embedder, model_id, store
    conn.close()


def test_a_verbatim_phrase_comes_back_first(index):
    """The literal half is the half a paraphrase cannot rescue.

    CLAUDE.md's first rule for choosing a search is "pass the words verbatim";
    the tool keeps that promise by sending an exact phrase past the router and
    past every rewrite. If a move breaks the lexical leg, meaning-search still
    returns five plausible passages at exit 0 and nothing says the guarantee is
    gone -- so this is the assertion most worth having.
    """
    conn, embedder, model_id, store = index
    run = pipeline()(conn, store, embedder, model_id, "hybrid", None)

    hits = run(SENTINEL, 5)

    assert hits, "an exact phrase present in the corpus returned nothing"
    from dyprys.search import resolve
    assert SENTINEL in (resolve(conn, hits[:1])[0].text or ""), (
        "the passage holding the phrase verbatim did not come back first")
    assert "phrase" in (run.why.get(hits[0].chunk_id) or ""), (
        f"rank 1 was not credited to the literal half: {run.why.get(hits[0].chunk_id)!r}")


def test_lexical_only_mode_still_finds_it_without_any_vector(index):
    """`--mode lexical` must not depend on the vector half having run.

    The two halves are fused, and fusion code is exactly where a move loses the
    single-ranking special case.
    """
    conn, embedder, model_id, store = index
    run = pipeline()(conn, store, embedder, model_id, "lexical", None)

    hits = run(SENTINEL, 5)

    from dyprys.search import resolve
    assert hits and SENTINEL in (resolve(conn, hits[:1])[0].text or "")


def test_why_and_cosine_are_attached_to_the_callable(index):
    """`_ask` and `_eval` read `run.why` / `run.cosine` off the function object.

    An attribute on a callable is an unusual contract and precisely the kind a
    refactor to a dataclass return drops silently -- `_eval` (cli.py:2801) is a
    second caller that is easy to forget, and it would break at a
    different time and in a different command from `ask`.
    """
    conn, embedder, model_id, store = index
    run = pipeline()(conn, store, embedder, model_id, "hybrid", None)

    hits = run("neurons and synapses", 5)

    assert hits
    for hit in hits:
        assert hit.chunk_id in run.why, f"chunk {hit.chunk_id} lost its provenance"
        assert hit.chunk_id in run.cosine, f"chunk {hit.chunk_id} lost its cosine"
        assert -1.0001 <= run.cosine[hit.chunk_id] <= 1.0001


def test_a_fresh_callable_starts_with_empty_provenance(index):
    """`run.why, run.cosine = {}, {}` before the first call is not decoration.

    `_ask` reads them for every passage it prints. Left unset, a search that
    returned nothing raises AttributeError in the display rather than saying
    "no passage matched".
    """
    conn, embedder, model_id, store = index
    run = pipeline()(conn, store, embedder, model_id, "hybrid", None)

    assert run.why == {} and run.cosine == {}


def test_k_is_honoured_and_the_order_is_stable(index):
    """Same question, same index, same answer -- twice in a row.

    Cheap, and it is the assertion that catches a moved pipeline that picked up
    a set iteration or a dict ordering somewhere in the fusion.
    """
    conn, embedder, model_id, store = index
    make = pipeline()

    first = [h.chunk_id for h in make(conn, store, embedder, model_id, "hybrid", None)(
        "neurons and synapses", 3)]
    second = [h.chunk_id for h in make(conn, store, embedder, model_id, "hybrid", None)(
        "neurons and synapses", 3)]

    assert len(first) == 3
    assert first == second, "the same search returned a different order"


def test_scoping_confines_the_ranking_half(index):
    """`-c` is a promise about which books were ranked, and the API exposes it
    as a request parameter. A scope that leaks on the ranking half is worse than
    no scope, because the citation still looks right."""
    conn, embedder, model_id, store = index
    from dyprys.search import resolve, scope_books

    only = scope_books(conn, "book0")
    assert only == {1}, "the fixture should scope to exactly book0"
    run = pipeline()(conn, store, embedder, model_id, "hybrid", only)

    # Deliberately not an ordered match anywhere in the corpus, so this exercises
    # the OR-of-words leg and the vector leg -- the two that `book_ids` binds.
    hits = run("synapses concerning neurons", 5)

    assert hits
    outside = [p.path for p in resolve(conn, hits) if "book0" not in str(p.path)]
    assert not outside, f"a scoped search ranked passages from {outside}"


def test_an_exact_phrase_deliberately_escapes_the_scope(index):
    """Documented behaviour, pinned here because it surprises every reader.

    `search_bm25` searches the whole library for an ordered match even when
    `book_ids` is set: confining the phrase attempt to a router's choices took
    lexical safety from 20/20 to 9/20, so the safety net is deliberately not
    tied to the failure it exists to catch.

    The measured case was the *router* guessing books. `-c` is not a guess -- it
    is what the caller asked for -- and it reaches the same argument, so a
    verbatim query under `-c book0` can return book1. That is today's behaviour
    and the refactor must not change it by accident; whether it should change on
    purpose is a decision, and this test is where that decision becomes visible.
    """
    conn, embedder, model_id, store = index
    from dyprys.search import resolve, scope_books

    only = scope_books(conn, "book0")
    run = pipeline()(conn, store, embedder, model_id, "hybrid", only)

    hits = run("neurons and synapses", 5)   # an ordered match in both books
    passages = resolve(conn, hits)

    escaped = [p for p in passages if "book0" not in str(p.path)]
    assert escaped, (
        "the phrase leg no longer searches outside `book_ids`. That may be an "
        "improvement, but it is a behaviour change and lexical safety was "
        "measured at 20/20 with it and 9/20 without -- re-measure before "
        "deleting this test")
    assert all("phrase" in (run.why.get(p.chunk_id) or "") for p in escaped), (
        "only the exact-phrase leg may leave the scope; a words or vector hit "
        "from outside `book_ids` is a real leak")
