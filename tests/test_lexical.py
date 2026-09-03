"""BM25, and fusing it with vector search by rank."""

import pytest

from dyprys import db
from dyprys.embed import store_for
from dyprys.ingest import ingest_paths
from dyprys.lexical import (
    backfill,
    drop_chunks,
    indexed_count,
    reciprocal_rank_fusion,
    search_bm25,
    to_match_query,
)
from dyprys.search import Hit

DIM = 8


@pytest.fixture
def indexed(conn, tmp_path, library):
    """Every chunk lexically indexed and marked embedded."""
    model = db.model_id(conn, "stub", DIM)
    store_for(conn, tmp_path / "index", model, DIM)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    return model


# --- query handling ----------------------------------------------------------


def test_punctuation_does_not_become_syntax(conn):
    """A natural question is a MATCH syntax error unless the terms are quoted."""
    assert to_match_query("what is long-term potentiation?") == (
        '"what" OR "is" OR "long" OR "term" OR "potentiation"'
    )


def test_a_query_of_only_punctuation_is_empty_not_broken(conn):
    assert to_match_query("?!  -- ") == ""


def test_an_empty_query_returns_nothing_rather_than_raising(conn, indexed):
    assert search_bm25(conn, "???", k=5) == []


# --- retrieval ---------------------------------------------------------------


def test_it_finds_the_passage_containing_a_phrase(conn, indexed, library):
    """The case vector search is measurably bad at."""
    hits = search_bm25(conn, "Paragraph 7 of book 1", k=5, model_id=indexed)

    assert hits
    from dyprys.evaluate import _chunk_text
    assert "paragraph 7 of book 1" in _chunk_text(conn, hits[0].chunk_id).lower()


def test_larger_score_means_better_for_every_hit_in_the_system(conn, indexed):
    """bm25() is negative and ascending; Hit.score must not be."""
    hits = search_bm25(conn, "neurons paragraph", k=5, model_id=indexed)
    assert hits
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_it_is_restricted_to_a_scoped_set_of_books(conn, indexed, library):
    from dyprys.search import scope_books
    wanted = scope_books(conn, "book1")

    hits = search_bm25(conn, "neurons", k=20, model_id=indexed, book_ids=wanted)

    assert hits
    for hit in hits:
        assert db.locate(conn, hit.chunk_id)["book_id"] in wanted


def test_it_only_returns_chunks_that_also_have_a_vector(conn, indexed, library):
    """Both halves of a hybrid must be drawn from the same population."""
    segment = conn.execute("SELECT id, chunk_start FROM segments ORDER BY id").fetchone()
    with conn:
        db.set_embedded_prefix(conn, indexed, segment["id"], 2)

    from dyprys.search import embedded_ranges
    reachable = {c for s, e in embedded_ranges(conn, indexed) for c in range(s, e)}
    hits = search_bm25(conn, "neurons", k=50, model_id=indexed)

    assert hits
    assert all(h.chunk_id in reachable for h in hits)


# --- staying in step with the chunks -----------------------------------------


def test_ingest_indexes_chunks_without_a_second_pass(conn, library):
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert indexed_count(conn) == total


def test_superseded_chunks_leave_the_index(conn, library):
    """A dead chunk id must never win a query."""
    before = indexed_count(conn)
    book = library / "book0.txt"
    book.write_text(book.read_text(encoding="utf-8") + "\n\nA tail.\n", encoding="utf-8")
    ingest_paths(conn, [library])

    live = conn.execute("SELECT COALESCE(SUM(chunk_count), 0) FROM segments").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert indexed_count(conn) == live
    assert total > live  # dead chunks exist, and are not in the index
    assert indexed_count(conn) != before or live == before


def test_dropping_chunks_removes_exactly_those(conn, library):
    before = indexed_count(conn)
    with conn:
        drop_chunks(conn, [1, 2])
    assert indexed_count(conn) == before - 2


def test_backfill_rebuilds_the_index_for_an_older_library(conn, library):
    with conn:
        conn.execute("DELETE FROM chunks_fts")
    assert indexed_count(conn) == 0

    n = backfill(conn)

    live = conn.execute("SELECT COALESCE(SUM(chunk_count), 0) FROM segments").fetchone()[0]
    assert n == live
    assert search_bm25(conn, "neurons", k=3)


# --- fusion ------------------------------------------------------------------


def test_fusion_rewards_agreement_between_the_two_halves(conn):
    vector = [Hit(1, 0.9), Hit(2, 0.8), Hit(3, 0.7)]
    words = [Hit(3, 12.0), Hit(1, 9.0)]

    fused = reciprocal_rank_fusion([vector, words], k=3)

    # 1 and 3 appear in both lists; 2 appears in one, so it must not lead
    assert {fused[0].chunk_id, fused[1].chunk_id} == {1, 3}
    assert fused[2].chunk_id == 2


def test_fusion_ignores_the_scales_it_is_given(conn):
    """BM25 and cosine are not comparable numbers; only the ranks are used."""
    a = [Hit(7, 0.51), Hit(8, 0.50)]
    b = [Hit(7, 900.0), Hit(8, 1.0)]
    huge = [Hit(7, 5e9), Hit(8, 1e-9)]

    assert [h.chunk_id for h in reciprocal_rank_fusion([a, b], k=2)] == \
           [h.chunk_id for h in reciprocal_rank_fusion([a, huge], k=2)]


def test_fusion_of_one_list_preserves_its_order(conn):
    ranked = [Hit(5, 0.9), Hit(6, 0.8), Hit(7, 0.1)]
    assert [h.chunk_id for h in reciprocal_rank_fusion([ranked], k=3)] == [5, 6, 7]


def test_fusion_of_nothing_is_nothing(conn):
    assert reciprocal_rank_fusion([[], []], k=5) == []


def test_weights_shift_the_balance(conn):
    vector = [Hit(1, 0.9)]
    words = [Hit(2, 9.0)]

    assert reciprocal_rank_fusion([vector, words], k=1, weights=[10.0, 1.0])[0].chunk_id == 1
    assert reciprocal_rank_fusion([vector, words], k=1, weights=[1.0, 10.0])[0].chunk_id == 2


def test_single_digits_survive_tokenisation(conn):
    """Dropping them rewrote "Paragraph 7 of book 1" into "paragraph of book"."""
    from dyprys.lexical import to_phrase_query
    assert to_phrase_query("Paragraph 7 of book 1") == '"paragraph 7 of book 1"'
    assert to_match_query("figure 3") == '"figure" OR "3"'


def test_a_phrase_query_lands_on_the_passage_that_contains_it(conn, indexed):
    """Adjacency is the whole point: an OR of the same words cannot promise it."""
    from dyprys.evaluate import _chunk_text
    from dyprys.lexical import _match, to_phrase_query

    hits = _match(conn, to_phrase_query("Paragraph 7 of book 1"), 3, None)

    assert hits
    for hit in hits:
        assert "paragraph 7 of book 1" in _chunk_text(conn, hit.chunk_id).lower()


def test_bm25_survives_scattered_progress(conn, indexed):
    """Merged ranges make this rare, not impossible — and rare is how the last
    version of this bug passed every test and broke at 1,000 books."""
    from dyprys.lexical import MAX_RANGE_CLAUSES, search_bm25
    from dyprys.search import embedded_ranges
    from tests.test_search import make_segments

    model = db.model_id(conn, "scattered", DIM)
    segments = make_segments(conn, model, MAX_RANGE_CLAUSES * 3, per=4)
    with conn:                       # leave every other segment partial
        for seg in segments[::2]:
            db.set_embedded_prefix(conn, model, seg, 2)

    assert len(embedded_ranges(conn, model)) > MAX_RANGE_CLAUSES
    search_bm25(conn, "neurons paragraph", k=5, model_id=model)  # must not raise


def test_the_python_filter_returns_only_scoped_chunks(conn, indexed):
    from dyprys.lexical import _match
    scope = [(1, 3)]
    hits = _match(conn, '"neurons"', 10, scope)
    assert hits
    assert all(1 <= h.chunk_id < 3 for h in hits)


def test_scoping_to_books_needs_a_model(conn):
    """The scope comes from a model's embedded ranges; without one it silently
    searched the whole corpus, which is the wrong answer to "these books"."""
    with pytest.raises(ValueError, match="model_id"):
        search_bm25(conn, "neurons", book_ids={1})


# --- the phrase attempt is not confined to the routed books ------------------
#
# `book_ids` comes from a router that scores books by vector similarity. On 117
# books it keeps the right book for 109/110 natural questions but only 25/60
# verbatim phrases — so routing the phrase search tied the lexical safety net to
# the very failure it exists to catch, taking it from 20/20 to 9/20.


@pytest.fixture
def two_books(conn, tmp_path):
    """Two books; the distinctive sentence is in the second one only."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "a.txt").write_text(
        "\n\n".join(f"Paragraph {i} about neurons and synapses. " * 12 for i in range(20)),
        encoding="utf-8")
    (lib / "b.txt").write_text(
        "\n\n".join(f"Section {i} concerning glial cells. " * 12 for i in range(20))
        + "\n\nThe quokka disembarked at Rottnest before the surveyors arrived.\n",
        encoding="utf-8")
    ingest_paths(conn, [lib])
    model = db.model_id(conn, "stub", DIM)
    store_for(conn, tmp_path / "index", model, DIM)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    books = {row["key"]: row["id"] for row in conn.execute("SELECT id, key FROM books")}
    holder = next(v for k, v in books.items() if k.endswith("b.txt"))
    other = next(v for k, v in books.items() if k.endswith("a.txt"))
    return model, holder, other


PHRASE = "quokka disembarked at Rottnest before the surveyors"


def test_a_remembered_phrase_survives_being_routed_past(conn, two_books):
    """Stage 1 picked the wrong book; the phrase is found regardless."""
    model, holder, other = two_books

    found = search_bm25(conn, PHRASE, 5, model_id=model, book_ids={other})

    assert found, "the phrase was in the library and was not returned"
    owner = db.locate(conn, found[0].chunk_id)
    assert owner["book_id"] == holder


def test_the_old_behaviour_is_still_reachable_for_measurement(conn, two_books):
    model, holder, other = two_books

    assert not search_bm25(conn, PHRASE, 5, model_id=model, book_ids={other},
                           phrase_everywhere=False)


def test_the_or_query_stays_inside_the_routed_books(conn, two_books):
    """Only the phrase escapes the scope. A ranking over common words does not.

    Otherwise `--route` would stop meaning anything for the lexical half, and
    the whole index would be ranked on every query.
    """
    model, holder, other = two_books

    # Words that appear in both books, so nothing matches as an ordered phrase.
    found = search_bm25(conn, "neurons glial paragraph section", 5,
                        model_id=model, book_ids={other})

    assert found
    assert all(db.locate(conn, h.chunk_id)["book_id"] == other for h in found)


def test_an_unrouted_search_is_unaffected(conn, two_books):
    """With no book_ids there is no second scope to compute, and no extra query."""
    model, holder, _ = two_books

    found = search_bm25(conn, PHRASE, 5, model_id=model)

    assert found and db.locate(conn, found[0].chunk_id)["book_id"] == holder


# --- fusion ties, which are the common case ---------------------------------


def test_ties_go_to_the_earlier_ranking():
    """Rank r of one list scores exactly rank r of another, so ties are constant.

    On 138 questions the top of the fused list was decided by an exact tie 82
    times, and which side won them was worth 18 questions of recall@1. This
    pins the rule so a refactor cannot quietly reverse it.
    """
    first = [Hit(11, 0.9), Hit(12, 0.8)]
    second = [Hit(21, 500.0), Hit(22, 400.0)]

    fused = reciprocal_rank_fusion([first, second], k=4)

    assert [h.chunk_id for h in fused] == [11, 21, 12, 22]
    assert fused[0].score == fused[1].score      # a genuine tie, broken by order
    assert fused[2].score == fused[3].score


def test_swapping_the_rankings_swaps_the_tie_winner():
    first = [Hit(11, 0.9)]
    second = [Hit(21, 0.9)]

    assert reciprocal_rank_fusion([first, second], k=2)[0].chunk_id == 11
    assert reciprocal_rank_fusion([second, first], k=2)[0].chunk_id == 21


def test_agreement_still_beats_any_single_ranking():
    """The tie-break must not override the thing fusion is for."""
    first = [Hit(11, 0.9), Hit(99, 0.5)]
    second = [Hit(21, 0.9), Hit(99, 0.5)]

    fused = reciprocal_rank_fusion([first, second], k=3)

    assert fused[0].chunk_id == 99, "a hit in both lists must outrank either list's first"


def test_the_order_is_total_even_for_identical_input():
    """No dependence on dict iteration: equal scores fall back to the id."""
    ranking = [Hit(7, 1.0), Hit(3, 1.0)]

    once = reciprocal_rank_fusion([ranking, list(reversed(ranking))], k=2)
    again = reciprocal_rank_fusion([ranking, list(reversed(ranking))], k=2)

    assert [h.chunk_id for h in once] == [h.chunk_id for h in again]


def test_weighting_a_half_can_push_the_other_off_the_end():
    """Why equal weights are load-bearing, in miniature.

    Measured full-size: a 1.3x weight on the vector half took exact-phrase
    lexical safety from 29/30 to 5/30, because every vector hit then outranks
    every lexical hit at the same rank.
    """
    vector = [Hit(i, 0.5) for i in range(1, 6)]
    lexical = [Hit(99, 40.0)]                     # the exact-phrase match

    equal = reciprocal_rank_fusion([vector, lexical], k=5)
    tilted = reciprocal_rank_fusion([vector, lexical], k=5, weights=[1.3, 1.0])

    assert 99 in [h.chunk_id for h in equal], "equal weights keep the phrase match"
    assert 99 not in [h.chunk_id for h in tilted], "a 1.3x nudge drops it entirely"


def test_damping_cannot_reorder_two_lists():
    vector = [Hit(1, 0.9), Hit(2, 0.8), Hit(3, 0.7)]
    lexical = [Hit(3, 40.0), Hit(4, 30.0)]

    orders = {tuple(h.chunk_id for h in reciprocal_rank_fusion(
        [vector, lexical], k=5, damping=d)) for d in (10, 30, 60, 120, 1000)}

    assert len(orders) == 1, f"damping changed the order: {orders}"


# --- saying why a result ranked where it did ---------------------------------


def test_provenance_names_the_rank_each_half_gave():
    """The fused score cannot say this, and printing it invited a false reading.

    An RRF score is sum(1/(60+rank)): for two lists it spans 0.016 to 0.033, so
    a perfect match prints as 0.033 — which reads as 3% — and two passages that
    each led one half tie exactly with nothing to separate them.
    """
    from dyprys.lexical import provenance

    vec = [Hit(7, 0.9), Hit(8, 0.8)]
    words = [Hit(8, 4.0), Hit(9, 3.0)]
    why = provenance([("vec", vec), ("bm25", words)], [Hit(8, 0.0), Hit(7, 0.0), Hit(9, 0.0)])

    assert why[8] == "vec 2 · bm25 1"      # both halves found it
    assert why[7] == "vec 1"               # semantic match only
    assert why[9] == "bm25 2"              # literal match only


def test_provenance_covers_a_hit_no_half_reported():
    """A reranker can promote something; the display must not raise on it."""
    from dyprys.lexical import provenance

    assert provenance([("vec", [Hit(1, 1.0)])], [Hit(99, 0.0)])[99] == "—"


def test_an_exact_phrase_hit_is_labelled_as_one(conn, library):
    """`phrase` and `words` are different kinds of answer to a reader.

    "This passage contains your sentence" and "this passage shares your words"
    are not the same claim, and the BM25 score cannot tell them apart — the two
    attempts are separate MATCH queries, so on a real index a remembered
    sentence returned 13.2 first and 46.6 second, the 13.2 being the exact
    phrase. Sorting by that number would bury the only literal match.
    """
    from dyprys.lexical import backfill, search_bm25

    backfill(conn)
    origin = {}
    hits = search_bm25(conn, "Paragraph 7 of book 0, about neurons", 5, origin=origin)

    assert hits
    assert origin[hits[0].chunk_id] == "phrase", "the exact phrase should lead"


def test_a_natural_question_falls_through_to_words(conn, library):
    """A question rarely occurs verbatim, so the OR is what answers it."""
    from dyprys.lexical import backfill, search_bm25

    backfill(conn)
    origin = {}
    search_bm25(conn, "what do neurons and synapses actually do here", 5, origin=origin)

    assert origin and set(origin.values()) == {"words"}


def test_provenance_can_relabel_a_side_per_hit():
    """So one lexical half can report `phrase 1` and `words 2` separately."""
    from dyprys.lexical import provenance

    words = [Hit(5, 13.2), Hit(6, 46.6)]
    why = provenance([("bm25", words)], [Hit(5, 0.0), Hit(6, 0.0)],
                     rename={"bm25": {5: "phrase", 6: "words"}})

    assert why[5] == "phrase 1" and why[6] == "words 2"
