"""The harness measures the passage, and refuses to flatter itself."""

import json

import pytest

from dyprys import db
from dyprys.embed import store_for
from dyprys.evaluate import Question, evaluate, load_questions, locate_terms
from dyprys.search import Hit

DIM = 8


@pytest.fixture
def embedded(conn, tmp_path, library):
    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "index", model, DIM)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    return model, store


def chunk_containing(conn, model, term):
    from dyprys.evaluate import chunk_texts
    for chunk_id, _book, text in chunk_texts(conn, model):
        if term.lower() in text.lower():
            return chunk_id
    return None


def chunk_without(conn, model, term):
    from dyprys.evaluate import chunk_texts
    for chunk_id, _book, text in chunk_texts(conn, model):
        if term.lower() not in text.lower():
            return chunk_id
    return None


# "neurons" is in every chunk of this fixture; this phrase is in exactly one.
UNIQUE = "Paragraph 7 of book 1"


def test_terms_are_located_in_the_embedded_corpus(conn, embedded):
    model, _ = embedded
    found = locate_terms(conn, model, ["neurons", "definitely-not-in-any-book"])

    assert found["neurons"]
    assert found["neurons"].books                      # for routing recall
    assert not found["definitely-not-in-any-book"]


def test_a_term_absent_from_the_corpus_is_excluded_not_counted_as_a_miss(conn, embedded, embedder):
    """Otherwise the score measures the corpus, not the retrieval."""
    model, store = embedded
    questions = [Question("what is a flibbertigibbet?", "flibbertigibbet")]

    report = evaluate(conn, store, embedder, model, questions,
                      search_fn=lambda v, k: [])

    assert report.asked == 1
    assert report.answerable == 0
    assert report.unanswerable == ["flibbertigibbet"]
    assert report.misses == []


def test_a_hit_at_rank_one_counts_for_both(conn, embedded, embedder):
    model, store = embedded
    target = chunk_containing(conn, model, "neurons")
    report = evaluate(conn, store, embedder, model, [Question("q", "neurons")],
                      search_fn=lambda v, k: [Hit(target, 0.9)])

    assert report.hit_at_1 == 1
    assert report.hit_at_5 == 1
    assert report.recall_at_1 == 1.0


def test_a_hit_further_down_counts_only_at_k(conn, embedded, embedder):
    """Hit@1 and Hit@5 must not be the same measurement wearing two names."""
    model, store = embedded
    target = chunk_containing(conn, model, UNIQUE)
    other = chunk_without(conn, model, UNIQUE)
    assert target and other and target != other
    report = evaluate(conn, store, embedder, model, [Question("q", UNIQUE)],
                      search_fn=lambda v, k: [Hit(other, 0.9), Hit(target, 0.8)])

    assert report.hit_at_1 == 0
    assert report.hit_at_5 == 1


def test_a_real_miss_is_recorded_with_its_question(conn, embedded, embedder):
    model, store = embedded
    report = evaluate(conn, store, embedder, model, [Question("where is it?", "neurons")],
                      search_fn=lambda v, k: [])

    assert report.hit_at_5 == 0
    assert report.misses == [("where is it?", "neurons")]


def test_matching_the_book_is_not_enough_only_the_passage_counts(conn, embedded, embedder):
    """The whole point: document-level hits cannot see chunk selection."""
    model, store = embedded
    target = chunk_containing(conn, model, UNIQUE)
    assert target is not None
    # a different chunk of the same book — right document, wrong passage
    wrong = chunk_without(conn, model, UNIQUE)
    report = evaluate(conn, store, embedder, model, [Question("q", UNIQUE)],
                      search_fn=lambda v, k: [Hit(wrong, 0.99)])

    assert report.hit_at_5 == 0


def test_questions_load_from_disk(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([["a question", "a term"]]))

    assert load_questions(path) == [Question("a question", "a term", "lookup")]


def test_scanned_fraction_is_recorded_for_comparison(conn, embedded, embedder):
    model, store = embedded
    report = evaluate(conn, store, embedder, model, [Question("q", "neurons")],
                      search_fn=lambda v, k: [], scanned=0.156)

    assert report.scanned == 0.156


def test_lexical_safety_reads_only_what_it_samples(conn, embedded, embedder, monkeypatch):
    """Reading every passage to pick twenty is the qmd failure in miniature."""
    from dyprys import evaluate as ev
    model, store = embedded
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert total >= 10  # enough that sampling 5 is meaningfully fewer

    reads = []
    real = ev.read_span
    monkeypatch.setattr(ev, "read_span", lambda *a, **k: reads.append(1) or real(*a, **k))

    ev.lexical_safety(conn, store, embedder, model, samples=5, search_fn=lambda v, k: [])

    assert len(reads) <= 5
    assert len(reads) < total


def test_lexical_safety_counts_a_phrase_finding_its_own_passage(conn, embedded, embedder):
    from dyprys.evaluate import lexical_safety
    from dyprys.search import Hit
    model, store = embedded

    # a search that always returns the chunk asked about
    seen = {}
    def perfect(vector, k):
        return [Hit(seen["current"], 1.0)]

    from dyprys import evaluate as ev
    original = ev._chunk_text
    def spy(c, chunk_id):
        seen["current"] = chunk_id
        return original(c, chunk_id)
    ev._chunk_text = spy
    try:
        hits, asked = lexical_safety(conn, store, embedder, model, samples=4, search_fn=perfect)
    finally:
        ev._chunk_text = original

    assert asked > 0 and hits == asked


def test_lexical_safety_on_an_unembedded_index_is_zero(conn, tmp_path, library, embedder):
    from dyprys.embed import store_for
    from dyprys.evaluate import lexical_safety
    model = db.model_id(conn, "fresh", DIM)
    store = store_for(conn, tmp_path / "index", model, DIM)

    assert lexical_safety(conn, store, embedder, model) == (0, 0)


# --- question kinds and measured difficulty ----------------------------------


def test_the_old_pair_format_still_loads(tmp_path):
    """An existing question file must not stop working."""
    path = tmp_path / "q.json"
    path.write_text(json.dumps([["a question", "a term"]]))
    assert load_questions(path) == [Question("a question", "a term", "lookup")]


def test_the_object_format_carries_a_kind(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"q": "why", "a": "term", "kind": "oblique"}]))
    assert load_questions(path) == [Question("why", "term", "oblique")]


def test_recall_is_reported_per_kind(conn, embedded, embedder):
    from dyprys.search import Hit
    model, store = embedded
    target = chunk_containing(conn, model, "neurons")

    report = evaluate(conn, store, embedder, model, [
        Question("a", "neurons", "lookup"),
        Question("b", "neurons", "oblique"),
    ], search_fn=lambda q, k: [Hit(target, 0.9)])

    assert report.by_kind["lookup"].hit_at_1 == 1
    assert report.by_kind["oblique"].hit_at_1 == 1
    assert sum(t.answerable for t in report.by_kind.values()) == report.answerable


def test_overlap_measures_shared_vocabulary(conn):
    from dyprys.evaluate import overlap_with
    passage = "The hippocampus is essential for forming new declarative memories."

    assert overlap_with("hippocampus memories", passage) == 1.0
    assert overlap_with("cerebellum balance walking", passage) == 0.0
    # "forms" must count against "forming", or easy questions look hard
    assert overlap_with("forms memories", passage) == 1.0
    assert overlap_with("which region forms new memories", passage) == 0.75


def test_overlap_ignores_words_every_passage_contains(conn):
    """Otherwise every question scores as easy."""
    from dyprys.evaluate import overlap_with
    assert overlap_with("what is the thing that does this", "zzz") == 0.0


def test_questions_are_banded_by_measured_overlap_not_by_label(conn, embedded, embedder):
    """The difficulty split must not depend on anyone's opinion of their own set."""
    from dyprys.search import Hit
    model, store = embedded
    target = chunk_containing(conn, model, "neurons")

    report = evaluate(conn, store, embedder, model, [
        Question("paragraph book neurons", "neurons", "lookup"),       # shares wording
        Question("gastronomy pottery archery", "neurons", "oblique"),  # shares none
    ], search_fn=lambda q, k: [Hit(target, 0.9)])

    assert set(report.by_overlap) == {"shares wording", "shares little"}
    assert report.by_overlap["shares wording"].answerable == 1
    assert report.by_overlap["shares little"].answerable == 1


def test_the_probe_is_a_phrase_that_actually_occurs(conn):
    """A probe built from filtered words is not a phrase, and cannot test one."""
    import random
    from dyprys.evaluate import _distinctive_phrase

    text = "The axon is a long thin fibre that conducts the impulse away " * 6
    phrase = _distinctive_phrase(text, random.Random(0))

    assert phrase
    assert phrase in " ".join(text.split())


def test_a_term_is_matched_as_a_word_not_as_a_substring():
    """"Saxon" does not contain "axon", and "produce" does not contain "rod".

    A plain substring test made 23% of book-hits on the real corpus spurious,
    and inflated the metric in the direction that matters: a passage scored as
    holding the answer while holding an unrelated word.
    """
    from dyprys.evaluate import contains_term

    assert not contains_term("the saxon kings of england", "axon")
    assert not contains_term("factories produce goods", "rod")
    assert not contains_term("the basal ganglia are deep", "glia")
    assert not contains_term("a famous biographer wrote it", "raphe")
    assert not contains_term("papyri from the tomb", "gyri")

    assert contains_term("the axon leaves the soma", "axon")
    assert contains_term("a rod responds to dim light", "rod")
    assert contains_term("radial glia guide migration", "glia")


def test_a_prefix_term_still_matches_its_longer_forms():
    """The question sets use prefixes on purpose, so the boundary is left-only."""
    from dyprys.evaluate import contains_term

    assert contains_term("electroencephalography shows waves", "electroencephalog")
    assert contains_term("a compensatory mechanism", "compensat")
    assert contains_term("the choroid plexus makes it", "choroid plexus")
