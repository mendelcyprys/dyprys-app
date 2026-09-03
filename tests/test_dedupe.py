"""Near-duplicate suppression: the lexical signal, and what it must not remove."""

import pytest

from dyprys import db
from dyprys.embed import store_for
from dyprys.search import Hit, drop_near_duplicates, overlap

DIM = 8


def test_identical_text_overlaps_completely(conn):
    assert overlap("the ark rested upon the mountain", "the ark rested upon the mountain") == 1.0


def test_unrelated_text_does_not_overlap(conn):
    assert overlap("the ark rested upon the mountain", "synaptic vesicles release glutamate") == 0.0


def test_shared_topic_is_not_shared_wording(conn):
    """The distinction the whole feature turns on."""
    a = "the waters covered the earth and the ark came to rest"
    b = "rainfall that season flooded every valley in the region"
    assert overlap(a, b) < 0.1


@pytest.fixture
def indexed(conn, tmp_path, library):
    model = db.model_id(conn, "stub", DIM)
    store_for(conn, tmp_path / "index", model, DIM)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    return model


def test_different_passages_from_one_book_are_all_kept(conn, indexed):
    """Capping a file at one passage is the failure this must not reintroduce."""
    hits = [Hit(i, 1.0 / i) for i in range(1, 6)]
    assert len(drop_near_duplicates(conn, hits, k=5)) == 5


def test_an_exact_repeat_is_dropped(conn, indexed):
    """The same chunk twice is the clearest possible duplicate."""
    hits = [Hit(1, 0.9), Hit(1, 0.8), Hit(2, 0.7)]
    kept = drop_near_duplicates(conn, hits, k=5)
    assert [h.chunk_id for h in kept] == [1, 2]


def test_the_best_of_a_group_survives(conn, indexed):
    hits = [Hit(3, 0.9), Hit(3, 0.5)]
    assert drop_near_duplicates(conn, hits, k=5)[0].score == 0.9


def test_at_most_k_come_back(conn, indexed):
    hits = [Hit(i, 0.5) for i in range(1, 10)]
    assert len(drop_near_duplicates(conn, hits, k=3)) == 3


def test_a_high_threshold_suppresses_nothing(conn, indexed):
    hits = [Hit(1, 0.9), Hit(1, 0.8)]
    assert len(drop_near_duplicates(conn, hits, k=5, threshold=1.01)) == 2


def test_an_unreadable_passage_is_kept_not_dropped(conn, indexed, library):
    """A failed read is not evidence of duplication."""
    (library / "book0.txt").unlink()
    hits = [Hit(i, 0.5) for i in range(1, 6)]
    assert len(drop_near_duplicates(conn, hits, k=5)) == 5


def test_order_is_preserved(conn, indexed):
    hits = [Hit(5, 0.9), Hit(2, 0.8), Hit(7, 0.7)]
    kept = drop_near_duplicates(conn, hits, k=5)
    assert [h.chunk_id for h in kept] == [5, 2, 7]
