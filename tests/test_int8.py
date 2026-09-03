"""int8 vector storage: a quarter the disk, and the same answers."""

import numpy as np
import pytest

from dyprys import db
from dyprys.embed import store_for
from dyprys.library import quantise_model
from dyprys.vectors import FP32, INT8, VectorStore, bytes_per_row

DIM = 64


def unit(rng, n=32):
    v = rng.normal(size=(n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def vectors():
    return unit(np.random.default_rng(0))


# --- the format --------------------------------------------------------------


def test_int8_is_a_quarter_the_width(vectors):
    assert bytes_per_row(FP32, DIM) == DIM * 4
    # one scale plus one byte per dimension
    assert bytes_per_row(INT8, DIM) == DIM + 4
    assert bytes_per_row(FP32, DIM) / bytes_per_row(INT8, DIM) > 3.7


def test_a_round_trip_barely_moves_the_vector(tmp_path, vectors):
    store = VectorStore(tmp_path / "v", DIM, len(vectors), INT8)
    store.write_many(1, vectors)
    back = store.slice(1, len(vectors))
    back /= np.linalg.norm(back, axis=1, keepdims=True)

    assert np.einsum("ij,ij->i", vectors, back).min() > 0.999


def test_fp32_is_exact(tmp_path, vectors):
    store = VectorStore(tmp_path / "v", DIM, len(vectors), FP32)
    store.write_many(1, vectors)
    assert np.allclose(store.slice(1, len(vectors)), vectors)


def test_a_zero_vector_stays_zero_rather_than_becoming_nan(tmp_path):
    """A failed chunk is written as zeros; dividing by its scale would poison it."""
    store = VectorStore(tmp_path / "v", DIM, 2, INT8)
    store.write(1, np.zeros(DIM, dtype=np.float32))
    row = store.read(1)
    assert np.all(np.isfinite(row))
    assert float(row @ np.ones(DIM, dtype=np.float32)) == 0.0


@pytest.mark.parametrize("quantisation", [FP32, INT8])
def test_score_agrees_with_slice(tmp_path, vectors, quantisation):
    """score() exists to bound memory; it must not change the arithmetic."""
    store = VectorStore(tmp_path / "v", DIM, len(vectors), quantisation)
    store.write_many(1, vectors)
    query = vectors[3]

    assert np.allclose(
        store.score(1, len(vectors), query), store.slice(1, len(vectors)) @ query, atol=1e-5
    )


def test_score_is_blockwise_and_still_correct(tmp_path, vectors):
    """Decoding a whole flat scan would allocate 42 GB at 5,000 books."""
    store = VectorStore(tmp_path / "v", DIM, len(vectors), INT8)
    store.write_many(1, vectors)
    query = vectors[0]

    whole = store.score(1, len(vectors), query)
    tiny = store.score(1, len(vectors), query, block=3)
    assert np.allclose(whole, tiny)


# --- moving rows -------------------------------------------------------------


def test_moving_rows_does_not_re_encode(tmp_path, vectors):
    """Compaction runs repeatedly; a decode/encode cycle would drift each time."""
    store = VectorStore(tmp_path / "v", DIM, 64, INT8)
    store.write_many(1, vectors)
    original = store.read(5).copy()

    for _ in range(20):
        store.move_rows(5, 2, 1)
        store.move_rows(2, 5, 1)

    assert np.array_equal(store.read(5), original)


def test_truncate_drops_the_tail(tmp_path, vectors):
    store = VectorStore(tmp_path / "v", DIM, len(vectors), INT8)
    store.write_many(1, vectors)
    before = store.path.stat().st_size

    store.truncate(4)

    assert store.path.stat().st_size < before
    assert store.path.stat().st_size == 4 * bytes_per_row(INT8, DIM)


# --- wiring ------------------------------------------------------------------


def test_the_format_comes_from_the_database_not_the_file(conn, tmp_path, library):
    """An int8 file opened as fp32 reads fine and returns nonsense."""
    model = db.model_id(conn, "quantised", DIM, "int8")
    assert db.model_quantisation(conn, model) == "int8"
    assert store_for(conn, tmp_path / "ix", model, DIM).quantisation == "int8"


def test_a_model_defaults_to_fp32(conn):
    model = db.model_id(conn, "plain", DIM)
    assert db.model_quantisation(conn, model) == "fp32"


def test_converting_shrinks_the_file_and_records_it(conn, tmp_path, library, vectors):
    model = db.model_id(conn, "convert-me", DIM)
    store = store_for(conn, tmp_path / "ix", model, DIM)
    store.write_many(1, vectors[: min(len(vectors), store.rows)])
    store.flush()
    store.close()

    wide = store.path.stat().st_size
    before, after = quantise_model(conn, tmp_path / "ix", model)
    narrow = store.path.stat().st_size

    assert db.model_quantisation(conn, model) == "int8"
    # The reported figures are blocks actually allocated, which is the honest
    # number for a sparse file but rounds to one block at test scale. The width
    # of the file is what the format guarantees.
    assert narrow < wide
    assert wide / narrow > 3.7
    assert after <= before
    assert store_for(conn, tmp_path / "ix", model, DIM).quantisation == "int8"


def test_converting_twice_is_refused(conn, tmp_path, library, vectors):
    model = db.model_id(conn, "convert-me", DIM)
    store = store_for(conn, tmp_path / "ix", model, DIM)
    store.write_many(1, vectors[: min(len(vectors), store.rows)])
    store.flush(); store.close()
    quantise_model(conn, tmp_path / "ix", model)

    with pytest.raises(ValueError, match="already quantised"):
        quantise_model(conn, tmp_path / "ix", model)


def test_conversion_preserves_what_the_vectors_meant(conn, tmp_path, library):
    """The whole claim: a quarter the disk, the same answers."""
    rng = np.random.default_rng(1)
    model = db.model_id(conn, "fidelity", DIM)
    store = store_for(conn, tmp_path / "ix", model, DIM)
    original = unit(rng, store.rows)
    store.write_many(1, original)
    store.flush(); store.close()

    quantise_model(conn, tmp_path / "ix", model)

    after = store_for(conn, tmp_path / "ix", model, DIM)
    query = original[0]
    before_scores = original @ query
    after_scores = after.score(1, len(original), query)
    # ranking, which is all retrieval depends on, is unchanged
    assert np.array_equal(np.argsort(-before_scores)[:5], np.argsort(-after_scores)[:5])


def test_scoring_holds_a_bounded_block_whatever_the_range(tmp_path):
    """The bound is bytes, not rows: a row count is silently dimension-dependent.

    The old default of 65,536 rows is 201 MB at dim 768 — and larger than a
    53,567-chunk corpus, so a flat scan decoded the whole thing at once, which
    is the one thing blocking exists to prevent.
    """
    import tracemalloc

    from dyprys.vectors import SCORE_BLOCK_BYTES, VectorStore, _score_block

    dim, rows = 768, 60_000
    store = VectorStore(tmp_path / "v.f32", dim=dim, rows=rows, quantisation="int8")
    store.write_many(1, np.random.rand(rows, dim).astype(np.float32))
    query = np.random.rand(dim).astype(np.float32)

    store.score(1, rows, query)                       # warm, then measure
    tracemalloc.start()
    store.score(1, rows, query)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # A few blocks' worth of slack for temporaries, and an order of magnitude
    # below what decoding the whole range would have cost.
    whole_range = rows * dim * 4
    assert peak < 6 * SCORE_BLOCK_BYTES
    assert peak < whole_range / 5
    assert _score_block(dim) < rows


def test_the_block_size_does_not_change_the_scores(tmp_path):
    from dyprys.vectors import VectorStore

    dim, rows = 128, 5_000
    store = VectorStore(tmp_path / "v.f32", dim=dim, rows=rows, quantisation="int8")
    store.write_many(1, np.random.rand(rows, dim).astype(np.float32))
    query = np.random.rand(dim).astype(np.float32)

    reference = store.score(1, rows, query, block=rows)
    for block in (1, 7, 256, 4096):
        assert np.allclose(store.score(1, rows, query, block=block), reference)
