"""Stage 1: profile each book, then search only the best few."""

import numpy as np
import pytest

from dyprys import db
from dyprys.embed import store_for
from dyprys.ingest import ingest_paths
from dyprys.vectors import VectorStore
from dyprys.routing import (
    build_centroids,
    is_built,
    route,
    scanned_fraction,
    spherical_kmeans,
    stale_books,
    two_stage_search,
)

DIM = 8


# --- the clustering itself ---------------------------------------------------


def unit(rows):
    a = np.asarray(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_centroids_come_back_unit_length(conn):
    """They are compared by dot product, so length would be a free score."""
    rng = np.random.default_rng(0)
    vectors = unit(rng.normal(size=(200, DIM)))

    centroids = spherical_kmeans(vectors, 4)

    assert centroids.shape == (4, DIM)
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-5)


def test_fewer_vectors_than_centroids_returns_them_all(conn):
    vectors = unit(np.eye(DIM)[:3])
    assert len(spherical_kmeans(vectors, 16)) == 3


def test_it_finds_separated_groups(conn):
    """Two tight clusters must produce two centroids, one in each."""
    rng = np.random.default_rng(0)
    a = unit(np.array([1.0] + [0.0] * (DIM - 1)) + rng.normal(scale=0.01, size=(60, DIM)))
    b = unit(np.array([0.0, 1.0] + [0.0] * (DIM - 2)) + rng.normal(scale=0.01, size=(60, DIM)))
    centroids = spherical_kmeans(np.vstack([a, b]), 2)

    near_a = max(float(c @ a[0]) for c in centroids)
    near_b = max(float(c @ b[0]) for c in centroids)
    assert near_a > 0.95 and near_b > 0.95


def test_no_centroid_is_left_at_zero(conn):
    """A zero centroid scores 0 against every query and wastes a slot silently."""
    vectors = unit(np.tile(np.eye(DIM)[0], (50, 1)) + 1e-6)  # all nearly identical
    centroids = spherical_kmeans(vectors, 8)
    assert np.all(np.linalg.norm(centroids, axis=1) > 0.5)


def test_it_is_deterministic(conn):
    rng = np.random.default_rng(1)
    vectors = unit(rng.normal(size=(150, DIM)))
    assert np.allclose(spherical_kmeans(vectors, 5), spherical_kmeans(vectors, 5))


# --- building and routing ----------------------------------------------------


@pytest.fixture
def routable(conn, tmp_path, library):
    """Two books whose vectors point in clearly different directions."""
    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "index", model, DIM)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])

    rng = np.random.default_rng(0)
    directions = {}
    for book in conn.execute("SELECT id FROM books ORDER BY id").fetchall():
        axis = np.zeros(DIM, dtype=np.float32)
        axis[book["id"] % DIM] = 1.0
        directions[book["id"]] = axis
        for start, end in conn.execute(
            "SELECT seg.chunk_start, seg.chunk_count FROM segments seg "
            "JOIN sources src ON src.id = seg.source_id WHERE src.book_id = ?",
            (book["id"],),
        ):
            for offset in range(end):
                noise = rng.normal(scale=0.02, size=DIM).astype(np.float32)
                vector = axis + noise
                store.write(start + offset, vector / np.linalg.norm(vector))
    store.flush()
    return model, store, directions


def test_building_profiles_every_embedded_book(conn, tmp_path, routable):
    model, store, _ = routable
    report = build_centroids(conn, tmp_path / "index", store, model, per_book=4)

    assert report.books == conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    assert report.centroids == report.books * 4
    assert is_built(conn, model)


def test_a_book_with_nothing_embedded_is_skipped_not_failed(conn, tmp_path, routable):
    model, store, _ = routable
    with conn:
        conn.execute("INSERT INTO books (key, title, added_at) VALUES ('/empty', 'empty', 'x')")

    report = build_centroids(conn, tmp_path / "index", store, model, per_book=4)
    assert report.skipped == 1


def test_rebuilding_replaces_rather_than_duplicates(conn, tmp_path, routable):
    model, store, _ = routable
    build_centroids(conn, tmp_path / "index", store, model, per_book=4)
    build_centroids(conn, tmp_path / "index", store, model, per_book=4)

    rows = conn.execute(
        "SELECT COUNT(*) FROM book_centroids WHERE model_id = ?", (model,)
    ).fetchone()[0]
    assert rows == conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]


def test_a_query_routes_to_the_book_it_belongs_to(conn, tmp_path, routable):
    model, store, directions = routable
    from dyprys.vectors import VectorStore
    build_centroids(conn, tmp_path / "index", store, model, per_book=4)
    centroids = VectorStore(db.centroids_path(tmp_path / "index", model), DIM, 999)

    for book_id, axis in directions.items():
        best = route(conn, centroids, axis, model, books=1)
        assert best[0][0] == book_id


def test_stage_two_only_touches_the_routed_books(conn, tmp_path, routable):
    model, store, directions = routable
    from dyprys.vectors import VectorStore
    build_centroids(conn, tmp_path / "index", store, model, per_book=4)
    centroids = VectorStore(db.centroids_path(tmp_path / "index", model), DIM, 999)
    wanted = next(iter(directions))

    hits, routed = two_stage_search(
        conn, store, centroids, directions[wanted], model, k=5, books=1
    )

    assert routed == {wanted}
    assert hits
    for hit in hits:
        assert db.locate(conn, hit.chunk_id)["book_id"] == wanted


def test_routing_reports_the_fraction_it_scanned(conn, tmp_path, routable):
    """The cost side of the claim, and the number flat search must be beaten on."""
    model, store, directions = routable
    from dyprys.vectors import VectorStore
    build_centroids(conn, tmp_path / "index", store, model, per_book=4)
    centroids = VectorStore(db.centroids_path(tmp_path / "index", model), DIM, 999)

    _, routed = two_stage_search(
        conn, store, centroids, next(iter(directions.values())), model, books=1
    )
    share = scanned_fraction(conn, model, routed)

    assert 0 < share < 1
    assert scanned_fraction(conn, model, None) == 1.0


def test_routing_more_widely_scans_more(conn, tmp_path, routable):
    model, store, directions = routable
    from dyprys.vectors import VectorStore
    build_centroids(conn, tmp_path / "index", store, model, per_book=4)
    centroids = VectorStore(db.centroids_path(tmp_path / "index", model), DIM, 999)
    query = next(iter(directions.values()))

    narrow = two_stage_search(conn, store, centroids, query, model, books=1)[1]
    wide = two_stage_search(conn, store, centroids, query, model, books=2)[1]

    assert scanned_fraction(conn, model, narrow) < scanned_fraction(conn, model, wide)


def test_embedding_more_makes_a_profile_stale(conn, tmp_path, routable):
    """A profile built from half a book is a profile of half a book."""
    model, store, _ = routable
    segment = conn.execute("SELECT id, chunk_count FROM segments ORDER BY id").fetchone()
    with conn:
        db.set_embedded_prefix(conn, model, segment["id"], 2)
    build_centroids(conn, tmp_path / "index", store, model, per_book=4)
    assert stale_books(conn, model) == 0

    with conn:
        db.set_embedded_prefix(conn, model, segment["id"], segment["chunk_count"])

    assert stale_books(conn, model) == 1


def test_routing_before_building_returns_nothing(conn, tmp_path, routable):
    model, store, directions = routable
    from dyprys.vectors import VectorStore
    centroids = VectorStore(tmp_path / "empty.f32", DIM, 8)

    assert not is_built(conn, model)
    assert route(conn, centroids, next(iter(directions.values())), model) == []


# --- k-means is run more than once -------------------------------------------


def test_restarts_never_return_a_worse_clustering(monkeypatch):
    """Restarts keep the best start by the objective routing itself uses.

    The objective is how well each vector is covered by its nearest direction,
    which is exactly what `route` scores a book on. Optimising anything else
    would tune for a quantity the search does not use.
    """
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(300, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    def coverage(centroids):
        return float((vectors @ centroids.T).max(axis=1).sum())

    # The starts are nested: restarts=8 tries everything restarts=1 tries and
    # more, so more starts can never come back with a worse clustering.
    for seed in range(4):
        fewer = coverage(spherical_kmeans(vectors, 8, seed=seed, restarts=1))
        more = coverage(spherical_kmeans(vectors, 8, seed=seed, restarts=8))
        assert more >= fewer - 1e-4, f"seed {seed}: 8 starts did worse than 1"

    # And it is a real choice, not a no-op: on this data some start wins.
    spread = [coverage(spherical_kmeans(vectors, 8, seed=s, restarts=1))
              for s in range(8)]
    assert max(spread) - min(spread) > 1e-3, "the starts were indistinguishable"


def test_one_start_is_still_available_and_deterministic():
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(200, 12)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    first = spherical_kmeans(vectors, 6, seed=1, restarts=1)
    again = spherical_kmeans(vectors, 6, seed=1, restarts=1)

    assert np.array_equal(first, again)


def test_fewer_vectors_than_clusters_still_short_circuits():
    """The restart loop must not run at all when every vector is its own centre."""
    vectors = np.eye(4, dtype=np.float32)
    assert np.array_equal(spherical_kmeans(vectors, 8), vectors)


# --- stage 1 is one matrix multiply -----------------------------------------


def _route_by_loop(conn, centroids, query, model_id, books, candidates=None):
    """The obvious implementation, kept as the definition of correct."""
    rows = conn.execute(
        "SELECT book_id, centroid_start, centroid_count FROM book_centroids "
        "WHERE model_id = ? ORDER BY centroid_start", (model_id,)).fetchall()
    scored = []
    for row in rows:
        if candidates is not None and row["book_id"] not in candidates:
            continue
        block = centroids.slice(row["centroid_start"], row["centroid_count"])
        scored.append((row["book_id"], float((block @ query).max())))
    scored.sort(key=lambda pair: -pair[1])
    return scored[:books]


@pytest.fixture
def profiled(conn, tmp_path, library, embedder):
    """Several books with centroids built, so routing has something to choose."""
    from dyprys.embed import embed_pending, store_for

    for n in range(2, 7):
        (library / f"book{n}.txt").write_text(
            "\n\n".join(f"Para {i} of book {n}, on topic {n}. " * 12 for i in range(12)),
            encoding="utf-8")
    ingest_paths(conn, [library])
    model = db.model_id(conn, embedder.name, embedder.dim)
    store = store_for(conn, tmp_path / "index", model, embedder.dim)
    embed_pending(conn, store, embedder, model)
    build_centroids(conn, tmp_path / "index", store, model, per_book=4)
    total = conn.execute("SELECT COALESCE(SUM(centroid_count),0) FROM book_centroids "
                         "WHERE model_id=?", (model,)).fetchone()[0]
    return model, VectorStore(db.centroids_path(tmp_path / "index", model),
                              embedder.dim, total)


def test_the_vectorised_route_matches_the_obvious_loop(conn, profiled, embedder):
    """A single matmul plus a segment max, and it must agree exactly.

    Written this way because stage 1 is the one part of a routed query that
    grows with the library: a BLAS call per book was 4.9 ms at 1,486 books
    against 1.21 ms for one call over all of them.
    """
    model, centroids = profiled
    everyone = {r["book_id"] for r in
                conn.execute("SELECT book_id FROM book_centroids WHERE model_id=?", (model,))}

    for text in ("neurons and synapses", "topic 3", "something unrelated entirely"):
        query = embedder.embed_query(text)
        for breadth in (1, 2, 3, 99):
            for candidates in (None, everyone, set(list(everyone)[:2])):
                expected = _route_by_loop(conn, centroids, query, model, breadth, candidates)
                actual = route(conn, centroids, query, model, breadth, candidates)
                assert [b for b, _ in actual] == [b for b, _ in expected]
                assert np.allclose([s for _, s in actual], [s for _, s in expected], atol=1e-5)


def test_routing_to_no_candidates_returns_nothing(conn, profiled, embedder):
    model, centroids = profiled
    assert route(conn, centroids, embedder.embed_query("x"), model, 5, candidates=set()) == []


def test_routing_never_returns_a_book_outside_the_candidates(conn, profiled, embedder):
    model, centroids = profiled
    everyone = sorted({r["book_id"] for r in
                       conn.execute("SELECT book_id FROM book_centroids WHERE model_id=?", (model,))})
    only = {everyone[0]}

    picked = route(conn, centroids, embedder.embed_query("neurons"), model, 5, candidates=only)

    assert [b for b, _ in picked] == [everyone[0]]
