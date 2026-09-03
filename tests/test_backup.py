"""An index that can be moved, kept, and restored elsewhere."""

import tarfile

import numpy as np
import pytest

from dyprys import db
from dyprys.backup import read_manifest, restore, source_prefix, write
from dyprys.embed import store_for
from dyprys.ingest import ingest_paths
from dyprys.search import flat_search
from dyprys.text import read_span

DIM = 8


@pytest.fixture
def packed(conn, tmp_path, library):
    """A small but complete index: books, chunks, vectors, lexical rows."""
    model = db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "index", model, DIM)
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    # distinct directions: v[i % DIM] alone ties chunk 2 with chunk 10
    rng = np.random.default_rng(0)
    for chunk_id in range(1, total + 1):
        v = rng.normal(size=DIM).astype(np.float32)
        store.write(chunk_id, v / np.linalg.norm(v))
    store.flush()
    return model, store, tmp_path / "index"


def test_the_archive_carries_all_three_stores(conn, tmp_path, packed):
    _, _, directory = packed
    out = tmp_path / "backup.tar.gz"

    write(conn, directory, out)

    names = tarfile.open(out).getnames()
    assert "manifest.json" in names
    assert any(n.endswith("dyprys.sqlite") for n in names)
    assert any(n.endswith(".f32") for n in names)
    assert any(n.startswith("sources/") for n in names)


def test_the_manifest_records_what_it_is(conn, tmp_path, packed):
    _, _, directory = packed
    out = tmp_path / "backup.tar.gz"
    write(conn, directory, out)

    manifest = read_manifest(out)
    assert manifest["schema_version"] == db.SCHEMA_VERSION
    assert manifest["books"] == 2
    assert manifest["includes_sources"] is True
    assert manifest["source_prefix"]


def test_sources_can_be_left_out(conn, tmp_path, packed):
    _, _, directory = packed
    out = tmp_path / "thin.tar.gz"

    write(conn, directory, out, include_sources=False)

    assert not any(n.startswith("sources/") for n in tarfile.open(out).getnames())
    assert read_manifest(out)["includes_sources"] is False


# --- restoring ---------------------------------------------------------------


def restored(conn, tmp_path, packed, name="restored"):
    _, _, directory = packed
    out = tmp_path / "backup.tar.gz"
    write(conn, directory, out)
    return restore(out, tmp_path / name / "index", tmp_path / name / "books")


def test_a_restored_index_holds_the_same_books_and_chunks(conn, tmp_path, packed):
    before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    report = restored(conn, tmp_path, packed)

    assert report.books == 2
    assert report.chunks == before
    assert report.missing == []


def test_every_vector_survives_the_round_trip(conn, tmp_path, packed):
    """Losing these would mean re-embedding, which is the whole point of a backup."""
    model, store, _ = packed
    original = {i: store.read(i).copy() for i in range(1, 6)}

    report = restored(conn, tmp_path, packed)

    moved = db.connect(report.data_dir)
    try:
        model_id = moved.execute("SELECT id FROM models").fetchone()["id"]
        after = store_for(moved, report.data_dir, model_id, DIM)
        for chunk_id, vector in original.items():
            assert np.allclose(after.read(chunk_id), vector)
    finally:
        moved.close()


def test_passages_are_readable_at_the_new_location(conn, tmp_path, packed):
    """Offsets point into files; a restore that did not relocate would break them."""
    report = restored(conn, tmp_path, packed)

    moved = db.connect(report.data_dir)
    try:
        assert report.relocated > 0
        for row in moved.execute(
            "SELECT id, byte_offset, byte_length, content_hash FROM chunks LIMIT 20"
        ):
            located = db.locate(moved, row["id"])
            text = read_span(located["path"], row["byte_offset"], row["byte_length"],
                             row["content_hash"])
            assert text is not None      # the stored hash still verifies
    finally:
        moved.close()


def test_search_works_in_the_restored_copy(conn, tmp_path, packed):
    report = restored(conn, tmp_path, packed)

    moved = db.connect(report.data_dir)
    try:
        model_id = moved.execute("SELECT id FROM models").fetchone()["id"]
        store = store_for(moved, report.data_dir, model_id, DIM)
        hits = flat_search(moved, store, store.read(2).copy(), model_id, k=3)
        assert hits and hits[0].chunk_id == 2
    finally:
        moved.close()


def test_the_lexical_index_travels_too(conn, tmp_path, packed):
    from dyprys.lexical import indexed_count
    before = indexed_count(conn)
    report = restored(conn, tmp_path, packed)

    moved = db.connect(report.data_dir)
    try:
        assert indexed_count(moved) == before
    finally:
        moved.close()


# --- refusals ----------------------------------------------------------------


def test_it_will_not_restore_over_an_existing_index(conn, tmp_path, packed):
    """Merging into a live index would be unrecoverable."""
    _, _, directory = packed
    out = tmp_path / "backup.tar.gz"
    write(conn, directory, out)

    with pytest.raises(ValueError, match="already holds an index"):
        restore(out, directory, tmp_path / "books")


def test_a_file_that_is_not_a_backup_is_refused(tmp_path):
    junk = tmp_path / "junk.tar.gz"
    with tarfile.open(junk, "w:gz") as tar:
        tar.add(__file__, arcname="something.py")

    with pytest.raises(ValueError, match="no manifest"):
        read_manifest(junk)


def test_restoring_without_a_library_leaves_paths_alone(conn, tmp_path, packed):
    """An index-only restore is legitimate; it just cannot show passages."""
    _, _, directory = packed
    out = tmp_path / "backup.tar.gz"
    write(conn, directory, out, include_sources=False)

    report = restore(out, tmp_path / "thin" / "index", None)

    assert report.relocated == 0
    assert report.chunks > 0


def test_the_prefix_is_the_deepest_shared_directory(conn, library, tmp_path, packed):
    prefix = source_prefix(conn)
    assert prefix and str(library) == prefix


# --- more than one model over one library ------------------------------------


@pytest.fixture
def two_models(conn, tmp_path, library, packed):
    """A second model, different width and format, covering only part of it."""
    first, _, directory = packed
    second = db.model_id(conn, "narrow@feed", 4, "int8",
                         provenance={"file_name": "narrow.gguf", "file_bytes": 9,
                                     "file_sha256": "fe" * 32})
    store = store_for(conn, directory, second, 4)
    segments = conn.execute("SELECT id, chunk_start, chunk_count FROM segments "
                            "ORDER BY id").fetchall()
    covered = segments[len(segments) // 2:]
    with conn:
        for seg in covered:
            db.set_embedded_prefix(conn, second, seg["id"], seg["chunk_count"])
    rng = np.random.default_rng(3)
    for seg in covered:
        for cid in range(seg["chunk_start"], seg["chunk_start"] + seg["chunk_count"]):
            v = rng.normal(size=4).astype(np.float32)
            store.write(cid, v / np.linalg.norm(v))
    store.flush()
    return first, second, directory


def test_both_models_travel_with_their_own_width_and_format(conn, tmp_path, two_models):
    first, second, directory = two_models
    out = tmp_path / "two.tar.gz"
    write(conn, directory, out)

    weights = {w["name"]: w for w in read_manifest(out)["weights"]}
    assert len(weights) == 2
    assert weights["narrow@feed"]["dim"] == 4
    assert weights["narrow@feed"]["quantisation"] == "int8"
    assert weights["stub"]["dim"] == DIM


def test_each_model_keeps_exactly_its_own_coverage(conn, tmp_path, two_models):
    """A model covering half the library must not come back covering all of it."""
    from dyprys.search import embedded_ranges
    first, second, directory = two_models
    before = {
        m: sum(e - s for s, e in embedded_ranges(conn, m)) for m in (first, second)
    }
    out = tmp_path / "two.tar.gz"
    write(conn, directory, out)
    report = restore(out, tmp_path / "back" / "ix", tmp_path / "back" / "books")

    moved = db.connect(report.data_dir)
    try:
        for name, was in (("stub", before[first]), ("narrow@feed", before[second])):
            mid = moved.execute("SELECT id FROM models WHERE name = ?", (name,)).fetchone()["id"]
            assert sum(e - s for s, e in embedded_ranges(moved, mid)) == was
        assert before[first] != before[second]      # the halves really do differ
    finally:
        moved.close()


def test_a_narrow_models_vectors_survive_alongside_a_wide_one(conn, tmp_path, two_models):
    first, second, directory = two_models
    ranges = __import__("dyprys.search", fromlist=["x"]).embedded_ranges(conn, second)
    sample = [c for s, e in ranges for c in range(s, min(e, s + 5))]
    original = {c: store_for(conn, directory, second, 4).read(c).copy() for c in sample}

    out = tmp_path / "two.tar.gz"
    write(conn, directory, out)
    report = restore(out, tmp_path / "back" / "ix", tmp_path / "back" / "books")

    moved = db.connect(report.data_dir)
    try:
        mid = moved.execute("SELECT id FROM models WHERE name = 'narrow@feed'").fetchone()["id"]
        after = store_for(moved, report.data_dir, mid, 4)
        for chunk_id, vector in original.items():
            assert np.allclose(after.read(chunk_id), vector)
    finally:
        moved.close()
