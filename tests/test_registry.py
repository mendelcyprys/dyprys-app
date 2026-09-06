"""One installation, many libraries."""

import json

import pytest

from dyprys import registry


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Never touch the real registry while testing."""
    monkeypatch.setenv("DYPRYS_HOME", str(tmp_path / "config"))
    return tmp_path


def test_nothing_is_registered_to_begin_with():
    assert registry.libraries() == []
    assert registry.resolve(None) is None


def test_the_first_library_becomes_the_default(isolated):
    entry = registry.add("neuro", isolated / "a")
    assert entry.is_default
    assert registry.resolve(None) == (isolated / "a").resolve()


def test_later_libraries_do_not_steal_the_default(isolated):
    registry.add("neuro", isolated / "a")
    registry.add("bible", isolated / "b")

    assert [l.name for l in registry.libraries()] == ["bible", "neuro"]
    assert registry.resolve(None) == (isolated / "a").resolve()


def test_the_default_can_be_moved(isolated):
    registry.add("neuro", isolated / "a")
    registry.add("bible", isolated / "b")

    assert registry.use("bible")
    assert registry.resolve(None) == (isolated / "b").resolve()


def test_using_an_unknown_name_fails_rather_than_guessing():
    assert registry.use("nope") is False


def test_a_name_resolves_to_its_path(isolated):
    registry.add("neuro", isolated / "a")
    assert registry.resolve("neuro") == (isolated / "a").resolve()
    assert registry.resolve("other") is None


def test_forgetting_a_name_leaves_the_files(isolated):
    (isolated / "a").mkdir()
    (isolated / "a" / "dyprys.sqlite").write_text("x")
    registry.add("neuro", isolated / "a")

    assert registry.remove("neuro")
    assert registry.libraries() == []
    assert (isolated / "a" / "dyprys.sqlite").exists()


def test_forgetting_the_default_promotes_another(isolated):
    registry.add("neuro", isolated / "a")
    registry.add("bible", isolated / "b")
    registry.use("neuro")

    registry.remove("neuro")

    assert registry.resolve(None) == (isolated / "b").resolve()


def test_forgetting_the_last_one_leaves_no_default(isolated):
    registry.add("only", isolated / "a")
    registry.remove("only")
    assert registry.resolve(None) is None


def test_a_library_knows_whether_an_index_is_there(isolated):
    registry.add("empty", isolated / "a")
    assert registry.libraries()[0].exists is False

    (isolated / "a").mkdir(parents=True, exist_ok=True)
    (isolated / "a" / "dyprys.sqlite").write_text("x")
    assert registry.libraries()[0].exists is True


def test_a_corrupt_registry_does_not_hide_every_library(isolated):
    """Losing the registry must lose names only — --data still opens anything."""
    registry.add("neuro", isolated / "a")
    registry.registry_path().write_text("{ not json")

    assert registry.libraries() == []
    assert registry.resolve("neuro") is None


def test_a_corrupt_registry_is_kept_rather_than_overwritten(isolated):
    """The write half of the same story, which used to lose everything.

    `write_text` truncates before it writes, so a crash or a full disk leaves a
    half-written file; `_load` reads that as an empty registry; and the next
    `add` writes over it. Three registered libraries became one, unrecoverably.
    """
    for name in ("neuro", "bible", "shared"):
        registry.add(name, isolated / name)
    path = registry.registry_path()
    intact = path.read_text()
    path.write_text(intact[: len(intact) // 2])          # a torn write

    assert not registry.readable()
    registry.add("newone", isolated / "d")

    kept = path.with_name(path.name + registry.UNREADABLE_SUFFIX)
    assert kept.exists(), "the unreadable registry was overwritten"
    # Not merely kept: the names are still in there to be recovered by hand.
    assert '"neuro"' in kept.read_text()
    assert [l.name for l in registry.libraries()] == ["newone"]


def test_each_rescue_is_kept_separately(isolated):
    """Preserving twice must not undo the first rescue."""
    registry.add("first", isolated / "a")
    path = registry.registry_path()
    for round_ in (1, 2, 3):
        path.write_text(f"{{ torn number {round_}")
        registry.add(f"lib{round_}", isolated / str(round_))

    kept = sorted(p.name for p in path.parent.iterdir() if registry.UNREADABLE_SUFFIX in p.name)
    assert len(kept) == 3
    saved = {path.parent.joinpath(name).read_text() for name in kept}
    assert saved == {"{ torn number 1", "{ torn number 2", "{ torn number 3"}


@pytest.mark.parametrize("junk", ["{}", "[]", "null", '"hello"', "42"])
def test_valid_json_of_the_wrong_shape_is_not_a_registry(isolated, junk):
    """`{}` used to raise KeyError out of whichever command touched it first."""
    registry.add("neuro", isolated / "a")
    registry.registry_path().write_text(junk)

    assert not registry.readable()
    assert registry.libraries() == []
    assert registry.resolve(None) is None
    assert registry.resolve("neuro") is None


def test_the_registry_is_replaced_by_rename_never_truncated(isolated):
    """The swap is a rename, so a reader sees one whole registry or the other.

    `write_text` opens the destination for writing, which truncates it first --
    that is the window a crash or a full disk turns into the torn file above.
    Writing a neighbour and renaming has no such window, and the inode changing
    is what shows the destination was never opened.
    """
    registry.add("neuro", isolated / "a")
    path = registry.registry_path()
    before = path.stat().st_ino

    registry.add("bible", isolated / "b")

    assert path.stat().st_ino != before, "written in place; a torn write is possible"
    assert not list(path.parent.glob("*.tmp")), "left a temp file behind"


def test_a_failed_write_leaves_no_debris(isolated, monkeypatch):
    """Whatever goes wrong, the old registry stands and no temp file survives."""
    registry.add("neuro", isolated / "a")
    registry.add("bible", isolated / "b")
    before = registry.registry_path().read_text()

    def full(*_a, **_k):
        raise OSError("no space left on device")

    monkeypatch.setattr(registry.json, "dumps", full)
    with pytest.raises(OSError):
        registry.add("newone", isolated / "c")

    assert registry.registry_path().read_text() == before
    assert [l.name for l in registry.libraries()] == ["bible", "neuro"]
    assert not list(registry.registry_path().parent.glob("*.tmp"))


def test_re_adding_a_name_repoints_it(isolated):
    registry.add("neuro", isolated / "a")
    registry.add("neuro", isolated / "b")

    assert registry.resolve("neuro") == (isolated / "b").resolve()
    assert len(registry.libraries()) == 1


def test_paths_are_stored_absolute(isolated, monkeypatch):
    """A relative path would mean something different from another directory."""
    monkeypatch.chdir(isolated)
    registry.add("here", "./somewhere")

    stored = json.loads(registry.registry_path().read_text())["libraries"]["here"]
    assert stored.startswith("/")


def test_a_library_pointing_nowhere_is_an_error_not_an_empty_index(isolated, tmp_path):
    """A stale entry or an unmounted drive must not become a fresh empty index."""
    import subprocess, sys as _sys
    registry.add("gone", isolated / "missing")

    result = subprocess.run(
        [_sys.executable, "-m", "dyprys.cli", "-L", "gone", "status"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "DYPRYS_HOME": str(registry.config_home())},
    )

    assert result.returncode == 2
    assert "no index at" in result.stderr
    assert not (isolated / "missing" / "dyprys.sqlite").exists()


def test_a_library_is_one_directory_and_nothing_else(isolated):
    """Which is why dropping one is a directory removal, not a migration."""
    from dyprys import db
    index = isolated / "lib"
    conn = db.connect(index)
    conn.close()

    outside = [p for p in (isolated).rglob("*") if p.is_file() and index not in p.parents]
    assert (index / "dyprys.sqlite").exists()
    # nothing was written anywhere but the index directory and the registry
    assert all(registry.config_home() in p.parents or index in p.parents for p in outside)


def test_contents_reports_what_dropping_would_remove(isolated):
    from dyprys import db
    index = isolated / "lib"
    db.connect(index).close()

    files, size = registry.contents(index)
    assert files >= 1 and size > 0


# --- summarising a library without touching it -------------------------------


def _index(tmp_path, name, books=2):
    """A small real index on disk."""
    from dyprys import db as _db
    from dyprys.ingest import ingest_paths

    lib = tmp_path / f"{name}_texts"
    lib.mkdir()
    for n in range(books):
        (lib / f"b{n}.txt").write_text(
            "\n\n".join(f"Para {i} of book {n}. " * 12 for i in range(10)), encoding="utf-8")
    conn = _db.connect(tmp_path / name)
    ingest_paths(conn, [lib])
    model = _db.model_id(conn, "stub", 8)
    with conn:
        seg = conn.execute("SELECT id, chunk_count FROM segments ORDER BY id").fetchone()
        _db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    conn.commit()
    conn.close()
    return tmp_path / name


def test_summarise_reports_books_chunks_and_coverage(isolated, tmp_path):
    path = _index(tmp_path, "lib")

    held = registry.summarise(path)

    assert held.books == 2
    assert held.chunks > 0
    assert [name for name, _done, _whole in held.models] == ["stub"]
    assert 0 < held.coverage < 1


def test_summarise_does_not_write_to_the_library(isolated, tmp_path):
    """Listing what exists must not migrate five indexes as a side effect.

    db.connect runs the schema script and any pending migration, which is right
    when you mean to use an index and wrong when you are only naming it.
    """
    path = _index(tmp_path, "lib")
    database = path / "dyprys.sqlite"
    before = (database.stat().st_mtime_ns, database.stat().st_size)

    registry.summarise(path)

    assert (database.stat().st_mtime_ns, database.stat().st_size) == before


def test_summarise_survives_a_directory_with_no_index(isolated, tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    assert registry.summarise(empty) is None


def test_a_renamed_index_is_listed_rather_than_reported_missing(isolated, tmp_path):
    """The listing must resolve the database the way opening it does.

    `db_path` finds an index by the tables in it, so a renamed or forked
    database opens and searches normally — but the registry still asked for
    `dyprys.sqlite` by name, so `dyp library list` printed "no index yet" and
    dashes beside a 3,453-book library that every other command read fine.
    """
    path = _index(tmp_path, "lib")
    (path / "dyprys.sqlite").rename(path / "cyprys.sqlite")

    entry = registry.Library(name="forked", path=path)
    assert entry.exists, "an index this build can open must not read as absent"

    held = registry.summarise(path)
    assert held is not None and held.books == 2


def test_two_indexes_in_one_directory_do_not_break_the_listing(isolated, tmp_path):
    """`db_path` refuses to guess between them; a listing may say nothing about
    that library, but must still name the others rather than raising."""
    path = _index(tmp_path, "lib")
    import shutil
    shutil.copy(path / "dyprys.sqlite", path / "other.sqlite")
    (path / "dyprys.sqlite").rename(path / "cyprys.sqlite")

    assert registry.summarise(path) is None


def test_summarise_survives_a_file_that_is_not_a_database(isolated, tmp_path):
    """A foreign or half-written file must not take down `dyp library list`."""
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "dyprys.sqlite").write_bytes(b"this is not a database" * 40)

    assert registry.summarise(fake) is None


def test_coverage_is_measured_against_a_model_s_own_chunking(tmp_path):
    """A model embeds one chunking, so the library total is the wrong divisor.

    The same per-model-numerator-over-library-denominator mistake that made
    `dyp watch` report over 100%. Here it reads the other way: a model that has
    finished its own chunking looks permanently short of complete because a
    second chunking it will never touch is in the denominator.
    """
    from dyprys import db as _db
    from dyprys.ingest import ingest_paths
    from dyprys.registry import summarise

    book = tmp_path / "book.txt"
    book.write_text("\n\n".join(
        f"Paragraph {n} about neurons and synapses. " * 8 for n in range(50)),
        encoding="utf-8")
    conn = _db.connect(tmp_path / "ix")
    ingest_paths(conn, [book], target=3600)
    ingest_paths(conn, [book], target=900)
    ways = _db.chunkings(conn)
    assert len(ways) == 2

    coarse = next(c for c in ways if c["target"] == 3600)
    model = _db.model_id(conn, "stub@aaaaaaaaaaaa", 8)
    _db.bind_chunking(conn, model, coarse["id"])
    with conn:
        for seg in conn.execute(
                "SELECT id, chunk_count FROM segments WHERE chunking_id = ?",
                (coarse["id"],)).fetchall():
            _db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    conn.close()

    held = summarise(tmp_path / "ix")
    name, done, whole = held.models[0]
    assert done == whole, "a finished model should read as finished"
    assert whole < held.chunks, "the fixture should have a second chunking"
    assert held.coverage == 1.0
