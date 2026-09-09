"""Libraries hold shelves hold books, and a book can leave without being lost.

Two features that only make sense together. A shelf is the level a person
actually reasons about -- `-c papers/` has always meant one -- and setting one
aside is what you want the moment you can see it: a shelf of 1,657 Gutenberg
texts that swamps every answer, or three books whose extraction lost every word
boundary. The alternative was `dyp remove`, which discards the embedding.
"""

from __future__ import annotations

import pytest

from dyprys import db, errors, service
from dyprys.library import books as inspect, shelves, text_root
from dyprys.search import embedded_ranges, flat_search, set_aside
from tests.indexes import SENTINEL, embedded_index, write_books


@pytest.fixture
def index(tmp_path):
    """Two books on one shelf, embedded."""
    books = write_books(tmp_path / "lib")
    conn, embedder, model_id, store = embedded_index(tmp_path / "ix", books)
    yield conn, embedder, model_id, store
    conn.close()


@pytest.fixture
def shelved(tmp_path):
    """Three shelves under one root, one of them nested, so the middle level is
    a real partition rather than a single directory renamed."""
    write_books(tmp_path / "texts" / "papers", count=2)
    write_books(tmp_path / "texts" / "papers" / "old", count=1)
    write_books(tmp_path / "texts" / "notes", count=1)
    conn, embedder, model_id, store = embedded_index(
        tmp_path / "ix", sorted((tmp_path / "texts").rglob("*.txt")))
    yield conn, tmp_path
    conn.close()


def aside(conn, title: str, *, on: bool = True):
    chosen = [b for b in inspect(conn) if b.title == title]
    assert chosen, f"no book called {title}"
    return service.set_books_aside(conn, chosen, on)


# --- the shelf, derived rather than stored ----------------------------------


def test_shelves_partition_the_books_rather_than_nesting(shelved):
    """A book is on exactly one shelf.

    `papers/old` is not counted inside `papers`. Nesting would read more
    naturally and would make "set this shelf aside" mean two different things
    depending on which row was clicked, which is the one thing this level must
    not do.
    """
    conn, _ = shelved
    found = {sh.path: sh.books for sh in shelves(conn)}

    assert found == {"notes": 1, "papers": 2, "papers/old": 1}
    assert sum(found.values()) == len(inspect(conn))


def test_the_root_comes_from_the_books_not_from_the_index_directory(shelved):
    """The index and the text are not the same place, on two real libraries here.

    `neuro` indexes at `neuro/index` and reads from `neuro/cyprys_scale_texts`,
    so a shelf computed against the registered library path would be an absolute
    path with no shared prefix at all.
    """
    conn, tmp_path = shelved

    assert text_root(conn) == str(tmp_path / "texts")
    # And the index lives somewhere else entirely.
    assert not str(tmp_path / "ix").startswith(text_root(conn))


def test_filtering_the_list_does_not_move_the_split_point(shelved):
    """`dyp books PATTERN` must report the same shelf `dyp books` does.

    The root is the deepest directory *every* book shares, so deriving it from a
    filtered list would rename every shelf on screen as soon as someone typed in
    the search box.
    """
    conn, _ = shelved
    # Keyed by `key`, not by title: `write_books` names one book0 per directory,
    # which is the same collision `sefaria` has with two books called Arakhin.
    whole = {b.key: b.shelf for b in inspect(conn)}
    one = inspect(conn, "book0")

    assert len(one) > 1, "the fixture no longer has a colliding title"
    for book in one:
        assert book.shelf == whole[book.key]


def test_a_shelf_is_addressed_exactly_and_not_by_glob(shelved):
    """Nothing destructive here takes a pattern.

    `-c` is fuzzy because a search that ranks the wrong book costs a second
    look. Setting a shelf aside does not, so `books_named` takes a whole
    directory or whole keys -- and a caller that means "everything matching
    *Atlas*" resolves that against the listing first, which also means the list
    it acted on is the list it showed.
    """
    conn, tmp_path = shelved
    on_it = service.books_named(conn, shelf=str(tmp_path / "texts" / "papers" / "old"))

    assert len(on_it) == 1
    with pytest.raises(errors.NoSuchBook):
        service.books_named(conn, shelf="papers")  # relative: not a directory
    with pytest.raises(errors.BadRequest):
        service.books_named(conn, keys=["a"], shelf="b")


# --- set aside: out of every search, and nothing lost -----------------------


def test_a_set_aside_book_is_returned_by_no_search(index):
    conn, embedder, model_id, store = index
    query = embedder.embed_query("neurons and synapses")
    before = flat_search(conn, store, query, model_id, k=20)
    assert before

    aside(conn, "book0")

    after = flat_search(conn, store, query, model_id, k=20)
    gone = {h.chunk_id for h in before} - {h.chunk_id for h in after}
    assert gone, "setting a book aside changed nothing"
    assert all(db.locate(conn, h.chunk_id)["title"] != "book0" for h in after)


def test_an_exact_phrase_cannot_reach_a_set_aside_book_either(index):
    """The leg that deliberately escapes the scope must not escape this.

    BM25's phrase attempt searches the library whole even under `-c`, on purpose
    -- confining it took lexical safety from 20/20 to 9/20. But `-c` is a
    statement about what was *ranked*, and setting a book aside is a statement
    about what exists to rank, so the escape hatch must not apply. This is the
    assertion that would fail if the filter were copied into `flat_search`
    instead of living in `embedded_ranges`.
    """
    from dyprys.lexical import search_bm25

    conn, _, model_id, _ = index
    book0 = {b.id for b in inspect(conn) if b.title == "book0"}
    book1 = {b.id for b in inspect(conn) if b.title == "book1"}

    def phrase_hits(where):
        origin: dict[int, str] = {}
        found = search_bm25(conn, SENTINEL, 5, model_id=model_id,
                            book_ids=where, origin=origin)
        return [h for h in found if origin.get(h.chunk_id) == "phrase"]

    def in_book0(hits):
        return [h for h in hits
                if db.locate(conn, h.chunk_id)["book_id"] in book0]

    # Scoped to book1, the phrase still reaches book0 -- that is the escape,
    # deliberate and measured, and it has to be here for the next line to mean
    # anything.
    assert in_book0(phrase_hits(book1)), "the deliberate scope escape is gone"

    aside(conn, "book0")

    assert not in_book0(phrase_hits(book1))
    assert not in_book0(phrase_hits(None))


def test_putting_a_book_back_returns_exactly_what_was_there(index):
    """The whole claim: this costs nothing and is undone by one call."""
    conn, embedder, model_id, store = index
    query = embedder.embed_query("neurons and synapses")
    before = flat_search(conn, store, query, model_id, k=20)

    aside(conn, "book0")
    aside(conn, "book0", on=False)

    assert flat_search(conn, store, query, model_id, k=20) == before


def test_nothing_is_deleted_and_the_accounting_says_so(index):
    """`everything=True` is what a backup, a compaction and a coverage bar read.

    Those numbers describe what is on disk. If they moved when someone hid a
    shelf, `dyp check` would report work outstanding that is already done.
    """
    conn, _, model_id, _ = index
    on_disk = sum(e - s for s, e in embedded_ranges(conn, model_id, everything=True))
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    aside(conn, "book0")

    assert sum(e - s for s, e in embedded_ranges(conn, model_id, everything=True)) == on_disk
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == chunks
    assert sum(e - s for s, e in embedded_ranges(conn, model_id)) < on_disk


def test_setting_aside_twice_changes_nothing_the_second_time(index):
    conn, *_ = index
    assert aside(conn, "book0")["changed"] == 1
    assert aside(conn, "book0")["changed"] == 0
    assert len(set_aside(conn)) == 1


# --- what a search says about it --------------------------------------------


def test_a_scope_that_is_entirely_set_aside_is_refused_not_answered(index, tmp_path):
    """Returning a full `k` from elsewhere is the opposite of what `-c` promises."""
    conn, embedder, model_id, store = index
    aside(conn, "book0")
    session = service.Session(tmp_path / "ix", conn,
                              load_model=lambda *a, **k: (embedder, model_id, store))

    with pytest.raises(errors.NoSuchBook) as refused:
        service.search(session, "neurons", service.SearchOptions(collection="book0"))

    assert "set aside" in refused.value.message


def test_a_scope_that_is_partly_set_aside_says_which_part(index, tmp_path):
    """The same class of silence as an unprofiled book, and the same treatment.

    The search is still worth running -- some of the scope is live -- but a
    result list that quietly dropped half the books someone named is a promise
    broken without a word.
    """
    conn, embedder, model_id, store = index
    aside(conn, "book0")
    session = service.Session(tmp_path / "ix", conn,
                              load_model=lambda *a, **k: (embedder, model_id, store))

    result = service.search(
        session, "neurons", service.SearchOptions(collection=["book0", "book1"]))

    kinds = [w.kind for w in result.warnings]
    assert "books_set_aside" in kinds, kinds
    assert result.passages


# --- removing for good is the other half ------------------------------------


def test_removing_without_confirming_removes_nothing(index):
    """The preview and the act are one function, so they cannot disagree."""
    conn, *_ = index
    chosen = inspect(conn)
    was = len(chosen)

    plan = service.drop_books(conn, chosen[:1])

    assert plan["removed"] is False
    assert plan["count"] == 1 and plan["chunks"] > 0
    assert len(inspect(conn)) == was


def test_confirming_removes_exactly_what_the_preview_named(index):
    conn, *_ = index
    chosen = inspect(conn)[:1]
    plan = service.drop_books(conn, chosen)

    done = service.drop_books(conn, chosen, confirm=True)

    assert done["removed"] is True
    assert (done["count"], done["chunks"]) == (plan["count"], plan["chunks"])
    assert [b.title for b in inspect(conn)] == ["book1"]


# --- the rest of the pipeline agrees ----------------------------------------


def test_routing_does_not_spend_a_slot_on_a_set_aside_book(index, tmp_path):
    """Correct is not enough here; it has to not be wasteful.

    Stage 2 would read nothing from a set-aside book, so a route that picked one
    would be right and would still have searched one book instead of two. On a
    five-book route with a shelf set aside that is most of the point of routing
    gone, silently.
    """
    from dyprys.routing import build_centroids, route
    from dyprys.vectors import VectorStore

    conn, embedder, model_id, store = index
    directory = tmp_path / "ix"
    build_centroids(conn, directory, store, model_id)
    conn.commit()
    centroids = VectorStore(
        db.centroids_path(directory, model_id), embedder.dim,
        conn.execute("SELECT SUM(centroid_count) FROM book_centroids").fetchone()[0])
    query = embedder.embed_query("neurons and synapses")
    everything = {b for b, _ in route(conn, centroids, query, model_id, books=5)}
    assert len(everything) == 2, "the fixture no longer has two books to choose between"

    aside(conn, "book0")
    hidden = {b.id for b in inspect(conn) if b.title == "book0"}

    session = service.Session(directory, conn,
                              load_model=lambda *a, **k: (embedder, model_id, store))
    router, _ = service._router(conn, directory, embedder, model_id, 5, None)
    assert router is not None
    assert not (router("neurons", query) & hidden)


def test_deleting_a_library_previews_before_it_destroys(tmp_path, monkeypatch):
    """The one irreversible act in the tool, and the reason it now has a route.

    `library_deletion` is what both a terminal and a browser build their
    confirmation from, so the sentence "the vectors go, the text stays" is one
    statement rather than two that can drift. Unconfirmed it must delete
    nothing -- otherwise the preview is the act.
    """
    from dyprys import registry

    books = write_books(tmp_path / "lib")
    conn, *_ = embedded_index(tmp_path / "ix", books)
    conn.close()
    entry = type("L", (), {"name": "scratch", "path": tmp_path / "ix", "exists": True})()
    monkeypatch.setattr(registry, "libraries", lambda: [entry])
    monkeypatch.setattr(registry, "remove", lambda name: True)

    plan = service.delete_library("scratch")

    assert plan["deleted"] is False
    assert plan["files"] > 0 and plan["bytes"] > 0
    # Where the text is, so "the text is not touched" is checkable rather than
    # a promise. It is a different directory from the index, which is the fact
    # that makes the promise true.
    assert plan["sources"] == str(tmp_path / "lib")
    assert plan["shares_directory"] is False
    assert plan["kept_files"] == 0, "a dedicated index directory leaves nothing behind"
    assert (tmp_path / "ix").exists()


def test_deleting_a_library_never_takes_the_books_with_it(tmp_path, monkeypatch):
    """The layout on which "the text is somewhere else" is simply false.

    `dyp library add x ~/texts` followed by `dyp add ~/texts` puts the index in
    the directory holding the books. It is an ordinary thing to do -- two of
    the three libraries this was first run against were built that way -- and
    deleting the index was `shutil.rmtree` on the registered path, which took
    every book with it. Both frontends meanwhile printed the opposite: the
    terminal said "the text itself is elsewhere and is NOT touched, e.g. <that
    same directory>", and the browser drew the one path under **Deleted** and
    again under **Kept**.

    So this asserts the guarantee rather than the wording. A deletion removes
    what the index wrote and nothing else, on any layout, and the count it
    offers for confirmation is a count of that and not of the directory.
    """
    from dyprys import registry

    inside = tmp_path / "texts"
    books = write_books(inside)
    conn, *_ = embedded_index(inside, books)          # the index, beside the books
    conn.close()
    entry = type("L", (), {"name": "inplace", "path": inside, "exists": True})()
    monkeypatch.setattr(registry, "libraries", lambda: [entry])
    monkeypatch.setattr(registry, "remove", lambda name: True)

    plan = service.library_deletion("inplace")
    assert plan["shares_directory"] is True
    # The books are what stays, and they are counted -- "not touched" is only
    # checkable against a number.
    assert plan["kept_files"] == len(books)
    assert plan["kept_bytes"] == sum(book.stat().st_size for book in books)
    # ...and they are not also counted as the index. That was the other half of
    # the same mistake: 249 MB offered for 347 MB, or a library's own books
    # described as "its vectors and its routing profile".
    assert plan["files"] == len(db.artefacts(inside))

    service.delete_library("inplace", confirm=True)

    assert all(book.exists() for book in books), "the books were deleted"
    assert inside.is_dir(), "the directory holding them went too"
    assert not db.index_exists(inside)
    assert not list(inside.glob("*.f32"))
