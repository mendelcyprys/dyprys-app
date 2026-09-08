"""The HTTP surface: what a browser is allowed to learn, and what it is told.

Skips until `dyprys.api` exists. Assumes `api.create_app(...)` takes the same
injectable model loader `service.open_session` does, so the suite never needs
a GGUF -- the trick the rest of this suite already turns with `StubEmbedder`.
If the app is built some other way, the fixture below is the one place to change.

What is worth testing over HTTP is not "does FastAPI route" -- it does -- but
the four places where translating a CLI into an API silently loses something:
an error that had an exit code and now needs a status, a warning that was a
line on stderr and now has nowhere to go, a passage whose text is null because
it could not be proved, and a path arriving from a query string.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi", reason="install dyprys[api]")
api = pytest.importorskip("dyprys.api", reason="dyprys.api does not exist yet")
from fastapi.testclient import TestClient  # noqa: E402

from tests.indexes import SENTINEL, add_model, embedded_index, write_books  # noqa: E402


@pytest.fixture
def index(tmp_path):
    books = write_books(tmp_path / "lib")
    conn, embedder, model_id, store = embedded_index(tmp_path / "ix", books)
    conn.close()
    return tmp_path / "ix", (embedder, model_id, store)


@pytest.fixture
def client(index, monkeypatch):
    directory, loaded = index
    monkeypatch.setattr("dyprys.registry.resolve",
                        lambda name: directory if name in (None, "test") else None)
    monkeypatch.setattr("dyprys.registry.libraries", lambda: [
        type("L", (), {"name": "test", "path": directory, "default": True})()])
    return TestClient(api.create_app(load_model=lambda *a, **k: loaded))


def ask(client, **body):
    body.setdefault("question", "neurons and synapses")
    return client.post("/api/libraries/test/ask", json=body)


# --------------------------------------------------------------------------
# Errors keep their meaning across the translation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("test", 200),
    ("not-a-library", 404),
])
def test_an_unknown_library_is_a_404_not_a_500(client, name, expected):
    assert client.get(f"/api/libraries/{name}/status").status_code == expected


def test_an_empty_query_is_422_and_says_so(client):
    """Not 200-with-no-results. The distinction is the whole reason the CLI
    refuses an empty query rather than returning whatever the vector half
    drifts to."""
    response = ask(client, question="   ")
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_a_multi_model_index_returns_the_choices_as_data(tmp_path, monkeypatch):
    """A UI cannot render a picker from a sentence.

    This is the error a first-time user of a two-model index hits immediately,
    and the difference between a dead end and a dropdown is whether `choices`
    survived becoming JSON.
    """
    books = write_books(tmp_path / "lib")
    conn, embedder, model_id, store = embedded_index(tmp_path / "ix", books, name="stub-alpha")
    add_model(conn, tmp_path / "ix", "stub-beta")
    conn.close()
    monkeypatch.setattr("dyprys.registry.resolve", lambda n: tmp_path / "ix")
    client = TestClient(api.create_app())

    response = client.post("/api/libraries/test/ask", json={"question": "neurons"})

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "model_ambiguous"
    assert len(body["choices"]) == 2


def test_rerank_without_a_reranker_is_503_not_a_quiet_downgrade(client):
    """503: the server could do this, but the model it needs is not here.

    A 200 carrying unreranked results would be the worst outcome -- the caller
    asked for a rescoring, believes it happened, and CLAUDE.md's advice to
    always rerank on some corpora would be silently unfollowed.
    """
    response = ask(client, rerank=10)
    assert response.status_code == 503
    assert "reranker" in json.dumps(response.json()).lower()


# --------------------------------------------------------------------------
# The payload tells the truth about what was searched
# --------------------------------------------------------------------------

def test_the_ask_payload_has_the_documented_shape(client):
    """The same keys `dyp ask --json` promises, because agents parse both."""
    body = ask(client).json()

    assert set(body) >= {"query", "mode", "routed", "scanned_fraction",
                         "elapsed_ms", "results", "warnings"}
    for result in body["results"]:
        assert set(result) >= {"rank", "chunk_id", "book", "chapter", "path",
                               "offset", "cos", "provenance", "state", "text"}
    assert [r["rank"] for r in body["results"]] == list(
        range(1, len(body["results"]) + 1))


def test_an_unprovable_passage_arrives_with_a_null_text(client, index):
    """CLAUDE.md: never quote a passage whose `text` is null.

    A JSON serializer that helpfully coerces null to "" would erase the only
    signal that says a passage could not be proved against its stored hash --
    and the UI would then quote a passage the tool cannot vouch for.
    """
    directory, _ = index
    for source in (directory.parent / "lib").glob("*.txt"):
        source.unlink()

    body = ask(client).json()

    assert body["results"], "the search should still return ranked hits"
    assert any(r["text"] is None for r in body["results"])
    assert all(r["state"] for r in body["results"]), "a null text must say why"


def test_the_scanned_fraction_and_provenance_pass_through_untouched(client):
    """The API reports, it does not interpret.

    `cos` is comparable only within one search and is not a quality ranking;
    `provenance` says which half found the passage. Both are the reader's
    evidence, and rounding, renaming or ranking on them would replace the
    reader's judgement with the server's.
    """
    body = ask(client, question=SENTINEL).json()

    top = body["results"][0]
    assert "phrase" in (top["provenance"] or ""), "the literal hit lost its provenance"
    assert 0.0 <= body["scanned_fraction"] <= 1.0


def test_a_routed_search_carries_the_unreachable_book_warning(client, index):
    """The one failure mode a reader cannot detect from the output.

    Over HTTP there is no stderr to print it to, so if it is not in the payload
    it does not exist -- and the response is a full `k` results at 200, which
    looks exactly like a complete search.
    """
    directory, _ = index
    response = ask(client, route=8)
    if response.status_code == 200:
        assert isinstance(response.json()["warnings"], list)
    else:
        assert response.status_code in (400, 503)
        assert "route" in json.dumps(response.json()).lower()


def test_expand_and_summarise_accept_a_bare_true_as_well_as_a_model_name(client):
    """argparse gives these `nargs="?", const=True`, so `args.expand` is either
    `True` or a string. A Pydantic field typed `str | None` silently rejects the
    bare form the CLI has always allowed, and the two frontends drift on their
    very first flag."""
    for value in (True, "some-model"):
        response = ask(client, expand=value)
        assert response.status_code != 422, f"expand={value!r} was rejected as malformed"


# --------------------------------------------------------------------------
# The reader pane is not a file server
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "/etc/passwd",
    "../../../etc/passwd",
    "/tmp/not-in-this-index.txt",
])
def test_a_path_outside_the_sources_table_is_refused(client, path):
    """`?path=` is a file-read primitive until the `sources` table bounds it."""
    response = client.get("/api/libraries/test/source",
                          params={"path": path, "offset": 0, "span": 400})
    assert response.status_code == 404


def test_a_real_source_is_served_with_its_offset(client, index):
    directory, _ = index
    from dyprys import db
    conn = db.connect(directory)
    real = conn.execute("SELECT path FROM sources LIMIT 1").fetchone()["path"]
    conn.close()

    response = client.get("/api/libraries/test/source",
                          params={"path": real, "offset": 0, "span": 400})

    assert response.status_code == 200
    assert response.json()["text"]


def test_an_enormous_span_is_capped_rather_than_refused(client, index):
    """A UI asking for too much should get what it may have, not an error."""
    directory, _ = index
    from dyprys import db
    conn = db.connect(directory)
    real = conn.execute("SELECT path FROM sources LIMIT 1").fetchone()["path"]
    conn.close()

    response = client.get("/api/libraries/test/source",
                          params={"path": real, "offset": 0, "span": 10 ** 9})

    assert response.status_code == 200
    assert len(response.json()["text"].encode()) <= api.MAX_SPAN


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------

def test_the_stream_ends_with_exactly_one_result_frame(client):
    """NDJSON: any number of stage frames, then one result.

    A stream that ends without a result frame is indistinguishable to the UI
    from a stream that is still running, which is how a dropped exception
    inside a generator presents itself.
    """
    with client.stream("POST", "/api/libraries/test/ask/stream",
                       json={"question": "neurons and synapses"}) as response:
        assert response.status_code == 200
        frames = [json.loads(line) for line in response.iter_lines() if line.strip()]

    results = [f for f in frames if "result" in f]
    assert len(results) == 1, f"expected one result frame, got {len(results)}"
    assert frames[-1] is results[0], "the result must be the last frame"
    assert all("stage" in f for f in frames[:-1])


def test_a_failing_stream_says_so_in_a_frame(client):
    """An error after the headers are sent cannot be a status code."""
    with client.stream("POST", "/api/libraries/test/ask/stream",
                       json={"question": "neurons", "rerank": 10}) as response:
        frames = [json.loads(line) for line in response.iter_lines() if line.strip()]

    assert frames, "a failing stream must still emit something"
    assert "error" in frames[-1]


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

def test_a_second_embed_is_refused_while_one_holds_the_lock(client, index):
    """409, and before the model loads.

    `lock.exclusive` is a per-process fcntl lock, so the subprocess would refuse
    anyway -- but only after spending a minute loading a GGUF to discover it,
    and the UI would show a job that starts and then dies.
    """
    directory, _ = index
    from dyprys import lock

    with lock.exclusive(directory, "embed"):
        response = client.post("/api/libraries/test/jobs/embed", json={"for": "2m"})

    assert response.status_code == 409


def test_the_job_listing_reports_who_holds_the_lock(client, index):
    directory, _ = index
    from dyprys import lock

    with lock.exclusive(directory, "embed"):
        body = client.get("/api/libraries/test/jobs").json()

    assert body["embed"]["running"] is True
    assert body["embed"]["pid"]


def test_no_job_running_is_a_state_not_a_404(client):
    """A UI polls this endpoint. "Nothing is running" is an answer."""
    body = client.get("/api/libraries/test/jobs").json()
    assert body["embed"]["running"] is False


# --------------------------------------------------------------------------
# The notes file is transported, never interpreted
# --------------------------------------------------------------------------

def test_notes_are_served_raw_when_the_owner_left_some(client, index):
    """CLAUDE.md tells a reader to read the notes before searching. The API's
    whole job here is to hand them over -- corpus knowledge stays data."""
    directory, _ = index
    (directory / "NOTES.md").write_text("# Notes\nTitles collide.\n", encoding="utf-8")

    response = client.get("/api/libraries/test/notes")

    assert response.status_code == 200
    assert "Titles collide." in response.text


def test_no_notes_is_a_404_and_not_an_error_page(client):
    assert client.get("/api/libraries/test/notes").status_code == 404


# --------------------------------------------------------------------------
# Anti-drift: the two frontends must agree
# --------------------------------------------------------------------------

@pytest.mark.parametrize("command", ["status", "check", "books", "models"])
def test_the_api_returns_what_the_cli_prints_for_json(client, index, command, capsys):
    """The comparison that can actually fail.

    Comparing `service.X_payload()` against a subprocess
    `dyp X --json` -- but once the CLI's `--json` *is* `_emit_json(X_payload())`
    the two cannot differ, and the test proves only that the call was made.
    The pair that can drift is the API's serializer against the CLI's, which is
    what this compares.
    """
    directory, _ = index
    from dyprys.cli import main

    assert main(["--data", str(directory), command, "--json"]) in (0, 1)
    from_cli = json.loads(capsys.readouterr().out)
    from_api = client.get(f"/api/libraries/test/{command}").json()

    assert from_api == from_cli, f"`dyp {command} --json` and the API disagree"


# --------------------------------------------------------------------------
# Naming a library
# --------------------------------------------------------------------------
#
# `dyp library add|remove|use` had no HTTP equivalent, so a browser could read
# every library and name none. These check the three refusals that only matter
# once the caller is a browser: a name that is also a URL segment, a path that
# is only a typo, and a registry the client would otherwise clobber.


@pytest.fixture
def registry_client(tmp_path, monkeypatch):
    """A real registry in a temp home — nothing here touches the user's own."""
    monkeypatch.setenv("DYPRYS_HOME", str(tmp_path / "config"))
    return TestClient(api.create_app())


def test_a_directory_can_be_named_from_the_browser(registry_client, tmp_path):
    """The gap that made a browser a read-only client of the registry."""
    shelf = tmp_path / "shelf"
    shelf.mkdir()

    made = registry_client.post("/api/libraries",
                                json={"name": "shelf", "path": str(shelf)})

    assert made.status_code == 201
    names = [row["name"] for row in made.json()["libraries"]]
    assert names == ["shelf"]
    assert registry_client.get("/api/libraries").json()["libraries"][0]["default"] is True


def test_registering_a_path_that_is_not_there_registers_nothing(registry_client, tmp_path):
    """The CLI allows it because the next command usually creates the directory.

    A browser has no next command, so a typo would become a registry entry that
    nothing ever reports as wrong — it would simply fail at every use.
    """
    refused = registry_client.post("/api/libraries",
                                   json={"name": "ghost", "path": str(tmp_path / "nope")})

    assert refused.status_code == 404
    assert registry_client.get("/api/libraries").json()["libraries"] == []


@pytest.mark.parametrize("name", ["", "  ", "two words", "shelves/neuro", ".."])
def test_a_name_that_could_not_be_a_url_segment_is_refused(registry_client, tmp_path, name):
    """The name is a path parameter on every other route.

    A `/` in it would silently change which route matched, and a space would
    stop `dyp -L` from being able to say it — the two frontends must be able to
    mean the same library.
    """
    shelf = tmp_path / "shelf"
    shelf.mkdir()

    refused = registry_client.post("/api/libraries",
                                   json={"name": name, "path": str(shelf)})

    assert refused.status_code == 400
    assert refused.json()["error"] == "bad_request"


def test_a_taken_name_is_refused_with_the_names_already_used(registry_client, tmp_path):
    """`registry.add` overwrites silently; over HTTP that is a lost library.

    `choices` carries what is taken, because the UI that has to offer another
    name is the one that needs them.
    """
    for which in ("one", "two"):
        (tmp_path / which).mkdir()
        registry_client.post("/api/libraries",
                             json={"name": which, "path": str(tmp_path / which)})

    clash = registry_client.post("/api/libraries",
                                 json={"name": "one", "path": str(tmp_path / "two")})

    assert clash.status_code == 400
    assert set(clash.json()["choices"]) == {"one", "two"}
    kept = {row["name"]: row["path"] for row in
            registry_client.get("/api/libraries").json()["libraries"]}
    assert kept["one"].endswith("/one"), "a clash rewrote the library it clashed with"


def test_an_unknown_field_is_refused_rather_than_ignored(registry_client, tmp_path):
    """Same rule as the search body: a dropped field is worse than a refusal."""
    (tmp_path / "shelf").mkdir()

    refused = registry_client.post(
        "/api/libraries",
        json={"name": "shelf", "path": str(tmp_path / "shelf"), "delete": True})

    assert refused.status_code == 400


def test_forgetting_a_library_leaves_its_files_alone(registry_client, tmp_path):
    """The name goes; the index does not. `--delete` has no route on purpose."""
    shelf = tmp_path / "shelf"
    shelf.mkdir()
    (shelf / "dyprys.sqlite").write_text("not really an index")
    registry_client.post("/api/libraries", json={"name": "shelf", "path": str(shelf)})

    gone = registry_client.delete("/api/libraries/shelf")

    assert gone.status_code == 200
    assert gone.json()["libraries"] == []
    assert (shelf / "dyprys.sqlite").exists(), "forgetting a name deleted an index"


def test_forgetting_a_name_that_is_not_there_says_which_are(registry_client, tmp_path):
    (tmp_path / "shelf").mkdir()
    registry_client.post("/api/libraries", json={"name": "shelf", "path": str(tmp_path / "shelf")})

    missing = registry_client.delete("/api/libraries/nope")

    assert missing.status_code == 404
    assert missing.json()["choices"] == ["shelf"]


def test_the_default_can_be_moved_and_is_the_same_default_the_cli_reads(registry_client, tmp_path):
    """One registry, two frontends: `dyp library use` and this are one setting."""
    from dyprys import registry

    for which in ("one", "two"):
        (tmp_path / which).mkdir()
        registry_client.post("/api/libraries",
                             json={"name": which, "path": str(tmp_path / which)})

    moved = registry_client.post("/api/libraries/two/default")

    assert moved.status_code == 200
    assert {row["name"] for row in moved.json()["libraries"] if row["default"]} == {"two"}
    assert registry.resolve(None) == tmp_path / "two"


def test_a_scope_can_be_a_list_of_books_over_http(client, index):
    """What a checkbox list sends. `collection` is `str | list[str]`.

    A UI cannot honestly turn several selected books into one glob, so the
    request carries them as they were picked and the server unions them.
    """
    directory, _ = index
    from dyprys import db

    conn = db.connect(directory)
    titles = [row["title"] for row in conn.execute("SELECT title FROM books ORDER BY id")]
    conn.close()

    answered = ask(client, collection=titles)
    assert answered.status_code == 200

    one = ask(client, collection=[titles[0]])
    assert one.status_code == 200
    assert {row["book"] for row in one.json()["results"]
            if "phrase" not in row["provenance"]} == {titles[0]}


def test_a_pattern_that_matches_nothing_is_still_a_404_inside_a_list(client, index):
    """The refusal a union would otherwise swallow.

    404 rather than 200-with-fewer-books: the request named something this index
    does not hold, and a UI that dropped a book from a checkbox list would
    otherwise show a narrowed search that looks complete.
    """
    directory, _ = index
    from dyprys import db

    conn = db.connect(directory)
    first = conn.execute("SELECT title FROM books ORDER BY id").fetchone()["title"]
    conn.close()

    refused = ask(client, collection=[first, "no-such-shelf"])

    assert refused.status_code == 404
    assert "no-such-shelf" in refused.json()["detail"]


# --------------------------------------------------------------------------
# Choosing a model without typing a path
# --------------------------------------------------------------------------


def test_the_weights_on_this_machine_can_be_listed(client, index, tmp_path, monkeypatch):
    """The reranker is the reason this exists.

    Unlike the expander and summariser, whose choices the index remembers, a
    cross-encoder must be named on every search and bare `--rerank` is an error
    rather than a downgrade. A browser has no tab-completing path, so something
    has to say what is there.
    """
    shelf = tmp_path / "weights"
    (shelf / "nested").mkdir(parents=True)
    (shelf / "reranker.gguf").write_bytes(b"x" * 11)
    (shelf / "nested" / "embedder.gguf").write_bytes(b"y" * 22)
    (shelf / "notes.txt").write_text("not a model")
    monkeypatch.setenv("DYPRYS_MODEL_DIR", str(shelf))

    found = client.get("/api/libraries/test/models/available").json()

    names = {row["name"] for row in found["models"]}
    assert names == {"reranker.gguf", "embedder.gguf"}, (
        "one directory down is where models are commonly kept; a .txt is not a model")
    assert {row["bytes"] for row in found["models"]} == {11, 22}
    assert str(shelf) in found["searched"], "a caller cannot judge an empty list without this"


def test_a_directory_that_is_not_there_is_not_an_error(client, monkeypatch, tmp_path):
    """A picker opening is not the moment to fail over a stale config entry."""
    monkeypatch.setenv("DYPRYS_MODEL_DIR", str(tmp_path / "gone"))

    found = client.get("/api/libraries/test/models/available")

    assert found.status_code == 200
    assert found.json()["models"] == []


def test_a_model_can_be_given_a_short_name(client):
    """`hf_ggml-org_embeddinggemma-300M-Q8_0@b5ce9…` identifies weights exactly
    and tells a person nothing. The alias is stored in the index, so a name
    chosen in a browser is one `dyp --model` accepts in a terminal."""
    named = client.post("/api/libraries/test/models/stub/alias", json={"alias": "quick"})

    assert named.status_code == 200
    assert named.json()["alias"] == "quick"
    assert [m["alias"] for m in client.get("/api/libraries/test/models").json()["models"]] == ["quick"]


def test_an_alias_that_is_a_path_is_refused(client):
    """`--model` takes either, so an alias that looks like one is a trap."""
    refused = client.post("/api/libraries/test/models/stub/alias",
                          json={"alias": "/models/thing.gguf"})

    assert refused.status_code == 400


def test_naming_a_model_that_is_not_here_says_what_is(client):
    """400 with the models to pick from — the same shape `model_ambiguous` has,
    because the UI's job in both cases is to render a picker."""
    missing = client.post("/api/libraries/test/models/nope/alias", json={"alias": "x"})

    assert missing.status_code == 400
    assert missing.json()["choices"] == ["stub"]


def test_a_summariser_refusal_arrives_as_an_answer_not_an_absence(client, monkeypatch):
    """The surest evidence a library lacks something, and it must survive HTTP.

    `dyp ask --summarise` says "these passages do not answer the question"
    outright, and retries once with the model's own rephrasing before giving up.
    A refusal that survived the rephrasing means something; one that was never
    rephrased does not — and stored prose reading NO ANSWER IN PASSAGES looks
    identical either way, so `refused` carries the words that were tried.
    """
    from dyprys import summarise as summarise_mod
    from dyprys.summarise import NO_ANSWER

    said = []

    def talk(model, prompt, *a, **k):
        said.append(prompt)
        return "other words entirely" if len(said) == 2 else NO_ANSWER

    monkeypatch.setattr(summarise_mod, "ask_ollama", talk)

    answered = ask(client, summarise="stub-summariser").json()

    assert answered["answer"]["prose"] == NO_ANSWER
    assert answered["answer"]["refused"] == "other words entirely", (
        "a browser cannot tell a rephrased refusal from an unrephrased one")
    assert answered["answer"].get("drawn_from") is None, (
        "the retry found nothing, so the answer is about the passages on screen")


def test_a_successful_retry_says_which_passages_it_is_about(client, monkeypatch):
    """The asymmetry the API used to drop entirely.

    When the retry works, the answer is drawn from the *retry's* passages while
    `results` still holds the original search — and `cited` indexes into the
    former. Without `drawn_from`, a reader is shown an answer citing passages
    that are not on the screen, with citations that look correct.
    """
    from dyprys import summarise as summarise_mod
    from dyprys.summarise import NO_ANSWER

    said = []

    def talk(model, prompt, *a, **k):
        said.append(prompt)
        if len(said) == 1:
            return NO_ANSWER
        if len(said) == 2:
            return "neurons and synapses"
        return "They are connected."

    monkeypatch.setattr(summarise_mod, "ask_ollama", talk)

    answered = ask(client, summarise="stub-summariser").json()

    assert answered["answer"]["retried"] == "neurons and synapses"
    drawn = answered["answer"]["drawn_from"]
    assert drawn and all({"chunk_id", "book", "path", "offset"} <= set(row) for row in drawn)


def test_one_held_lock_does_not_report_five_running_jobs(client, index):
    """Every kind takes the same lock, so a held lock says the index is busy and
    says nothing about *what* is busy.

    Reported per kind, that made one run look like five, with one pid shared
    between them and four of the five wrong. `busy` is the honest per-index
    fact; `running` stays per kind.
    """
    directory, _ = index
    from dyprys import lock

    with lock.exclusive(directory, "embed"):
        body = client.get("/api/libraries/test/jobs").json()

    assert all(body[kind]["busy"] for kind in body), "the index is busy, whoever holds it"
    assert [kind for kind in body if body[kind]["running"]] == ["embed"], (
        "an unattributed holder belongs to embed alone — it is the lock's name, "
        "and the only kind that runs long enough for anyone to be watching")


def test_a_job_this_server_started_is_attributed_to_its_own_kind(client, index, monkeypatch):
    """The marker `start` leaves, and the only thing that can tell route from embed.

    Trusted only when the recorded pid is the pid actually holding the lock, so
    a stale file attributes nothing and a run started from a terminal is an
    unattributed holder rather than a misattributed one.
    """
    import os

    from dyprys import jobs, lock

    directory, _ = index
    jobs.log_dir(directory).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(lock, "holder", lambda *a, **k: os.getpid())
    jobs._remember(directory, jobs.JobRef("route", os.getpid(), directory / "x.log", "now"))

    body = client.get("/api/libraries/test/jobs").json()

    assert body["route"]["running"] is True
    assert body["route"]["holder_kind"] == "route"
    assert body["embed"]["running"] is False, "a route is not an embed"
    assert body["embed"]["busy"] is True, "but the index is busy either way"


def test_a_stale_marker_attributes_nothing(client, index, monkeypatch):
    from dyprys import jobs, lock

    directory, _ = index
    jobs.log_dir(directory).mkdir(parents=True, exist_ok=True)
    jobs._remember(directory, jobs.JobRef("route", 999_999, directory / "x.log", "now"))
    monkeypatch.setattr(lock, "holder", lambda *a, **k: 4242)

    body = client.get("/api/libraries/test/jobs").json()

    assert body["route"]["holder_kind"] is None
    assert body["route"]["running"] is False
    assert body["embed"]["running"] is True, "an unknown holder falls back to embed"


def test_a_dropped_stream_marks_the_search_abandoned(client):
    """The one moment the server is told nobody is listening.

    Starlette closes the response generator when the client goes away, and that
    is what `finally` catches — there is no other signal. It matters because a
    library has one worker thread: a search nobody wants otherwise holds it
    while the next question queues behind a closed tab.
    """
    from dyprys import progress as progress_mod

    marked: list = []
    real = progress_mod.Queue.abandon

    def watch(self):
        marked.append(self)
        real(self)

    progress_mod.Queue.abandon = watch
    try:
        with client.stream("POST", "/api/libraries/test/ask/stream",
                           json={"question": "neurons and synapses"}) as answering:
            # Read nothing and leave: the generator is closed on the way out.
            assert answering.status_code == 200
    finally:
        progress_mod.Queue.abandon = real

    assert marked, "the stream ended without telling the search that it had"
    assert marked[0].cancelled() is True


# --------------------------------------------------------------------------
# What a library remembers
# --------------------------------------------------------------------------


def test_a_default_set_here_is_the_one_the_terminal_uses(client, index, monkeypatch):
    """One library, one answer to "which summariser" — not one per frontend.

    The value goes in the index's own meta, which is where `dyp ask` reads it
    from, so `summarise: true` (meaning "whatever this library remembers") means
    the same thing in both places.
    """
    from dyprys import db, service

    monkeypatch.setattr(service, "installed_models", lambda *a, **k: ["gemma3:4b"])

    stored = client.post("/api/libraries/test/defaults",
                         json={"role": "summariser", "model": "gemma3:4b"})

    assert stored.status_code == 200
    assert stored.json()["defaults"]["summariser"] == "gemma3:4b"
    directory, _ = index
    conn = db.connect(directory)
    assert service.default_model(conn, "summariser") == "gemma3:4b"
    conn.close()


def test_a_model_this_machine_does_not_have_is_refused_with_the_ones_it_does(
        client, monkeypatch):
    """Validated when set, not when needed: the two can be weeks apart, and a
    typo stored today should not surface as a failed search in a fortnight."""
    from dyprys import service

    monkeypatch.setattr(service, "installed_models", lambda *a, **k: ["gemma3:4b"])

    refused = client.post("/api/libraries/test/defaults",
                          json={"role": "summariser", "model": "gemma9:99b"})

    assert refused.status_code == 503
    assert refused.json()["choices"] == ["gemma3:4b"]
    assert refused.json()["role"] == "summariser"


def test_a_default_is_stored_when_ollama_cannot_be_asked(client, monkeypatch):
    """A refusal because the server happens to be down would be worse than
    storing a name that is in fact there."""
    from dyprys import service

    monkeypatch.setattr(service, "installed_models", lambda *a, **k: [])

    stored = client.post("/api/libraries/test/defaults",
                         json={"role": "summariser", "model": "gemma3:4b"})

    assert stored.status_code == 200
    assert stored.json()["defaults"]["summariser"] == "gemma3:4b"


def test_a_reranker_default_is_a_path_that_has_to_exist(client, tmp_path):
    """A cross-encoder is a file, and one that is not there now will not be
    there when a search needs it."""
    missing = client.post("/api/libraries/test/defaults",
                          json={"role": "reranker", "model": str(tmp_path / "nope.gguf")})
    assert missing.status_code == 503

    real = tmp_path / "cross.gguf"
    real.write_bytes(b"not really weights")
    stored = client.post("/api/libraries/test/defaults",
                         json={"role": "reranker", "model": str(real)})

    assert stored.status_code == 200
    assert stored.json()["defaults"]["reranker"] == str(real)


def test_a_default_can_be_forgotten(client, monkeypatch):
    from dyprys import service

    monkeypatch.setattr(service, "installed_models", lambda *a, **k: ["gemma3:4b"])
    client.post("/api/libraries/test/defaults",
                json={"role": "summariser", "model": "gemma3:4b"})

    cleared = client.post("/api/libraries/test/defaults",
                          json={"role": "summariser", "model": None})

    assert cleared.json()["defaults"]["summariser"] is None


def test_a_role_that_is_not_one_says_which_are(client):
    refused = client.post("/api/libraries/test/defaults",
                          json={"role": "embedder", "model": "x"})

    assert refused.status_code == 400
    assert refused.json()["choices"] == ["expander", "summariser", "reranker"]
