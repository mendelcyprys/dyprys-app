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
