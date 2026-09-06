import pytest
"""The package imports and the console script is wired up."""

import subprocess
import sys
import time


def test_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "dyprys.cli"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "Search a personal library" in result.stdout


def test_a_stored_timestamp_renders_in_the_reader_s_timezone(monkeypatch):
    """Stored UTC, shown local. Slicing the ISO string did neither.

    `[:19]` drops the `+00:00` and prints UTC digits under a heading that reads
    as local time -- an hour out on BST, thirteen in Auckland. Found by
    reconciling the embed journal against a shell log of the same runs, which is
    the one job the journal has.
    """
    import os
    import time as time_mod

    from dyprys.cli import _local

    monkeypatch.setitem(os.environ, "TZ", "Europe/London")
    time_mod.tzset()
    try:
        # 2026-07-01 is inside British Summer Time, so local is UTC+1.
        assert _local("2026-07-01T12:00:00+00:00") == "2026-07-01 13:00:00"
        # ...and midwinter is not, so the offset is not hardcoded anywhere.
        assert _local("2026-01-01T12:00:00+00:00") == "2026-01-01 12:00:00"
    finally:
        monkeypatch.undo()
        time_mod.tzset()


def test_a_timestamp_without_an_offset_is_read_as_utc(monkeypatch):
    """Rows written before the journal recorded offsets must not shift."""
    import os
    import time as time_mod

    from dyprys.cli import _local

    monkeypatch.setitem(os.environ, "TZ", "Europe/London")
    time_mod.tzset()
    try:
        assert _local("2026-07-01T12:00:00") == "2026-07-01 13:00:00"
    finally:
        monkeypatch.undo()
        time_mod.tzset()


def test_every_command_appears_in_the_grouped_help():
    """The grouped listing replaces argparse's own, so it must stay complete.

    `dyp --help` prints `COMMAND_GROUPS` instead of the flat alphabetical list,
    which is the point — eighteen names in one column say nothing about which
    three you need today. The cost is that a command added without a line here
    becomes invisible, so this asserts the two cannot drift apart.
    """
    import argparse
    import re

    from dyprys.cli import COMMAND_GROUPS, build_parser

    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    listed = set(re.findall(r"\b[a-z]+\b", COMMAND_GROUPS))

    assert sub.choices, "no subcommands registered"
    missing = sorted(set(sub.choices) - listed)
    assert not missing, f"missing from the grouped help: {missing}"


def _flags(command):
    """The option strings a subcommand accepts."""
    from dyprys.cli import build_parser

    parser = build_parser()
    sub = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
    found = set()
    for action in sub.choices[command]._actions:
        found.update(action.option_strings)
    return found - {"-h", "--help"}


def test_ask_and_eval_agree_on_every_flag_they_share():
    """They are the same search reached two ways, and they drift by hand.

    `--expand` and `--expander` were added to both in one sitting by editing two
    places; the next such flag is where one gets forgotten, and the symptom is
    an eval that cannot measure what `ask` actually does — which is the one
    thing the harness exists for.
    """
    ask, ev = _flags("ask"), _flags("eval")
    # eval alone takes the question set and the arms to compare.
    eval_only = {"--questions", "--lexical", "--compare"}
    # These change how results are *presented*, not which results they are, so a
    # retrieval harness has nothing to measure in any of them — any *other*
    # ask-only flag is drift. --json is the machine-readable presentation, --full
    # and --quiet the human one, --summarise drafts prose from what was found.
    ask_only = {"--summarise", "--full", "--json", "--quiet", "-q"}

    assert ask - ev == ask_only, f"unexpected ask-only flags: {sorted(ask - ev - ask_only)}"
    assert ev - ask == eval_only, f"unexpected eval-only flags: {sorted(ev - ask - eval_only)}"


def test_counts_are_spelled_the_same_way_everywhere():
    """`--limit` on one command and `-n` on another is a thing to look up twice."""
    assert "--limit" in _flags("embed")
    assert "--limit" in _flags("history")


def test_the_long_running_commands_can_be_scoped_to_a_shelf():
    """The gap that mattered: search could target part of a library, work could not."""
    assert {"-c", "--collection"} <= _flags("embed")
    assert {"-c", "--collection"} <= _flags("ask")
    assert {"-c", "--collection"} <= _flags("eval")


@pytest.mark.parametrize("argv", [
    ["history"],
    ["history", "-n", "3"],
    ["history", "--limit", "3"],
    ["status"],
    ["books"],
    ["check"],
])
def test_the_reporting_commands_actually_run(tmp_path, argv):
    """Inspecting a parser is not running a command.

    Adding `--limit` beside `-n` moved argparse's dest from `n` to `limit`, so
    `dyp history` raised AttributeError — while a test that only asked the
    parser which flags exist passed, and the break shipped. Every read-only
    command is now actually invoked against a real index.
    """
    from dyprys.cli import main

    book = tmp_path / "book.txt"
    book.write_text("\n\n".join(f"Paragraph {n} about neurons. " * 8 for n in range(12)),
                    encoding="utf-8")
    data = str(tmp_path / "ix")
    assert main(["--data", data, "add", str(book)]) == 0

    assert main(["--data", data, *argv]) in (0, 1)


def test_every_result_carries_a_cosine_even_when_only_bm25_found_it(tmp_path):
    """A literal match with a low cosine is the useful case, not an edge case.

    It says the passage contains your words without being about them, which is
    exactly when a keyword hit misleads — and on a real index the passage
    holding a remembered phrase scored 0.42 while an unrelated book scored 0.63.
    Computing it for the fused list is k dot products against vectors already on
    disk, so there is no reason to show it for only half the results.
    """
    import numpy as np

    from dyprys import db as _db
    from dyprys.cli import _searcher
    from dyprys.embed import store_for
    from dyprys.lexical import backfill
    from tests.conftest import DIM, StubEmbedder

    book = tmp_path / "book.txt"
    book.write_text("\n\n".join(
        f"Paragraph {n} concerning neurons and synapses. " * 9 for n in range(30)),
        encoding="utf-8")
    conn = _db.connect(tmp_path / "ix")
    from dyprys.ingest import ingest_paths
    ingest_paths(conn, [book])
    embedder = StubEmbedder()
    model = _db.model_id(conn, "stub", DIM)
    store = store_for(conn, tmp_path / "ix", model, DIM)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            _db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    for cid in range(1, total + 1):
        store.write(cid, embedder.embed_query(f"chunk {cid}"))
    store.flush()
    backfill(conn)

    run = _searcher(conn, store, embedder, model, "hybrid", None)
    hits = run("Paragraph 7 concerning neurons", 5)

    assert hits, "the fixture should return something"
    for hit in hits:
        assert hit.chunk_id in run.cosine, f"chunk {hit.chunk_id} has no cosine"
        assert -1.0001 <= run.cosine[hit.chunk_id] <= 1.0001
        assert hit.chunk_id in run.why
    conn.close()


def test_a_missing_optional_model_names_something_runnable(monkeypatch):
    """Naming the flag is not help.

    Someone who has never installed ollama and owns no .gguf learns nothing
    from "pass --expander PATH.gguf". All three optional roles failed that way,
    in three different wordings, while `dyp models` listed only the embedding
    model — so there was no path from the error to a working command.
    """
    from dyprys import cli

    monkeypatch.setattr(cli, "ollama_models", lambda *a, **k: ["gemma3:4b", "qwen3:4b"])
    message = cli._no_model_for("expansion", "--expand", "DYPRYS_EXPANDER")

    assert "gemma3:4b" in message
    assert "dyp ask" in message and "--expand gemma3:4b" in message
    assert "dyp models" in message


def test_the_reranker_is_not_offered_a_chat_model(monkeypatch):
    """It scores (query, passage) pairs; a chat name cannot do the job.

    Suggesting one would be a recommendation that fails, which is worse than
    the unhelpful message it replaced.
    """
    from dyprys import cli

    monkeypatch.setattr(cli, "ollama_models", lambda *a, **k: ["gemma3:4b"])
    message = cli._no_model_for("reranker", "--reranker", "DYPRYS_RERANKER", chat=False)

    assert "gemma3:4b" not in message
    assert "cross-encoder" in message


def test_an_absent_ollama_does_not_stall_the_listing():
    """`dyp models` calls this, and most users will not have ollama."""
    import time

    from dyprys.cli import ollama_models

    began = time.monotonic()
    assert ollama_models("http://localhost:1") == []
    assert time.monotonic() - began < 2.0


def test_every_optional_role_is_listed_and_settable():
    """A role that exists only in argparse is a role nobody finds.

    Each must have an environment variable, a flag, and a key that both
    `dyp models --KEY` and the stored default agree on.
    """
    from dyprys.cli import OPTIONAL_ROLES, _env_for, build_parser

    settable = _flags("models")
    for role, env, flag, key, what in OPTIONAL_ROLES:
        assert env.startswith("DYPRYS_"), role
        assert flag.startswith("--"), role
        assert f"--{key}" in settable, f"{role} has no `dyp models --{key}`"
        assert _env_for(key) == env, f"{role} names the wrong variable"
        assert what, role
    assert build_parser()


def _index(tmp_path):
    from dyprys import db as _db

    return _db.connect(tmp_path / "ix")


def test_a_remembered_default_is_used_when_nothing_else_says(tmp_path, monkeypatch):
    from dyprys import db as _db
    from dyprys.cli import resolve_model

    conn = _index(tmp_path)
    monkeypatch.delenv("DYPRYS_EXPANDER", raising=False)
    with conn:
        _db.set_meta(conn, "model.expander", "gemma3:4b")

    assert resolve_model(conn, "expander", "DYPRYS_EXPANDER") == "gemma3:4b"
    conn.close()


def test_explicit_beats_remembered_and_the_environment(tmp_path, monkeypatch):
    """The same rule `--data` follows against a stored default library.

    A command that names its model must never be redirected by a setting made
    weeks earlier — that is the failure mode of remembered state, and it is
    silent, because the wrong model still returns results.
    """
    from dyprys import db as _db
    from dyprys.cli import resolve_model

    conn = _index(tmp_path)
    monkeypatch.setenv("DYPRYS_EXPANDER", "from-env")
    with conn:
        _db.set_meta(conn, "model.expander", "from-index")

    assert resolve_model(conn, "expander", "DYPRYS_EXPANDER", "typed") == "typed"
    # ...and the environment beats the index, being scoped to this shell.
    assert resolve_model(conn, "expander", "DYPRYS_EXPANDER") == "from-env"
    conn.close()


def test_a_default_can_be_forgotten(tmp_path, monkeypatch):
    from dyprys.cli import _set_default_model, default_model

    conn = _index(tmp_path)
    monkeypatch.setattr("dyprys.cli.ollama_models", lambda *a, **k: ["gemma3:4b"])
    _set_default_model(conn, "expander", "gemma3:4b")
    assert default_model(conn, "expander") == "gemma3:4b"

    _set_default_model(conn, "expander", "none")
    assert default_model(conn, "expander") is None
    conn.close()


def test_a_default_is_checked_when_it_is_set_not_when_it_is_needed(tmp_path, monkeypatch):
    """Setting and using can be weeks apart.

    A typo stored today should be refused today, not surface as a failed search
    in a fortnight with no clue where it came from.
    """
    from dyprys.cli import _set_default_model, default_model

    conn = _index(tmp_path)
    monkeypatch.setattr("dyprys.cli.ollama_models", lambda *a, **k: ["gemma3:4b"])

    assert _set_default_model(conn, "expander", "gemma3-4b-typo") == 1
    assert default_model(conn, "expander") is None, "a rejected value was stored"
    conn.close()


def test_progress_leaves_what_it_printed_on_the_screen(monkeypatch, capsys):
    """The interesting part must not flash past and vanish.

    An earlier version wrote a transient line and erased it, so the rephrasing
    the model chose and the books routing picked were gone before they could be
    read. Nothing is erased now, and nothing writes cursor-control codes — a
    captured log used to be full of `[K`.
    """
    from dyprys import cli

    stages = cli.Stages(on=True)
    stages.note("routed to 5 of 3,453 books")
    stages.note("Principles of Neural Science", indent=1)
    stages.working("drafting …")

    err = capsys.readouterr().err
    assert "\033[K" not in err and "\r" not in err
    assert "routed to 5 of 3,453 books" in err
    assert "Principles of Neural Science" in err
    assert "drafting" in err


def test_progress_can_be_silenced(capsys):
    """`-q` is for scripts; the commentary is on stderr but still noise there."""
    from dyprys import cli

    cli.Stages(on=False).note("routed to", "somewhere")

    assert capsys.readouterr().err == ""


def test_ask_only_flags_are_the_presentation_ones():
    """Anything else appearing only on `ask` is drift between it and `eval`."""
    ask, ev = _flags("ask"), _flags("eval")

    assert ask - ev == {"--summarise", "--full", "--json", "--quiet", "-q"}


def test_a_question_and_its_answer_are_kept(tmp_path):
    """A question worth asking twice should not need reconstructing from memory."""
    import json

    from dyprys import db as _db

    conn = _db.connect(tmp_path / "ix")
    _db.record_question(conn, "what is myelin", "hybrid", 431.2, {
        "routed": True, "books": 5,
        "models": {"expander": "gemma3:4b"},
        "hits": [{"chunk": 7, "title": "A Book", "path": "/b.txt",
                  "offset": 12, "cos": 0.44, "why": "vec 1"}],
        "answer": {"prose": "…", "verified": [], "rejected": []},
    })

    rows = _db.questions_asked(conn)
    assert len(rows) == 1
    kept = json.loads(rows[0]["detail"])
    assert kept["models"]["expander"] == "gemma3:4b"
    assert kept["hits"][0]["chunk"] == 7
    conn.close()


def test_questions_can_be_searched_and_forgotten(tmp_path):
    from dyprys import db as _db

    conn = _db.connect(tmp_path / "ix")
    for q in ("about myelin", "about synapses", "about myelin again"):
        _db.record_question(conn, q, "hybrid", 1.0, {})

    assert len(_db.questions_asked(conn, match="myelin")) == 2
    assert _db.forget_questions(conn, which=_db.questions_asked(conn)[0]["id"]) == 1
    assert len(_db.questions_asked(conn)) == 2
    assert _db.forget_questions(conn) == 2
    assert _db.questions_asked(conn) == []
    conn.close()


def test_forgetting_by_date_keeps_the_recent_ones(tmp_path):
    from dyprys import db as _db

    conn = _db.connect(tmp_path / "ix")
    with conn:
        conn.execute("INSERT INTO asked (at, question, mode, ms, detail) "
                     "VALUES ('2020-01-01T00:00:00+00:00', 'old', 'hybrid', 1.0, '{}')")
        conn.execute("INSERT INTO asked (at, question, mode, ms, detail) "
                     "VALUES ('2030-01-01T00:00:00+00:00', 'new', 'hybrid', 1.0, '{}')")

    assert _db.forget_questions(conn, before="2025-01-01") == 1
    assert [r["question"] for r in _db.questions_asked(conn)] == ["new"]
    conn.close()


def test_the_listing_columns_line_up_without_colour(tmp_path, capsys, monkeypatch):
    """Escape codes have no width on screen and full width to str.format.

    Padding a styled string makes every column ragged; the first version did
    exactly that and the header sat four characters from its own numbers.
    """
    from dyprys import cli, db as _db
    from dyprys import term as term_mod

    monkeypatch.setattr(term_mod, "COLOUR", False)
    conn = _db.connect(tmp_path / "ix")
    _db.record_question(conn, "a short question", "hybrid", 12.0, {})
    _db.record_question(conn, "another question", "hybrid", 3456.0, {})

    class Args:
        which = None
        limit = 10
        find = None
        forget = None
        yes = False

    cli._asked(conn, Args())
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    header, *rows = lines
    for row in rows[:2]:
        assert row.index("hybrid") if "hybrid" in row else True
        assert len(row) > 0 and row[:5].strip().isdigit()
    assert header.index("when") == rows[0].index("2"), "columns do not line up"
    conn.close()


def test_watch_follows_one_model_not_the_sum_of_all(tmp_path):
    """A second model in a finished library showed over 100% done.

    `SUM(n_embedded)` across every model counted the first model's completed
    284,627 toward the second model's total, so watching a fresh embed into an
    already-embedded library reported more progress than there was work.
    """
    from dyprys import db as _db
    from dyprys.cli import _model_being_embedded
    from dyprys.ingest import ingest_paths

    library = tmp_path / "lib"
    library.mkdir()
    (library / "b.txt").write_text("\n\n".join(
        f"Paragraph {n} about neurons. " * 10 for n in range(30)), encoding="utf-8")
    conn = _db.connect(tmp_path / "ix")
    ingest_paths(conn, [library])
    total = conn.execute("SELECT SUM(chunk_count) FROM segments").fetchone()[0]
    chunking = _db.chunkings(conn)[0]["id"]

    finished = _db.model_id(conn, "done@aaaaaaaaaaaa", 8)
    running = _db.model_id(conn, "busy@bbbbbbbbbbbb", 8)
    _db.bind_chunking(conn, finished, chunking)
    _db.bind_chunking(conn, running, chunking)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            _db.set_embedded_prefix(conn, finished, seg["id"], seg["chunk_count"])
            _db.set_embedded_prefix(conn, running, seg["id"], 1)

    model_id, name, bound = _model_being_embedded(conn)
    assert model_id == running, "watch would have followed the finished model"

    done = conn.execute(
        "SELECT COALESCE(SUM(n_embedded), 0) FROM segment_progress WHERE model_id = ?",
        (model_id,)).fetchone()[0]
    assert done < total, f"{done} of {total} is over 100%"
    conn.close()


def test_k_below_one_is_refused_by_both_ask_and_eval():
    """`-k 0` asks for no passages. It used to fall through to an empty result and
    be reported as "nothing embedded yet — run `dyp embed`", sending the user to
    re-embed a finished index over a bad argument."""
    from dyprys.cli import build_parser

    parser = build_parser()
    for command in ("ask", "eval"):
        for bad in ("0", "-3"):
            argv = [command, "q"] if command == "ask" else [command]
            with pytest.raises(SystemExit) as caught:
                parser.parse_args(argv + ["-k", bad])
            assert caught.value.code == 2


def test_a_k_of_one_or_more_is_accepted():
    from dyprys.cli import build_parser

    assert build_parser().parse_args(["ask", "q", "-k", "1"]).k == 1
    assert build_parser().parse_args(["ask", "q", "-k", "50"]).k == 50


def test_an_empty_query_is_refused_before_any_search(tmp_path, capsys):
    """An empty or whitespace query embeds to a meaningless vector and matches no
    words, so hybrid search returns whatever the vector half drifts to. Refuse it
    rather than present noise as answers at exit 0."""
    from dyprys import db as _db
    from dyprys.cli import _ask, build_parser
    from dyprys.ingest import ingest_paths

    book = tmp_path / "b.txt"
    book.write_text("Paragraph about neurons. " * 400, encoding="utf-8")
    conn = _db.connect(tmp_path / "ix")
    ingest_paths(conn, [book])

    for blank in ("", "   ", "\t \n"):
        args = build_parser().parse_args(["ask", blank])
        assert _ask(conn, tmp_path / "ix", args) == 2
    assert "empty query" in capsys.readouterr().err
    conn.close()


def test_a_padded_query_is_stripped_not_rejected(tmp_path):
    """Surrounding whitespace is trimmed, so "  synapse  " searches for "synapse"."""
    from dyprys.cli import build_parser

    args = build_parser().parse_args(["ask", "  synapse  "])
    # The strip happens inside _ask; here we assert the parser keeps it verbatim
    # so _ask is the single place that owns the cleaning.
    assert args.question == "  synapse  "


def test_json_payload_shape_is_stable_and_parseable():
    """The machine-readable form an agent parses. Built as a pure function from
    resolved passages, so it needs no model or index to check."""
    import json

    from dyprys.cli import results_as_json
    from dyprys.search import Passage
    from dyprys.text import EXACT, MISSING

    passages = [
        Passage(chunk_id=21482, score=0.03, title="Principles", path="/b/p.txt",
                chapter=0, text="the axon leaves the soma", state=EXACT, offset=523017),
        Passage(chunk_id=99, score=0.01, title="Other", path="/b/o.txt",
                chapter=3, text=None, state=MISSING, offset=0),
    ]
    why = {21482: "vec 1 · phrase 1", 99: "words 2"}
    cosine = {21482: 0.5199}

    payload = results_as_json("what is an axon", "hybrid", True, 0.0141, 91.23,
                              passages, why, cosine)
    # round-trips as JSON
    back = json.loads(json.dumps(payload))
    assert back["query"] == "what is an axon"
    assert back["mode"] == "hybrid" and back["routed"] is True
    assert back["scanned_fraction"] == 0.0141 and back["elapsed_ms"] == 91.2

    first, second = back["results"]
    assert first == {
        "rank": 1, "chunk_id": 21482, "book": "Principles", "chapter": None,
        "path": "/b/p.txt", "offset": 523017, "cos": 0.5199,
        "provenance": "vec 1 · phrase 1", "state": EXACT,
        "text": "the axon leaves the soma",
    }
    # an unproved passage carries text=null and its state, not a fabricated quote
    assert second["text"] is None and second["state"] == MISSING
    assert second["cos"] is None and second["chapter"] == 4


def _run_capturing(argv, capsys):
    """Run `dyp argv` and return (exit_code, parsed_json_stdout)."""
    import json

    from dyprys.cli import main

    capsys.readouterr()          # drain any output left by a previous command
    code = main(argv)
    out = capsys.readouterr().out
    return code, json.loads(out)


def test_inspection_commands_emit_valid_json(tmp_path, capsys):
    """status/check/books all --json on an ingested (unembedded) index parse and
    carry the fields an agent drives on. No model needed — this is the shape."""
    from dyprys.cli import main

    book = tmp_path / "b.txt"
    book.write_text("\n\n".join(f"Paragraph {n} about neurons. " * 9 for n in range(20)),
                    encoding="utf-8")
    data = str(tmp_path / "ix")
    assert main(["--data", data, "add", str(book)]) == 0

    code, status = _run_capturing(["--data", data, "status", "--json"], capsys)
    assert code == 0
    assert status["books"] == 1 and status["chunks"] > 0
    assert status["models"] == [] and status["compaction_interrupted"] is False

    code, check = _run_capturing(["--data", data, "check", "--json"], capsys)
    assert code == 0
    assert check["drift"]["clean"] is True
    assert check["live_chunks"] == status["chunks"]
    assert "lexical_complete" in check and check["models"] == []

    code, books = _run_capturing(["--data", data, "books", "--json"], capsys)
    assert code == 0
    assert len(books["books"]) == 1
    b = books["books"][0]
    assert b["title"] == "b" and b["chunks"] == status["chunks"]
    assert isinstance(b["sources"], list) and b["sources"][0]["present"] is True


def _split_two_ways(tmp_path, capsys):
    """An index chunked twice, with one model finished on the coarse half."""
    from dyprys import db as _db
    from dyprys.cli import main

    book = tmp_path / "b.txt"
    book.write_text("\n\n".join(f"Paragraph {n} about neurons. " * 40 for n in range(60)),
                    encoding="utf-8")
    data = str(tmp_path / "ix")
    assert main(["--data", data, "add", str(book)]) == 0
    assert main(["--data", data, "add", str(book), "--target", "1200"]) == 0
    capsys.readouterr()

    conn = _db.connect(tmp_path / "ix")
    coarse, fine = (r["id"] for r in conn.execute("SELECT id FROM chunkings ORDER BY id"))
    model = _db.model_id(conn, "coarse-model@aaaaaaaaaaaa", 8)
    _db.bind_chunking(conn, model, coarse)
    with conn:
        for seg in conn.execute(
            "SELECT id, chunk_count FROM segments WHERE chunking_id = ?", (coarse,)
        ).fetchall():
            _db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    mine = conn.execute(
        "SELECT SUM(chunk_count) FROM segments WHERE chunking_id = ?", (coarse,)
    ).fetchone()[0]
    conn.close()
    return data, mine


def test_a_finished_model_reads_as_finished_in_a_split_library(tmp_path, capsys):
    """`status` and `books` divide a model's progress by its own chunking.

    Over the library total instead, a model that has embedded every chunk it
    is responsible for reported ~25% and every book as part-way -- which reads
    as "keep embedding" for work that is already done.
    """
    data, mine = _split_two_ways(tmp_path, capsys)

    code, status = _run_capturing(["--data", data, "status", "--json"], capsys)
    assert code == 0
    model = status["models"][0]
    assert model["embedded"] == mine
    assert model["live_chunks"] == mine, "its own chunking, not the library"
    assert model["live_chunks"] < status["chunks"], "the fixture must be split two ways"
    assert model["coverage"] == 1.0

    code, books = _run_capturing(["--data", data, "books", "--json"], capsys)
    assert code == 0
    one = books["books"][0]
    assert one["live_chunks"]["coarse-model@aaaaaaaaaaaa"] == mine
    assert one["embedded"]["coarse-model@aaaaaaaaaaaa"] == mine

    # and `check`, which was already right, must agree with both
    code, check = _run_capturing(["--data", data, "check", "--json"], capsys)
    assert check["models"][0]["coverage"] == 1.0


def test_a_finished_model_is_counted_complete_not_part_way(tmp_path, capsys):
    """The human summary said "0 books complete" for a model with nothing left."""
    from dyprys.cli import main

    data, _ = _split_two_ways(tmp_path, capsys)
    assert main(["--data", data, "books"]) == 0
    out = capsys.readouterr().out

    assert "1 books complete" in out
    assert "part-way" not in out, out


def test_the_next_step_sees_a_model_that_is_not_the_furthest_ahead(tmp_path, capsys):
    """Asked only of the busiest model, an unfinished second model is invisible.

    A finished model has nothing outstanding, so the hint fell through it and
    went quiet while another model still had hours of embedding to do.
    """
    from dyprys import db as _db
    from dyprys.cli import next_step

    data, _ = _split_two_ways(tmp_path, capsys)
    conn = _db.connect(tmp_path / "ix")
    fine = [r["id"] for r in conn.execute("SELECT id FROM chunkings ORDER BY id")][1]
    behind = _db.model_id(conn, "fine-model@bbbbbbbbbbbb", 8)
    _db.bind_chunking(conn, behind, fine)
    segments = conn.execute(
        "SELECT id, chunk_count FROM segments WHERE chunking_id = ?", (fine,)
    ).fetchall()
    with conn:  # started, deliberately not finished
        _db.set_embedded_prefix(conn, behind, segments[0]["id"], 1)

    step = next_step(conn)
    conn.close()
    assert step is not None, "a model with work outstanding must not be silent"
    assert "embed" in step


def test_the_model_name_an_error_suggests_can_be_pasted_back(tmp_path, capsys):
    """A middle-elided name is not a name: pasted back it resolves to nothing."""
    from dyprys import db as _db
    from dyprys.cli import _handle, _weights_for

    data, _ = _split_two_ways(tmp_path, capsys)
    conn = _db.connect(tmp_path / "ix")
    _db.model_id(conn, "second-model@bbbbbbbbbbbb", 8)

    _, problem = _weights_for(conn, None)
    assert problem and "2 models" in problem

    for row in conn.execute("SELECT name, alias FROM models").fetchall():
        handle = _handle(row["name"], row["alias"])
        assert "…" not in handle, f"{handle!r} cannot be typed"
        assert handle in problem, "the error must offer the handle that works"
        assert _db.find_model(conn, handle) is not None, f"{handle!r} resolves to nothing"
    conn.close()


def test_watch_waits_out_the_gap_before_an_embed_takes_the_lock(tmp_path):
    """An embed loads its weights *before* it locks the index.

    For those seconds the index is unlocked while a run is plainly starting,
    and `dyp watch` in the next terminal lands there nearly every time. It
    reported "no embedding run in progress" and exited, which reads as the
    embed having died and invites starting a second one.
    """
    import threading
    import time as _time

    from dyprys.cli import _await_embed
    from dyprys.lock import exclusive

    directory = tmp_path / "ix"
    directory.mkdir()
    assert _await_embed(directory, grace=0) is None, "nothing has started yet"

    running, finish = threading.Event(), threading.Event()

    def starts_late():
        _time.sleep(0.4)  # stands in for loading the weights
        with exclusive(directory, "embed"):
            running.set()
            finish.wait(5)

    embed = threading.Thread(target=starts_late)
    embed.start()
    try:
        assert _await_embed(directory, grace=5, tick=0.05) is not None, (
            "watch gave up on a run that was still starting")
        assert running.is_set()
    finally:
        finish.set()
        embed.join(5)


def test_watch_still_gives_up_when_nothing_is_running(tmp_path):
    """The grace period must not turn a wrong guess into a hang."""
    from dyprys.cli import _await_embed

    directory = tmp_path / "ix"
    directory.mkdir()

    began = time.monotonic()
    assert _await_embed(directory, grace=0.3, tick=0.05) is None
    assert time.monotonic() - began < 3, "waited far longer than the grace given"


def test_status_points_at_a_notes_file_when_the_owner_left_one(tmp_path, capsys):
    """What a corpus needs said about it has to be discoverable somewhere.

    A library carries facts the index cannot expose — which shelf `-c` cuts
    along, which terms its translation keeps — and a reader who is not told
    they exist rediscovers them, or searches worse without knowing why.
    """
    from dyprys.cli import main

    book = tmp_path / "b.txt"
    book.write_text("\n\n".join(f"Paragraph {n} about neurons. " * 9 for n in range(12)),
                    encoding="utf-8")
    data = tmp_path / "ix"
    assert main(["--data", str(data), "add", str(book)]) == 0
    capsys.readouterr()

    # Nothing to say when there is no file, and nothing broken by its absence.
    code, status = _run_capturing(["--data", str(data), "status", "--json"], capsys)
    assert code == 0 and status["notes"] is None

    (data / "NOTES.md").write_text("-c cuts by shelf here, not by topic",
                                   encoding="utf-8")

    code, status = _run_capturing(["--data", str(data), "status", "--json"], capsys)
    assert status["notes"] == str(data / "NOTES.md")

    assert main(["--data", str(data), "status"]) == 0
    out = capsys.readouterr().out
    assert "NOTES.md" in out
    assert "read it first" in out, "a path with no instruction is easy to skim past"


def test_rerank_given_the_model_file_says_which_flag_wanted_it(tmp_path):
    """`--rerank` takes a count and `--reranker` the GGUF; they differ by two
    letters. argparse's own reply repeats the path and names neither flag."""
    import argparse as _argparse

    from dyprys.cli import _rerank_depth

    assert _rerank_depth("25") == 25

    with pytest.raises(_argparse.ArgumentTypeError) as caught:
        _rerank_depth("/models/qwen3-reranker-0.6b.gguf")
    assert "--reranker" in str(caught.value), "must name the flag that wanted it"

    # A plain typo is not a path, so it should not be told to use --reranker.
    with pytest.raises(_argparse.ArgumentTypeError) as plain:
        _rerank_depth("ten")
    assert "--reranker" not in str(plain.value)


def test_models_json_with_no_model_is_empty_and_missing(tmp_path, capsys):
    """An index with no embedding model: valid JSON, empty list, exit 1 (a miss),
    the same as the human path returning non-zero."""
    from dyprys.cli import main

    book = tmp_path / "b.txt"
    book.write_text("Para about neurons. " * 200, encoding="utf-8")
    data = str(tmp_path / "ix")
    main(["--data", data, "add", str(book)])

    code, models = _run_capturing(["--data", data, "models", "--json"], capsys)
    assert code == 1
    assert models == {"models": []}


def test_history_and_asked_emit_valid_json(tmp_path, capsys):
    """The two log commands round out the --json set; on a fresh index they are
    valid JSON with empty collections rather than a human 'nothing yet' message."""
    from dyprys.cli import main

    book = tmp_path / "b.txt"
    book.write_text("Para about neurons. " * 200, encoding="utf-8")
    data = str(tmp_path / "ix")
    main(["--data", data, "add", str(book)])

    code, hist = _run_capturing(["--data", data, "history", "--json"], capsys)
    assert code == 0
    assert hist["runs"] == []                                   # nothing embedded yet
    assert hist["totals"]["chunks"] == 0
    # ...but the add itself is a recorded index operation
    assert any(e["action"] == "add" for e in hist["events"])

    code, asked = _run_capturing(["--data", data, "asked", "--json"], capsys)
    assert code == 0
    assert asked == {"questions": []}
