"""The HTTP frontend: parse, call `service`, serialise. Nothing else.

The sibling of `cli.py`, and deliberately the same shape — every route below is
one call into `service` or `jobs` plus a serialisation, and the only thing it
knows that they do not is what a failure costs over HTTP.

Three things here are load-bearing rather than incidental.

**One thread per library.** `db.connect` does not pass `check_same_thread`, so
sqlite's default applies and a connection belongs to the thread that opened it;
`llama_cpp.Llama` is not re-entrant either. So each library gets a
`ThreadPoolExecutor(max_workers=1)` and every touch of that index goes through
it. That buys one connection and one warm embedder per library, owned by one
thread, with no locking subtleties — and a second simultaneous `ask` queues,
which is right, because the model is the bottleneck and running two at once
would only make both slower.

**`ask` is in-process, every mutation is a subprocess.** An earlier attempt at
a web layer shelled out to `dyp ... --json` per query, which reloads the model
each time and so pays the cost that dominates a search. Writes go the other
way, through `jobs`, because a run can last days and must outlive this process.

**The API reports; it does not interpret.** `cos`, `provenance`, `state`,
`scanned_fraction` and `warnings` pass through untouched, and `text` stays null
when a passage could not be proved against its stored hash. A serialiser that
helpfully coerced that null to `""` would erase the only signal saying the tool
cannot vouch for the words, and the page would then quote them.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Body, FastAPI, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from dyprys import db, errors, jobs, lock, service
from dyprys import progress as progress_mod
from dyprys.progress import Event
from dyprys.service import MAX_SPAN, SOURCE_SPAN  # noqa: F401  -- part of the surface

# Vite's default. Configurable because it is the one thing about this server
# that depends on what is talking to it.
DEFAULT_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


# --------------------------------------------------------------------------
# What a failure costs over HTTP
# --------------------------------------------------------------------------
#
# By isinstance rather than a dict keyed on the class, so a subclass added
# later inherits its parent's status instead of silently falling to 400.


def status_for(failed: Exception) -> int:
    if isinstance(failed, (errors.NoSuchLibrary, errors.NoSuchBook,
                           errors.SourceUnavailable)):
        return 404
    if isinstance(failed, errors.EmptyQuery):
        return 422
    if isinstance(failed, (errors.WeightsAbsent, errors.OptionalModelMissing,
                           errors.ModelUnavailable)):
        # 503, not 400: the request was fine and the server could do this —
        # the model it needs is not on the machine. A 200 carrying unreranked
        # results would be the worst of the three, because the caller asked
        # for a rescoring and would believe it happened.
        return 503
    if isinstance(failed, (lock.AlreadyRunning, errors.JobUnreachable)):
        return 409
    return 400


def as_error(failed: Exception) -> dict:
    """`kind` for a machine, `detail` for a person, `choices` for a picker.

    `choices` is why `ModelAmbiguous` carries data at all: a UI cannot render a
    dropdown from a sentence, and that error is the one a first-time user of a
    two-model index hits immediately.
    """
    return {
        "error": getattr(failed, "kind", "error"),
        "detail": getattr(failed, "message", str(failed)),
        "choices": list(getattr(failed, "choices", []) or []),
        "role": getattr(failed, "role", None),
    }


def _json(payload, status_code: int = 200) -> Response:
    """Serialised exactly as `cli._emit_json` does, `default=str` included.

    Not a convenience. `tests/test_api.py` asserts that this and `dyp X --json`
    return the same object for `status`, `check`, `books` and `models`, and the
    two share a payload builder — so the only way left for them to disagree is
    for one of them to coerce a value the other does not.
    """
    return Response(
        content=json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        media_type="application/json", status_code=status_code)


# --------------------------------------------------------------------------
# One worker thread and one session per library
# --------------------------------------------------------------------------


class Indexes:
    """The libraries this server has opened, and the thread each one lives on.

    Sessions are created lazily *on their own worker thread*, which is the
    whole point: a sqlite connection made on the event loop and used from a
    worker would raise, and one made on a worker and used from another worker
    would too.
    """

    def __init__(self, load_model=None, keep: int = 2):
        self._load_model = load_model
        self._keep = keep
        self._pools: dict[str, ThreadPoolExecutor] = {}
        self._sessions: dict[str, service.Session] = {}
        self._guard = threading.Lock()

    def _pool(self, name: str) -> ThreadPoolExecutor:
        with self._guard:
            if name not in self._pools:
                self._pools[name] = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"dyp-{name}")
            return self._pools[name]

    def _session(self, name: str) -> service.Session:
        """Called only from that library's worker thread."""
        if name not in self._sessions:
            directory = service.resolve_index(library=name, env=os.environ)
            if not db.index_exists(directory):
                raise errors.NoSuchLibrary(
                    f"no index at {directory} for library {name!r}; "
                    f"`dyp add` creates one.")
            self._sessions[name] = service.Session(
                directory, db.connect(directory),
                load_model=self._load_model, keep=self._keep)
        return self._sessions[name]

    def submit(self, name: str, work):
        """Run `work(session)` on this library's thread. Returns a Future."""
        return self._pool(name).submit(lambda: work(self._session(name)))

    async def run(self, name: str, work):
        return await asyncio.wrap_future(self.submit(name, work))

    def forget(self, name: str) -> None:
        """Drop a library's warm session, because its name stopped meaning that.

        A registry edit changes what a name resolves to; the session cached
        under it still holds a connection to the old directory and a model
        loaded out of it, and would go on answering from there.
        """
        with self._guard:
            pool = self._pools.pop(name, None)
        if pool is None:
            return
        session = self._sessions.pop(name, None)
        if session is not None:
            pool.submit(session.close).result(timeout=10)
        pool.shutdown(wait=False)

    @property
    def loaded(self) -> dict[str, list[str]]:
        return {name: session.loaded for name, session in self._sessions.items()}

    def close(self) -> None:
        for name, pool in self._pools.items():
            session = self._sessions.pop(name, None)
            if session is not None:
                pool.submit(session.close).result(timeout=10)
            pool.shutdown(wait=True)
        self._pools.clear()


# --------------------------------------------------------------------------
# The request bodies
# --------------------------------------------------------------------------


def _options_from(body: dict) -> service.SearchOptions:
    """A request body as the same dataclass argparse fills.

    `expand` and `summarise` are `bool | str` and must stay that way: argparse
    declares them `nargs="?", const=True`, so bare means "whatever this library
    remembers" and a value names a model. A field typed `str | None` would
    silently reject the bare form the CLI has always accepted, and the two
    frontends would drift on their very first flag.
    """
    known = {f.name for f in dataclasses.fields(service.SearchOptions)}
    unknown = set(body) - known - {"question"}
    if unknown:
        raise errors.BadRequest(
            f"unknown option(s): {', '.join(sorted(unknown))}. "
            f"The search takes: {', '.join(sorted(known))}.")
    return service.SearchOptions(
        **{name: value for name, value in body.items() if name in known})


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------


def create_app(load_model=None, origins=None, web=None, keep: int = 2) -> FastAPI:
    """Build the app. Not a module-level singleton, on purpose.

    `load_model` is the same injectable `service.open_session` takes, so the
    test suite drives the whole HTTP surface against a stub embedder and never
    needs a GGUF on the machine.
    """
    from contextlib import asynccontextmanager

    from fastapi.middleware.cors import CORSMiddleware

    indexes = Indexes(load_model=load_model, keep=keep)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        # Each session's connection belongs to its own worker thread, so it is
        # closed from there rather than here.
        indexes.close()

    app = FastAPI(title="dyprys", version="0.1.0", lifespan=lifespan,
                  description="Local semantic search over a personal library.")
    app.state.indexes = indexes

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(origins or DEFAULT_ORIGINS),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def failed(request: Request, exc: Exception) -> Response:
        return JSONResponse(status_code=status_for(exc), content=as_error(exc))

    app.add_exception_handler(errors.DyprysError, failed)
    # Not a DyprysError -- `lock` predates the typed errors and is imported by
    # modules that must not depend on them -- so it needs its own handler.
    app.add_exception_handler(lock.AlreadyRunning, failed)

    # --- the server itself ------------------------------------------------

    @app.get("/api/health")
    async def health():
        return {"ok": True, "loaded": indexes.loaded,
                "libraries": [row["name"]
                              for row in service.libraries_payload()["libraries"]]}

    @app.get("/api/libraries")
    async def libraries():
        return _json(service.libraries_payload())

    @app.post("/api/libraries", status_code=201)
    async def register(body: dict = Body(default_factory=dict)):
        """Give a directory on *this machine* a name.

        The path is server-side text, not an upload: a browser cannot pick a
        directory, and this server is loopback-only and already reads the whole
        filesystem through `dyp add`. Registering creates nothing — it is the
        registry entry alone, so a wrong path costs a DELETE.
        """
        unknown = set(body) - {"name", "path", "default"}
        if unknown:
            raise errors.BadRequest(
                f"unknown field(s): {', '.join(sorted(unknown))}. "
                f"Registering a library takes: name, path, default.")
        return _json(service.register_library(
            body.get("name", ""), body.get("path", ""), body.get("default")), 201)

    @app.delete("/api/libraries/{name}")
    async def forget(name: str):
        """Forget a name. Nothing on disk is touched.

        The CLI's `--delete` — which erases the index directory — deliberately
        has no route. Over HTTP that is one misclick from days of embedding,
        and a terminal is the right place to confirm it.
        """
        payload = service.forget_library(name)
        indexes.forget(name)
        return _json(payload)

    @app.post("/api/libraries/{name}/default")
    async def make_default(name: str):
        """Which library a bare `dyp` command means. Shared with the terminal."""
        return _json(service.default_library(name))

    # --- reading one library ----------------------------------------------

    @app.get("/api/libraries/{name}/status")
    async def status(name: str):
        return _json(await indexes.run(
            name, lambda s: service.status_payload(s.conn, s.directory)))

    @app.get("/api/libraries/{name}/check")
    async def check(name: str, deep: bool = False):
        return _json(await indexes.run(
            name, lambda s: service.check_payload(s.conn, deep)))

    @app.get("/api/libraries/{name}/books")
    async def books(name: str, pattern: str | None = None):
        return _json(await indexes.run(
            name, lambda s: service.books_payload(s.conn, pattern)))

    @app.get("/api/libraries/{name}/models")
    async def models(name: str):
        return _json(await indexes.run(
            name, lambda s: service.models_payload(s.conn, s.directory)))

    @app.post("/api/libraries/{name}/models/{model}/alias")
    async def name_model(name: str, model: str, body: dict = Body(default_factory=dict)):
        """A short name to type instead of a weights handle.

        Stored in the index, not in this client, so an alias chosen here is one
        `dyp --model` accepts in a terminal.
        """
        unknown = set(body) - {"alias"}
        if unknown:
            raise errors.BadRequest(
                f"unknown field(s): {', '.join(sorted(unknown))}. Takes: alias.")
        return _json(await indexes.run(
            name, lambda s: service.name_model(s.conn, model, body.get("alias", ""))))

    @app.get("/api/libraries/{name}/models/available")
    async def available_models(name: str):
        """The .gguf files on this machine, so a reranker can be chosen.

        Where to look is the frontend's question and is answered here rather
        than in `service`: the directories of the weights this index already
        remembers (the likeliest home, and needing no configuration),
        `$DYPRYS_MODEL_DIR` for anywhere else, and the cache `dyp` documents.

        Reading the environment is allowed here and nowhere below. A core module
        that resolved a model directory itself would answer according to the
        shell that launched the server rather than the request that arrived.
        """
        def look(session):
            known = service.models_payload(session.conn, session.directory)
            directories = [str(Path(m["file_path"]).parent)
                           for m in known["models"] if m["file_path"]]
            configured = os.environ.get("DYPRYS_MODEL_DIR", "")
            directories += [part for part in configured.split(os.pathsep) if part]
            directories.append(str(Path.home() / ".cache" / "qmd" / "models"))
            return service.available_models(dict.fromkeys(directories))

        return _json(await indexes.run(name, look))

    @app.get("/api/libraries/{name}/history")
    async def history(name: str, limit: int = 20):
        return _json(await indexes.run(
            name, lambda s: service.history_payload(s.conn, limit)))

    @app.get("/api/libraries/{name}/asked")
    async def asked(name: str, limit: int = 15, find: str | None = None,
                    which: int | None = None):
        payload = await indexes.run(
            name, lambda s: service.asked_payload(s.conn, limit, find, which))
        if payload is None:
            raise errors.NoSuchBook(f"no question numbered {which} in this library")
        return _json(payload)

    @app.get("/api/libraries/{name}/notes", response_class=PlainTextResponse)
    async def notes(name: str):
        """The library's NOTES.md, served raw and never interpreted.

        CLAUDE.md tells a reader to read the notes before searching, because a
        corpus carries facts the index cannot: which shelf `-c` cuts along
        cleanly, two subjects whose vocabulary overlaps, a title that means two
        books. The API's whole job here is to hand them over — corpus knowledge
        stays data, and this server stays corpus-independent.
        """
        from dyprys.library import notes_path

        path = await indexes.run(name, lambda s: notes_path(s.directory))
        if path is None:
            raise errors.NoSuchBook(f"library {name!r} has no NOTES.md")
        return PlainTextResponse(Path(path).read_text(encoding="utf-8"),
                                 media_type="text/markdown; charset=utf-8")

    # --- searching --------------------------------------------------------

    def _search(session, body, progress):
        result = service.search(session, body.get("question", ""),
                                _options_from(body), progress)
        payload = service.results_as_json(
            result.question, result.mode, result.routed, result.scanned_fraction,
            result.elapsed_ms, result.passages, result.why, result.cosine)
        # Rides on the payload because over HTTP there is no stderr to print it
        # to: an unreachable book is the one failure a reader cannot detect from
        # the output, since the response is a full `k` results at 200.
        payload["warnings"] = [{"kind": w.kind, "message": w.message}
                               for w in result.warnings]
        payload["answer"] = result.answer.as_record() if result.answer else None
        payload["expansion"] = result.expansion
        payload["models"] = result.models
        return payload

    @app.post("/api/libraries/{name}/ask")
    async def ask(name: str, body: dict = Body(default_factory=dict)):
        payload = await indexes.run(
            name, lambda s: _search(s, body, progress_mod.Silent()))
        return _json(payload)

    @app.post("/api/libraries/{name}/ask/stream")
    async def ask_stream(name: str, body: dict = Body(default_factory=dict)):
        """NDJSON: any number of stage frames, then exactly one final frame.

        The final frame is `{"result": ...}` or `{"error": ...}`. An error after
        the headers are sent cannot be a status code, and a stream that just
        stops is indistinguishable to a UI from one still running — which is how
        a dropped exception inside a generator presents itself.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        say = progress_mod.Queue(queue, loop)

        def work(session):
            try:
                last = {"result": _search(session, body, say)}
            except Exception as failure:   # including a DyprysError
                last = as_error(failure)
            loop.call_soon_threadsafe(queue.put_nowait, last)

        indexes.submit(name, work)

        async def frames():
            # `finally` rather than a disconnect poll: when the client goes
            # away, Starlette closes this generator, which is the one moment we
            # are told. Marking the search abandoned lets it stop at its next
            # checkpoint — it cannot be killed, because it is a worker thread —
            # and that matters because each library has exactly one such thread.
            # A rescoring nobody is waiting for otherwise holds it for half a
            # minute while the next question queues behind a dead tab.
            try:
                while True:
                    item = await queue.get()
                    if isinstance(item, Event):
                        yield json.dumps({"stage": {
                            "kind": item.kind, "label": item.label,
                            "detail": item.detail, "indent": item.indent}}) + "\n"
                        continue
                    yield json.dumps(item, ensure_ascii=False, default=str) + "\n"
                    return
            finally:
                say.abandon()

        return StreamingResponse(frames(), media_type="application/x-ndjson")

    @app.post("/api/libraries/{name}/warm")
    async def warm(name: str, body: dict = Body(default_factory=dict)):
        """Load a model before the first question, so it does not pay for it."""
        def load(session):
            session.model(body.get("model"))
            return session.loaded

        return {"loaded": await indexes.run(name, load)}

    # --- the reader pane --------------------------------------------------

    @app.get("/api/libraries/{name}/source")
    async def source(name: str, path: str = Query(...), offset: int = 0,
                     span: int = SOURCE_SPAN):
        """Bytes around an offset, for a path this index owns and no other.

        `path` arrives from a query string, so this is a file-read primitive
        until the `sources` table bounds it — and a prefix check is not
        equivalent: symlinks, `..` and a second library's files all pass one.
        The span is capped rather than refused, because a UI asking for too much
        should get what it may have.
        """
        text, at = await indexes.run(
            name, lambda s: service.source_window(s, path, offset, span))
        return {"path": path, "offset": at, "text": text}

    # --- the long mutations -----------------------------------------------

    @app.get("/api/libraries/{name}/jobs")
    async def job_states(name: str):
        return _json(await indexes.run(
            name, lambda s: jobs.states(s.conn, s.directory)))

    @app.get("/api/libraries/{name}/jobs/{kind}")
    async def job_state(name: str, kind: str, tail: int = 40):
        def look(session):
            if kind not in jobs.KINDS:
                raise errors.NoSuchJob(f"no such job {kind!r}")
            answer = jobs.state(session.conn, session.directory, kind)
            answer["log_tail"] = jobs.log_tail(session.directory, kind, tail)
            return answer

        return _json(await indexes.run(name, look))

    @app.post("/api/libraries/{name}/jobs/{kind}")
    async def job_start(name: str, kind: str, body: dict = Body(default_factory=dict)):
        """Start `dyp KIND` as a detached subprocess against this index.

        409 while something already holds the lock, and the refusal happens
        here rather than in the child: `dyp embed` loads its GGUF *before* it
        locks, so letting the subprocess discover the clash would show a job
        that starts and then dies a minute later for no visible reason.
        """
        ref = await indexes.run(
            name, lambda s: jobs.start(s.directory, kind, body))
        return _json({"kind": ref.kind, "pid": ref.pid, "log": str(ref.log),
                      "started_at": ref.started_at}, status_code=202)

    @app.delete("/api/libraries/{name}/jobs/{kind}")
    async def job_stop(name: str, kind: str, force: bool = False):
        """Ask a run to stop. Safe because embedding is resumable.

        SIGINT, which `dyp embed` handles by finishing the batch in flight and
        committing it — so nothing is lost and the next run picks up where this
        one left off. `?force=true` escalates to SIGTERM.
        """
        return _json(await indexes.run(
            name, lambda s: jobs.stop(s.directory, kind, force)))

    if web:
        from fastapi.staticfiles import StaticFiles

        # Mounted last, so it cannot shadow a route above. `html=True` serves
        # index.html for unknown paths, which is what a client-side router needs.
        app.mount("/", StaticFiles(directory=str(web), html=True), name="web")
    return app
