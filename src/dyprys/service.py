"""Everything `dyp ask` does, said as values rather than as terminal output.

This is the layer the CLI and the HTTP API both sit on. It exists because the
retrieval pipeline was the one part of dyprys that was not already a library:
`check`, `library`, `search`, `lexical`, `routing`, `rerank`, `expand`,
`summarise` and `embed` all compute and return, while `ask` computed, printed,
and returned an exit code. A web server cannot use a function that answers by
printing, which is why `tests/test_expand.py` and `tests/test_cli.py` had
already resorted to importing private names out of `cli`.

Nothing here is new behaviour. The bodies moved with their comments attached and
their measurements intact; what changed is the three ways they used to talk to
the outside:

  * `print(...)` on failure became `raise` -- see `dyprys.errors`;
  * `print(...)` on progress became `Progress.note` -- see `dyprys.progress`;
  * three module-level dicts that a search filled and the recorder read back
    became fields on the value the search returns. That last one is not tidying.
    Those dicts are shared by the module, not by the connection, so the moment
    two searches run at once -- which is the entire premise of a server -- the
    loser's expansion is recorded against the winner's question, silently, at
    exit 0. `SearchResult` carries its own.

The frontends keep three jobs, all of which are theirs: reading the environment
(a model named by `$DYPRYS_*` is per-shell in a terminal and per-request over
HTTP, so a function that reads it can only ever serve one caller), turning an
error into an exit code or a status code, and formatting. Nothing below decides
any of those.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from dyprys import db, errors
from dyprys import text as text_mod
from dyprys.progress import Progress, Silent
from dyprys.term import handle as _handle
from dyprys.term import shorten as _shorten
from dyprys.term import size as _size

# How much of a book the reader pane may pull in one request. Uncapped, `?span=`
# turns one call into "send me this whole book", which is a different product
# and a much larger response than anything else the API returns.
SOURCE_SPAN = 4_000
MAX_SPAN = 200_000

# Whether a rewritten query is also given to BM25. It is not, and the reason is
# the same one that already sends an exact phrase past the router: a phrase is a
# lookup, not a ranking, and a lookup cannot be improved by rewording the thing
# being looked up. Measured -- see the docstring of `dyprys.expand`.
EXPAND_LEXICAL = False


@dataclass(frozen=True)
class Advisory:
    """Something the caller should know that did not stop the search.

    CLAUDE.md names the case this exists for: a book embedded since the last
    `dyp route` has no profile, so `--route` cannot return it at any rank -- and
    the search still returns a full `k` and exits 0, so nothing in the output
    says part of the library was skipped. On a terminal `cli._router` printed
    that. Over HTTP a print is a warning that does not exist, so it rides on the
    payload instead.
    """

    kind: str
    message: str


# --------------------------------------------------------------------------
# Where the index is
# --------------------------------------------------------------------------


def resolve_index(library: str | None = None, data=None, env: dict | None = None) -> Path:
    """Which index a command acts on.

    An explicit path wins, then a name, then whichever library was made the
    default, and only then the ./data fallback. Explicit always beats
    remembered, so a stale default can never silently redirect a command that
    named its target.

    `env` is an argument rather than a read of `os.environ`, which is the whole
    difference between a CLI helper and something a server can call: one process
    serves many requests, and the shell that launched it does not get a vote in
    any of them. The CLI passes `os.environ` and behaves exactly as before.
    """
    from dyprys import registry

    env = env or {}
    if data:
        return Path(data)
    if library:
        found = registry.resolve(library)
        if found is None:
            if not registry.readable():
                raise errors.RegistryUnreadable(
                    f"the registry at {registry.registry_path()} could not be read, "
                    f"so no name resolves. Open the index directly with --data DIR.")
            raise errors.NoSuchLibrary(
                f"no library named {library!r}; try `dyp library list`")
        return found
    if not env.get("DYPRYS_DATA"):
        default = registry.resolve(None)
        if default is not None:
            return default
    return Path(env["DYPRYS_DATA"]) if env.get("DYPRYS_DATA") else db.DEFAULT_DATA_DIR


# --------------------------------------------------------------------------
# Which model
# --------------------------------------------------------------------------


def default_model(conn, role: str) -> str | None:
    """The model remembered for `role` in this index, if any."""
    return db.get_meta(conn, f"model.{role}") if conn is not None else None


def _weights_for(conn, model_arg) -> str:
    """Where to load weights from: a path, or a remembered model.

    The index already records the file name, size and digest of the weights that
    made its vectors, and now where they were last opened. So the common case --
    one model, still on this machine -- needs no argument at all.

    `model_arg` is already resolved: a path, a name, or None. Whether that came
    from a flag, an environment variable or a request body is the frontend's
    business, and reading the environment here would mean a server answered
    every request with whatever model the shell that started it named -- which
    is why the variables that pick a model are not named anywhere below the
    frontends, not even in a message. `tests/test_layering.py` holds that.
    """
    given = model_arg
    if given and Path(given).exists():
        return given

    row = db.find_model(conn, given) if given else db.sole_model(conn)
    if row is None:
        if given:
            raise errors.ModelMissing(
                f"no model matches {given!r}, and it is not a file. "
                f"`dyp models` lists what this index knows.")
        rows = conn.execute("SELECT name, alias FROM models ORDER BY id").fetchall()
        if len(rows) > 1:
            # Listing them here costs one query we have already made and saves
            # the round trip through `dyp models` -- and any unique substring
            # of a name works, so the whole name never has to be typed.
            handles = [_handle(r["name"], r["alias"]) for r in rows]
            choices = "\n".join(f"  --model {h!r}" for h in handles)
            raise errors.ModelAmbiguous(
                f"this index has {len(rows)} models — say which with "
                f"--model:\n{choices}\n"
                f"any unique part of a name works; "
                f"`dyp models --name NAME ALIAS` sets a short one.",
                choices=handles)
        raise errors.ModelMissing(
            "no model yet: pass --model PATH.gguf. "
            "It is remembered after the first run.", role="model")

    remembered = row["file_path"]
    if remembered and Path(remembered).exists():
        # A hint, not a promise. Size is the cheap half of the check; the digest
        # is the real one and happens on load, where wrong weights register as a
        # different model rather than quietly polluting this one.
        size = Path(remembered).stat().st_size
        if row["file_bytes"] and size != row["file_bytes"]:
            raise errors.WeightsMismatch(
                f"the weights remembered at {remembered} are "
                f"{_size(size)}, not the {_size(row['file_bytes'])} that "
                f"made these vectors. Pass --model explicitly.")
        return remembered
    where = f" (last seen at {remembered})" if remembered else ""
    raise errors.WeightsAbsent(
        f"this index needs {row['file_name'] or row['name']}{where}, "
        f"which is not on this machine now.\n"
        f"pass --model PATH.gguf; `dyp models --needed` shows how to "
        f"verify a candidate.")


def load_model(conn, directory, model_arg):
    """Open the named model and its vector file. Returns `(embedder, id, store)`."""
    path = _weights_for(conn, model_arg)
    from dyprys.embed import store_for
    from dyprys.embedder import Embedder

    embedder = Embedder(path)
    # An index whose vectors were truncated needs its queries truncated to
    # match. Without this the model row says 512, the embedder says 768, and
    # every command refuses to open the index it just shrank.
    known = conn.execute("SELECT dim FROM models WHERE name = ?",
                         (embedder.name,)).fetchone()
    if known and known["dim"] < embedder.dim:
        from dyprys.embedder import Truncated

        embedder = Truncated(embedder, known["dim"])
    model_id = db.model_id(
        conn, embedder.name, embedder.dim, provenance=embedder.provenance()
    )
    db.remember_weights(conn, model_id, Path(path).resolve())
    return embedder, model_id, store_for(conn, directory, model_id, embedder.dim)


# --------------------------------------------------------------------------
# A session: one index, one connection, models kept warm
# --------------------------------------------------------------------------


class Session:
    """One open index, and whatever models have been loaded against it.

    The cache is the reason the API exists in this shape at all. An earlier
    attempt at a web layer shelled out to `dyp ... --json` per query, which
    reloads a 300 MB - 1 GB GGUF every time and so pays the model load that
    dominates a search's latency. Holding the embedder between requests is the
    whole win; everything else here is bookkeeping around it.

    A session is not thread-safe and is not meant to be. `db.connect` does not
    pass `check_same_thread`, so the connection belongs to the thread that
    opened it, and `llama_cpp.Llama` is not re-entrant either -- so a server
    gives each library one worker thread and one session, and a second
    simultaneous search queues. That is correct rather than a limitation: the
    model is the bottleneck, so running two at once would only make both slower.
    """

    def __init__(self, directory: Path, conn, *, load_model=None, keep: int = 2):
        self.directory = Path(directory)
        self.conn = conn
        # Injectable so tests can hand back a stub and never need a GGUF on the
        # machine -- the trick the existing suite already relies on.
        self._load = load_model or globals()["load_model"]
        # Insertion-ordered, and re-inserted on use, so the first key is the
        # least recently used. A GGUF is 300 MB to 1 GB resident, so this is a
        # memory budget rather than a cache policy -- a server left running
        # against a many-model index would otherwise hold all of them.
        self._models: dict[str, tuple] = {}
        # The same budget for cross-encoders, kept separately because they are a
        # different kind of model with a different key -- a filesystem path
        # rather than an index handle -- and because evicting an embedder to
        # make room for a reranker would trade the load a search always pays
        # for the one it sometimes does.
        self._rerankers: dict[str, object] = {}
        self.keep = max(1, keep)

    def model(self, name: str | None = None):
        """`(embedder, model_id, store)` for `name`, loading it once."""
        key = name or ""
        if key in self._models:
            self._models[key] = self._models.pop(key)
        else:
            self._models[key] = self._load(self.conn, self.directory, name)
            while len(self._models) > self.keep:
                self._models.pop(next(iter(self._models)))
        return self._models[key]

    def reranker(self, path, load):
        """The cross-encoder at `path`, loaded once and kept.

        `_reranker_for` has always accepted an already-loaded object -- its
        comment names "a warm server cache" -- and nothing ever filled one, so
        every reranked request read a 600 MB GGUF off disk. Measured on `neuro`:
        three identical reranked searches took 30.4s, 31.2s and 31.5s, nearly
        all of it the load. Holding the embedder between requests is the whole
        reason this class exists; the reranker was outside it by omission, and
        it is the model CLAUDE.md tells a reader to reach for.
        """
        key = str(path)
        if key in self._rerankers:
            self._rerankers[key] = self._rerankers.pop(key)
        else:
            self._rerankers[key] = load()
            while len(self._rerankers) > self.keep:
                self._rerankers.pop(next(iter(self._rerankers)))
        return self._rerankers[key]

    @property
    def loaded(self) -> list[str]:
        """Which models this session is holding in memory, for a health check."""
        return ([name or "(default)" for name in self._models]
                + [Path(path).name for path in self._rerankers])

    def close(self) -> None:
        self.conn.close()


@contextmanager
def open_session(library: str | None = None, data=None, *, load_model=None,
                 env: dict | None = None):
    """Open the index `library`/`data` names, and close it afterwards."""
    directory = resolve_index(library, data, env)
    session = Session(directory, db.connect(directory), load_model=load_model)
    try:
        yield session
    finally:
        session.close()


# --------------------------------------------------------------------------
# What a search was asked for, and what it found
# --------------------------------------------------------------------------


@dataclass
class SearchOptions:
    """Every knob `dyp ask` has, defined once.

    argparse and a request model both mirror this, so the two frontends cannot
    drift on a flag. Two of the fields are deliberately `bool | str`, exactly as
    argparse declares them with `nargs="?", const=True`: `--expand` alone means
    "use the remembered default" and `--expand gemma3:4b` names a model. A field
    typed `str | None` silently rejects the bare form the CLI has always
    accepted.

    `expander`, `reranker` and `summariser` also accept an already-constructed
    object, which is how a server reuses one warm model across requests and how
    a test supplies a stub.
    """

    k: int = 5
    mode: str = "hybrid"
    model: str | None = None
    collection: str | list[str] | None = None
    route: int = 0
    rerank: int = 0
    depth: int = 0
    reranker: object | None = None
    expand: bool | str | None = None
    expander: object | None = None
    summarise: bool | str | None = None
    summariser: object | None = None
    dedupe: bool = False
    # Presentation only, carried so one dataclass describes the whole command.
    full: bool = False
    # Which prompt an ollama expander is given. Config, not a model choice.
    prompt: str = "full"


@dataclass
class SearchResult:
    """Everything a frontend needs to render a search, and nothing it must fetch.

    `why` and `cosine` are keyed by chunk id. `expansion` and `answer` are the
    two that used to live in module-level dicts; carrying them here is what
    makes two concurrent searches safe.
    """

    question: str
    mode: str
    passages: list
    why: dict
    cosine: dict
    routed: bool = False
    routed_books: set | None = None
    scanned_fraction: float = 0.0
    elapsed_ms: float = 0.0
    expansion: dict | None = None
    answer: object | None = None
    warnings: list = field(default_factory=list)
    # True when the index holds text but no vectors -- the difference between
    # "nothing matched" and "nothing has been embedded yet", which are different
    # problems with different fixes.
    nothing_embedded: bool = False
    models: dict = field(default_factory=dict)


@dataclass
class Summary:
    """A drafted answer, the passages it was drawn from, and what checked out.

    `passages` is not always the passages the search returned: when the first
    attempt reports that nothing answers the question, the library is searched
    again with the model's own rephrasing, and a citation has to name something
    the reader can actually see.
    """

    model: str
    answer: object
    passages: list
    widened: list
    spans: dict
    retried: str | None = None
    failed: str | None = None

    def as_record(self) -> dict:
        """The shape `dyp asked` stores, so a good answer can be found again.

        `retried` and `refused` are the two things the terminal prints and a
        record without them cannot say. A refusal that survived a rephrasing is
        strong evidence the library lacks the answer; one that was never
        rephrased is not, and stored prose reading "NO ANSWER IN PASSAGES" looks
        identical either way.

        `drawn_from` appears only after a successful retry, and then it matters
        a great deal: the answer is about the *retry's* passages, not the ones
        the search returned, and `cited` indexes into these. Without it a reader
        is shown an answer citing passages that are not on the screen. Omitted
        otherwise because it would only repeat the results beside it.
        """
        record = {
            "prose": self.answer.prose.strip(),
            "verified": [{"cited": c.cited, "quote": c.quote, "where": c.location}
                         for c in self.answer.verified],
            "rejected": [{"cited": c.cited, "quote": c.quote}
                         for c in self.answer.rejected],
            "retried": self.retried,
            "refused": self.failed,
        }
        if self.retried:
            record["drawn_from"] = [
                {"chunk_id": p.chunk_id, "book": p.title, "path": str(p.path),
                 "offset": p.offset}
                for p in self.passages]
        return record


# --------------------------------------------------------------------------
# The optional models
# --------------------------------------------------------------------------
#
# Each takes a value that is already resolved -- a path, a name, or an object
# somebody else loaded. Where that value came from (a flag, `$DYPRYS_*`, a
# request body, the index's remembered default) is settled above this line,
# because a function that consults the environment itself can only ever serve
# the one process it is running in.


def _reranker_for(conn, options: SearchOptions, progress: Progress, session=None):
    """The cross-encoder, or None when reranking was not asked for.

    The reranker is the one model the index does not remember, so this is the
    refusal users actually hit. It is a refusal and not a fallback on purpose:
    unreranked results returned to a caller who believes they were rescored is
    the worst of the three outcomes, because nothing about them says so.

    `session` is where a loaded one is kept between requests. Optional because
    the CLI has no session and no second request to keep it for -- it loads,
    answers, and exits -- while a server that reloaded it every time would pay
    a 600 MB read per search, which is more than the search itself costs.
    """
    if not options.rerank:
        return None
    given = options.reranker
    if given is not None and not isinstance(given, (str, Path)):
        return given  # already loaded -- a warm server cache, or a test's stub
    path = given or default_model(conn, "reranker")
    if not path or not Path(path).exists():
        raise errors.OptionalModelMissing(
            "no reranker model. Pass one: `--reranker PATH.gguf`. "
            "It needs a cross-encoder .gguf, not a chat model.", role="reranker")
    from dyprys.rerank import Reranker

    def load():
        progress.working(f"loading {Path(path).name} …", kind="loading")
        return Reranker(path)

    return session.reranker(path, load) if session is not None else load()


def _expander_for(conn, options: SearchOptions):
    """The expansion model, or None when expansion was not asked for.

    `--expand gemma3:4b` names the model inline, as `--summarise` does;
    `--expander` stays as the older spelling.
    """
    asked = options.expand
    if not asked:
        return None
    given = options.expander
    if given is not None and not isinstance(given, (str, Path)):
        return given
    named = (asked if isinstance(asked, str) else None) or given \
        or default_model(conn, "expander")
    if not named:
        raise errors.OptionalModelMissing(
            "no expansion model. Pass one: `--expand MODEL`.", role="expander")
    named = str(named)
    # A path to weights we load ourselves; anything else is a name for the local
    # ollama server, which owns each model's chat template. Guessing a template
    # is how the reranker was made worse than no reranker at all.
    if named.endswith(".gguf") or Path(named).exists():
        from dyprys.expand import Expander

        return Expander(named)
    from dyprys.expand import OllamaExpander

    expander = OllamaExpander(named, prompt=options.prompt)
    problem = expander.unavailable()
    if problem:
        # Its own words, not a list of alternatives: "cannot reach ollama at
        # http://localhost:11434" names the fix, and replacing it with the
        # models ollama has installed would be answering a different question.
        raise errors.ModelUnavailable(problem, role="expander")
    return expander


def _summariser_for(conn, options: SearchOptions) -> str | None:
    """Which model drafts the answer, or None when none was asked for."""
    if not options.summarise:
        return None
    named = (options.summarise if isinstance(options.summarise, str) else None) \
        or options.summariser or default_model(conn, "summariser")
    if not named:
        raise errors.OptionalModelMissing(
            "no summariser model. Pass one: `--summarise MODEL`.", role="summariser")
    return str(named)


def _pipeline(conn, store, embedder, model_id, mode, books, router=None,
              reranker=None, depth=0, dedupe=False, expander=None, progress=None):
    """One callable per mode, so ask and eval cannot drift apart.

    `router` narrows the books per query, which is stage 1. It applies to the
    vector half and to BM25's OR-of-words ranking, but *not* to BM25's exact
    phrase attempt, which searches the library whole -- see `search_bm25`. So
    the reported scanned fraction is the share of the *vectors* read, and the
    output says so.
    """
    from dyprys.lexical import reciprocal_rank_fusion, search_bm25
    from dyprys.search import flat_search

    def run(question, k):
        # Expansion first, because everything below may be run several times
        # over. Routing still sees the question as asked: stage 1 picking books
        # from a rewritten query is a separate change with its own measurement,
        # and mixing the two would leave neither attributable.
        say = progress or Silent()
        note, working = say.note, say.working
        # Before the first expensive thing, and again before each one after it.
        # A search runs on the single worker thread its library has, so one
        # nobody is waiting for is one the next caller is queued behind.
        say.check()
        expansion = None
        if expander:
            working(f"rephrasing with {getattr(expander, 'model', 'a model')} …")
            expansion = expander.expand(question)
            # Attached to the callable, not stored in a module dict. The dict
            # version was read back at record time by whoever asked last,
            # which is the right answer for exactly one search at a time.
            run.expansion = {
                "vector": list(expansion.vector), "hyde": list(expansion.hyde),
                "lexical": list(expansion.lexical)}
            for kind, lines in (("as a question", expansion.vector),
                                ("as an answer", expansion.hyde),
                                ("as keywords", expansion.lexical)):
                for line in lines:
                    note(kind, line[:120], indent=1)
            if expansion.empty:
                note("nothing usable came back; searching as asked", indent=1)
        say.check()
        vector = embedder.embed_query(question)
        scope = router(question, vector) if router else books
        if router and scope:
            total = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
            titles = [r["title"] for r in conn.execute(
                "SELECT title FROM books WHERE id IN (%s)" % ",".join("?" * len(scope)),
                sorted(scope))]
            note(f"routed to {len(scope)} of {total:,} books")
            for title in titles:
                note(title[:76], indent=1)
        # Retrieval fetches a deeper shortlist when something downstream will
        # thin or reorder it; recall@depth of this list is the ceiling on what
        # reranking can reach.
        #
        # Depth is deliberately independent of the reranker. Tying the two
        # together made an arm change two things at once -- a reranked run
        # retrieved deeper *and* reordered -- and the two are separable: jina's
        # recall@1 is 63/110 at depth 5 and 67/110 at depth 20 with no reranker
        # at all, because RRF fuses however many candidates each half returned.
        # Crediting that +4 to the reranker overstated it, and did so in the
        # direction the measurement was hoping for.
        want = max(k, depth)
        # Over-fetch when a filter will thin the list, or it cannot return k.
        if dedupe:
            want = max(want, k * 4)
        # The query as asked leads each half, and leads the fused list, because
        # RRF breaks ties in favour of the earlier ranking. A rewrite is a guess
        # about what was meant; it may add an answer the original missed, but it
        # does not get to overrule the original on a tie.
        by_vector, by_words = [], []
        origin: dict[int, str] = {}
        if mode in ("vector", "hybrid"):
            by_vector.append(flat_search(conn, store, vector, model_id, want, book_ids=scope))
        if mode in ("lexical", "hybrid"):
            by_words.append(search_bm25(conn, question, want, model_id=model_id,
                                        book_ids=scope, origin=origin))

        if expansion and not expansion.empty:
            if mode in ("vector", "hybrid"):
                # A rewritten question is a query; a hypothetical answer is a
                # passage. EmbeddingGemma is trained with a different prefix for
                # each, and the model's own line kinds say which is which.
                for phrase in expansion.vector:
                    by_vector.append(flat_search(
                        conn, store, embedder.embed_query(phrase), model_id, want, book_ids=scope))
                for passage in expansion.hyde:
                    by_vector.append(flat_search(
                        conn, store, embedder.embed_documents([passage])[0],
                        model_id, want, book_ids=scope))
            if mode in ("lexical", "hybrid") and EXPAND_LEXICAL:
                for phrase in expansion.lexical:
                    by_words.append(search_bm25(
                        conn, phrase, want, model_id=model_id, book_ids=scope))

        # Each half is fused down to ONE ranking before the halves meet, so the
        # balance between them does not depend on how many phrasings the model
        # happened to emit for each. Handing fusion four vector lists and two
        # lexical ones is a 2:1 vector weight arrived at by accident, and a
        # deliberate 1.3x weight was already measured to cost 24 of 30
        # exact-phrase probes. The first version did exactly that, and lexical
        # safety fell 18/20 -> 13/20 on two separate question sets.
        def fused(rankings):
            rankings = [r for r in rankings if r]
            if not rankings:
                return []
            if len(rankings) == 1:
                return rankings[0]
            return reciprocal_rank_fusion(rankings, k=want)

        vector_side, lexical_side = fused(by_vector), fused(by_words)
        sides = [r for r in (vector_side, lexical_side) if r]
        shortlist = sides[0] if len(sides) == 1 else reciprocal_rank_fusion(sides, k=want)

        if reranker:
            say.check()
            working(f"rescoring {min(len(shortlist), want)} passages …")
            from dyprys.rerank import rerank
            shortlist = rerank(conn, reranker, question, shortlist[:want],
                               max(k * 4, k), say)
        if dedupe:
            from dyprys.search import drop_near_duplicates
            shortlist = drop_near_duplicates(conn, shortlist, k)
        found = shortlist[:k]
        # Attached rather than returned, so `eval` and every other caller keep
        # the same signature. Same pattern as `router.last`.
        from dyprys.lexical import provenance
        # "phrase" and "words" rather than one "bm25": an exact-phrase hit and a
        # keyword-overlap hit are different kinds of answer, and the reader is
        # the one who knows which they wanted.
        run.why = provenance(
            [n for n in (("vec", vector_side), ("bm25", lexical_side)) if n[1]],
            found, rename={"bm25": origin})
        # Cosine for *every* result, including the ones only BM25 found — k dot
        # products against vectors already on disk. A literal match with a low
        # cosine is worth seeing: it says the passage contains your words without
        # being about them, which is exactly when a keyword hit misleads.
        run.cosine = {}
        for hit in found:
            try:
                run.cosine[hit.chunk_id] = float(store.read(hit.chunk_id) @ vector)
            except (IndexError, ValueError):
                pass
        return found

    run.why, run.cosine, run.expansion = {}, {}, None
    return run


def _router(conn, directory, embedder, model_id, books_wanted, candidates):
    """Stage 1 as a callable, plus whatever it could not reach.

    Returns `(router, advisories)` -- `(None, [])` when routing was not asked
    for. The advisories are the half that used to be a print, and losing them
    is the failure CLAUDE.md singles out as undetectable from the output.
    """
    if not books_wanted:
        return None, []
    from dyprys.routing import is_built, profile_gap, route
    from dyprys.vectors import VectorStore

    if not is_built(conn, model_id):
        raise errors.NoRoutingProfile("no routing profile yet — run `dyp route`")

    # A book embedded since the last `dyp route` has no centroids, so stage 1
    # can never choose it. The search still returns k results and exits 0, so
    # without this line the omission is invisible -- the one failure mode a
    # reader has no way to detect from the output.
    advisories = []
    unprofiled = profile_gap(conn, model_id).unprofiled
    if unprofiled:
        advisories.append(Advisory(
            "unprofiled_books",
            f"--route cannot reach {unprofiled} book(s) embedded since the last "
            f"`dyp route`; run it to include them."))

    total = conn.execute(
        "SELECT COALESCE(SUM(centroid_count), 0) FROM book_centroids WHERE model_id = ?",
        (model_id,),
    ).fetchone()[0]
    centroids = VectorStore(db.centroids_path(directory, model_id), embedder.dim, total)
    last: dict = {}

    def go(question, vector):
        chosen = {b for b, _ in route(conn, centroids, vector, model_id, books_wanted, candidates)}
        last["books"] = chosen
        return chosen

    go.last = last
    return go, advisories


def results_as_json(question, mode, routed, scanned, elapsed_ms, passages, why, cosine):
    """The `--json` payload, built from resolved passages and nothing else.

    Kept a pure function so it can be tested without a model or an index: it turns
    what a search already produced into the shape an agent parses. `text` is null
    when the source could not be proved, and `state` says why; `provenance` is the
    tool's own rank signal ("vec 1 · phrase 1"), which carries more than a single
    fused score could.
    """
    results = []
    for rank, p in enumerate(passages, 1):
        results.append({
            "rank": rank,
            "chunk_id": p.chunk_id,
            "book": p.title,
            "chapter": (p.chapter + 1) if p.chapter else None,
            "path": str(p.path),
            "offset": p.offset,
            "cos": round(cosine[p.chunk_id], 4) if p.chunk_id in cosine else None,
            "provenance": why.get(p.chunk_id),
            "state": p.state,
            "text": p.text,
        })
    return {
        "query": question,
        "mode": mode,
        "routed": routed,
        "scanned_fraction": round(scanned, 4),
        "elapsed_ms": round(elapsed_ms, 1),
        "results": results,
    }


def _widen(passages):
    """Passages re-read with context, and where each chunk sits inside its window.

    A chunk boundary is a byte budget, so a passage handed to a model often
    begins mid-sentence and it copies the fragment. The span is kept so a quote
    can be told apart from one taken out of the margin either side.
    """
    import dataclasses

    widened, spans = [], {}
    for p in passages:
        if p.text is None:
            widened.append(p)
            continue
        length = len(p.text.encode())
        window = text_mod.read_window(p.path, p.offset, length)
        if not window:
            widened.append(p)
            continue
        text, at = window
        widened.append(dataclasses.replace(p, text=text, offset=at))
        head = len(text.encode()[: p.offset - at].decode("utf-8", errors="ignore"))
        spans[p.chunk_id] = (head, head + len(text[head:].encode()[:length]
                                              .decode("utf-8", errors="ignore")))
    return widened, spans

# --------------------------------------------------------------------------
# Drafting an answer from what was found
# --------------------------------------------------------------------------


def summarise(session: Session, question: str, passages: list, model: str, *,
              k: int = 5, run=None, progress: Progress | None = None,
              talk=None) -> Summary:
    """Draft an answer from the passages, and say which quotes checked out.

    Never instead of the passages. The passages are the result; this is a
    reading of them, and a reading that cannot be verified is worth less than
    the list it was drawn from -- so `Summary` carries the rejected claims too,
    and a frontend that hides them wastes the catch.
    """
    from dyprys import summarise as summarise_mod
    from dyprys.summarise import NO_ANSWER, ask_ollama

    say = progress or Silent()
    talk = talk or (lambda prompt: ask_ollama(model, prompt))

    # The model reads the same widened passages the reader sees, so a chunk cut
    # mid-sentence does not become a fragment it has to guess around. Verifying
    # against the widened text is still exact: it is the text that was shown,
    # read from the file at a recorded byte.
    widened, spans = _widen(passages)
    say.check()
    answer = summarise_mod.summarise(question, widened, talk)
    retried = failed = None

    # A refusal means the *search* failed, not that the library lacks an answer,
    # and a failed search can be tried with other words. This costs nothing when
    # the first attempt works, and only ever runs when the alternative is
    # nothing at all. One retry: a model that cannot find it twice is telling
    # you something, and a loop here would spend minutes proving it.
    if answer.prose.strip() == NO_ANSWER and run is not None:
        say.note("nothing here answers it — asking the same model for "
                 "other words")
        other = summarise_mod.rephrase(question, talk)
        if other:
            say.note("trying instead", other, indent=1)
            from dyprys.search import resolve

            again = resolve(session.conn, run(other, k))
            widened, spans = _widen(again)
            second = summarise_mod.summarise(question, widened, talk)
            if second.prose.strip() != NO_ANSWER:
                answer, passages, retried = second, again, other
            else:
                failed = other
                # The retry's passages are discarded, so the widened set must go
                # back to the ones on screen: a citation has to name something
                # the reader can see.
                widened, spans = _widen(passages)
    return Summary(model=model, answer=answer, passages=passages, widened=widened,
                   spans=spans, retried=retried, failed=failed)


# --------------------------------------------------------------------------
# The search itself
# --------------------------------------------------------------------------


def search(session: Session, question: str, options: SearchOptions | None = None,
           progress: Progress | None = None) -> SearchResult:
    """Ask this library a question and get back everything it learned.

    The whole of `dyp ask` except the printing. Two things it does that the CLI
    used not to:

      * it records the question itself, so `dyp asked` is cross-*interface*
        memory rather than cross-session. `cli._ask` returned from its `--json`
        branch before the recording call, which meant every agent-driven search
        -- the form CLAUDE.md prescribes -- was absent from the history. That is
        a deliberate change, and the hazard in the other direction is a caller
        that records again and doubles every terminal search;
      * it returns its expansion and its drafted answer instead of leaving them
        in module dicts for whoever asks next.
    """
    options = options or SearchOptions()
    say = progress or Silent()
    conn = session.conn

    # An empty or whitespace query embeds to a meaningless vector and matches no
    # words, so hybrid search returns whatever the vector half drifts to -- noise
    # presented as answers, at exit 0. Refuse it instead.
    question = (question or "").strip()
    if not question:
        raise errors.EmptyQuery("empty query — give something to search for")

    # Before the embedder, which is the first thing a search loads and is 300 MB
    # to 1 GB on a cold session. This is also the checkpoint that matters for a
    # request that was *queued*: it waited behind another search, and whoever
    # asked may well have given up during the wait.
    say.check()
    embedder, model_id, store = session.model(options.model)

    from dyprys.search import resolve, scanned_fraction, scope

    books = None
    if options.collection:
        books, missed = scope(conn, options.collection)
        # Every miss, not only the case where nothing matched at all. Under a
        # union a mistyped pattern contributes no books and changes no result,
        # so it would otherwise narrow the search silently -- and `-c` is a
        # promise about which books were ranked.
        if missed:
            named = ", ".join(repr(pattern) for pattern in missed)
            raise errors.NoSuchBook(
                f"no book matches {named}; try `dyp books`")

    router, warnings = _router(conn, session.directory, embedder, model_id,
                               options.route, books)
    for advisory in warnings:
        say.note(advisory.message, kind=advisory.kind)
    # Before loading a cross-encoder, which is the single longest thing a search
    # ever does -- reading a 600 MB GGUF off disk, and uninterruptible once
    # begun. Checking after it would be checking after the cost.
    say.check()
    reranker = _reranker_for(conn, options, say, session)
    expander = _expander_for(conn, options)
    summariser = _summariser_for(conn, options)

    started = time.time()
    run = _pipeline(conn, store, embedder, model_id, options.mode, books, router,
                    reranker, options.rerank or options.depth or 0, options.dedupe,
                    expander, say)
    hits = run(question, options.k)
    elapsed = (time.time() - started) * 1000
    # Copied off the callable *now*, before anything downstream can call it
    # again. The summariser's retry does exactly that -- it searches the library
    # a second time with the model's own rephrasing -- and every call rebinds
    # these three, so reading them later reports the retry's provenance against
    # the passages the caller was actually given.
    why, cosine, expansion = dict(run.why), dict(run.cosine), run.expansion
    reached = router.last.get("books") if router else books
    share = scanned_fraction(conn, model_id, reached)
    passages = resolve(conn, hits)

    # "Nothing matched" and "nothing has been embedded yet" are different
    # problems with different fixes, and vector search always returns something
    # -- so an empty list on an embedded index means a lexical-only mode found
    # no words, which is a real answer.
    nothing_embedded = False
    if not hits:
        nothing_embedded = not conn.execute(
            "SELECT COALESCE(SUM(n_embedded), 0) FROM segment_progress "
            "WHERE model_id = ?", (model_id,)).fetchone()[0]

    answer = None
    if summariser and passages:
        answer = summarise(session, question, passages, summariser,
                           k=options.k, run=run, progress=say)
        # Deliberately *not* `passages = answer.passages`. When the summariser's
        # first attempt reports that nothing answers the question, the library is
        # searched again with its own rephrasing -- but the result of `ask` is
        # still what `ask` found. The retry's passages belong to the answer, are
        # carried on it, and are shown under it.

    models = {name: value for name, value in (
        ("embed", _shorten(embedder.name)),
        ("expander", getattr(expander, "model", None) if expander else None),
        ("summariser", answer.model if answer else None),
    ) if value}

    # Kept for the person, not for the search. A question worth asking twice
    # should not have to be reconstructed from memory, and a good answer should
    # be findable again — `dyp asked`.
    db.record_question(conn, question, options.mode, elapsed, {
        "routed": bool(router),
        "books": len(reached) if reached else None,
        "models": models,
        "expansion": expansion or None,
        "hits": [{"chunk": p.chunk_id, "title": p.title, "path": p.path,
                  "offset": p.offset, "cos": round(cosine.get(p.chunk_id, 0.0), 3),
                  "why": why.get(p.chunk_id)} for p in passages],
        "answer": answer.as_record() if answer else None,
    })

    return SearchResult(
        question=question, mode=options.mode, passages=passages,
        why=why, cosine=cosine, routed=bool(router), routed_books=reached,
        scanned_fraction=share, elapsed_ms=elapsed, expansion=expansion,
        answer=answer, warnings=list(warnings), nothing_embedded=nothing_embedded,
        models=models)


# --------------------------------------------------------------------------
# The reader pane
# --------------------------------------------------------------------------


def source_window(session: Session, path, offset: int, span: int = SOURCE_SPAN):
    """Bytes around an offset in a book this index holds. `(text, offset)`.

    The path is checked against the `sources` table and nothing else. A prefix
    check is **not** equivalent and must not be substituted: symlinks, `..`, and
    a second library's files all pass one, and this is reached from a query
    string, so anything weaker is a file-read primitive with a nice name.

    `span` is capped for the same reason -- uncapped, one request becomes "send
    me this whole book".
    """
    wanted = str(path)
    row = session.conn.execute(
        "SELECT path FROM sources WHERE path = ?", (wanted,)).fetchone()
    if row is None:
        raise errors.NoSuchBook(
            f"this index does not hold {wanted!r}; `dyp books` lists what it has")
    span = max(1, min(int(span or SOURCE_SPAN), MAX_SPAN))
    window = text_mod.read_window(row["path"], max(0, int(offset)), span,
                                  before=0, after=0)
    if window is None:
        raise errors.SourceUnavailable(
            f"{wanted} is in this index and could not be read; run `dyp check`")
    return window


# --------------------------------------------------------------------------
# The inspection payloads
# --------------------------------------------------------------------------
#
# One builder per read-only command, returning exactly the dict `--json` has
# always printed. They live here rather than in `cli` so the CLI and the API
# cannot answer `status` differently -- which is the pair that can actually
# drift, since once `--json` *is* `_emit_json(status_payload(...))` the two
# have one implementation between them.
#
# Everything here is JSON-shaped but not JSON-clean: a `Path` may still be a
# `Path`. Both frontends serialise with `default=str`, so both render it the
# same way, and neither has to remember to coerce.


def status_payload(conn, directory=None) -> dict:
    """Totals, per-chunking counts, per-model coverage, and the notes file."""
    from dyprys.compact import interrupted
    from dyprys.library import notes_path

    books, sources, chunks, text_bytes = conn.execute(
        "SELECT (SELECT COUNT(*) FROM books), "
        "       (SELECT COUNT(*) FROM sources), "
        "       (SELECT COUNT(*) FROM chunks), "
        "       (SELECT COALESCE(SUM(size_bytes), 0) FROM sources)"
    ).fetchone()
    chunkings_rows = conn.execute(
        "SELECT ch.id, ch.target, ch.overlap, "
        "       COALESCE(SUM(seg.chunk_count), 0) AS n "
        "FROM chunkings ch LEFT JOIN segments seg ON seg.chunking_id = ch.id "
        "GROUP BY ch.id ORDER BY ch.id"
    ).fetchall()
    model_rows = conn.execute(
        "SELECT m.id, m.name, m.dim, m.chunking_id, "
        "       COALESCE(SUM(p.n_embedded), 0) AS done "
        "FROM models m LEFT JOIN segment_progress p ON p.model_id = m.id "
        "GROUP BY m.id ORDER BY m.id"
    ).fetchall()
    # Each model's progress belongs over the chunking it embeds, never over
    # every chunk in the library: the same divisor `dyp models` already uses.
    by_chunking = {c["id"]: c["n"] for c in chunkings_rows}
    live_for = {
        m["id"]: by_chunking.get(m["chunking_id"], chunks)
        if m["chunking_id"] is not None else chunks
        for m in model_rows
    }
    notes = notes_path(directory) if directory else None
    return {
        "books": books, "sources": sources, "chunks": chunks,
        "text_bytes": text_bytes,
        "chunkings": [
            {"id": c["id"], "target": c["target"], "overlap": c["overlap"],
             "chunks": c["n"]} for c in chunkings_rows],
        "models": [
            {"name": m["name"], "dim": m["dim"], "embedded": m["done"],
             "live_chunks": live_for[m["id"]],
             "coverage": (m["done"] / live_for[m["id"]])
             if live_for[m["id"]] else 0.0}
            for m in model_rows],
        "pending_carries": conn.execute(
            "SELECT COUNT(*) FROM chunk_carry").fetchone()[0],
        "failed_chunks": conn.execute(
            "SELECT COUNT(*) FROM chunk_failures").fetchone()[0],
        "compaction_interrupted": interrupted(conn),
        # Null unless the owner left one. Read it before searching: it holds
        # what the index cannot tell you about the corpus in it.
        "notes": str(notes) if notes else None,
    }


def check_payload(conn, deep: bool = False) -> dict:
    """What drifted and what is outstanding. Changes nothing."""
    from dyprys.check import survey
    from dyprys.routing import is_built, profile_gap

    report = survey(conn, deep=deep)

    def routing_for(name):
        row = conn.execute("SELECT id FROM models WHERE name = ?", (name,)).fetchone()
        if not row or not is_built(conn, row["id"]):
            return {"built": False, "stale_books": None,
                    "drifted_books": None, "unprofiled_books": None}
        gap = profile_gap(conn, row["id"])
        # `unprofiled` is the one worth branching on: those books cannot be
        # returned under --route at all, where a drifted one still can.
        return {"built": True, "stale_books": gap.total,
                "drifted_books": gap.drifted, "unprofiled_books": gap.unprofiled}

    return {
        "deep": deep,
        "sources": report.sources,
        "drift": {
            "missing": report.drift.missing,
            "changed": report.drift.changed,
            "intact": report.drift.intact,
            "clean": report.drift.clean,
        },
        "live_chunks": report.live_chunks,
        "dead_chunks": report.dead_chunks,
        "lexical_chunks": report.lexical_chunks,
        "lexical_complete": report.lexical_chunks == report.live_chunks,
        "garbled": [
            {"title": g.title, "chunks": g.chunks, "p90_token": g.p90_token}
            for g in report.garbled],
        # Distinct from `garbled`: those have unusable text, these have none.
        "empty": [
            {"title": e.title, "key": e.key, "bytes_on_disk": e.bytes_on_disk}
            for e in report.empty],
        "models": [
            {"name": m.name, "dim": m.dim, "embedded": m.embedded,
             "to_copy": m.to_copy, "to_embed": m.to_embed, "failed": m.failed,
             "outstanding": m.outstanding,
             "coverage": (m.embedded / (m.embedded + m.outstanding))
             if (m.embedded + m.outstanding) else 0.0,
             "routing": routing_for(m.name)}
            for m in report.models],
    }


def books_payload(conn, pattern: str | None = None) -> dict:
    """Every book, or the ones matching `pattern`. Empty is a valid answer."""
    from dyprys.library import books as inspect

    return {"books": [
        {
            "title": b.title,
            "key": str(b.key),
            "chunks": b.chunks,
            "lexical_indexed": b.lexical,
            "sources": [
                {"ordinal": s.ordinal, "path": str(s.path),
                 "size_bytes": s.size_bytes, "present": s.present}
                for s in b.sources],
            "chunkings": [
                {"id": c.id, "target": c.target, "overlap": c.overlap,
                 "chunks": c.chunks} for c in b.chunkings],
            "embedded": dict(b.per_model),
            # The denominator that goes with `embedded`, per model: this
            # book's chunks under the one chunking that model embeds.
            "live_chunks": {name: b.live_for(name) for name in b.per_model},
        }
        for b in inspect(conn, pattern)]}


# The extensions a local model is packaged as. One, today, and named rather
# than inlined so the two places that scan agree about it.
WEIGHTS_SUFFIX = ".gguf"


# The optional roles, and the shape each one's value takes. `expander` and
# `summariser` name an ollama model; `reranker` is a path to a cross-encoder
# GGUF. Kept here rather than in a frontend because both frontends set them and
# the validation is the same one twice.
ROLES = ("expander", "summariser", "reranker")


def installed_models(host: str = "http://localhost:11434",
                     timeout: float = 0.7) -> list[str]:
    """What the local ollama server has, or [] if it is not running.

    A short timeout on purpose: the server is local, so it answers in
    milliseconds or it is not there, and a listing that pauses three seconds to
    discover nothing is worse than one that says nothing.

    Takes the host as an argument rather than reading it, so the two frontends
    can differ about where ollama lives without this knowing which is asking.
    """
    import json as _json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as response:
            return sorted(m["name"] for m in _json.loads(response.read()).get("models", []))
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return []


def defaults_payload(conn) -> dict:
    """What this library remembers for each optional role."""
    return {"defaults": {role: default_model(conn, role) for role in ROLES}}


def check_reranker(path: str) -> None:
    """Refuse a file that cannot serve as a cross-encoder. Silent when it can.

    Two refusals, and the second is the one that needed writing. A missing file
    announces itself the moment a search reaches for it; a *wrong* file never
    does. `llama.cpp` honours `pooling_type=RANK` on any model, so an embedding
    model loads without complaint and returns one number per pair that looks
    exactly like a relevance score -- measured on one obvious pair,
    embeddinggemma-300M ranked an irrelevant passage above the answer and said
    nothing. So this reads what the GGUF declares about itself.

    Shared by every caller that stores or uses a reranker, because the CLI and
    the API each having their own copy of this is how the two drift.
    """
    from dyprys.rerank import can_rerank

    if not Path(path).exists():
        raise errors.ModelUnavailable(
            f"no such file: {path}. A reranker is a cross-encoder .gguf on this "
            f"machine.", role="reranker")
    if can_rerank(path) is False:
        raise errors.NotAReranker(
            f"{Path(path).name} declares itself an embedding model, not a "
            f"cross-encoder. Used for reranking it returns numbers that are not "
            f"relevance, and ranks silently and plausibly wrong.",
            role="reranker")


def remember_model(conn, role: str, value, *, known=None) -> dict:
    """Store a default optional model for this library, or forget one.

    Validated when it is set rather than when it is next needed, because the two
    can be weeks apart -- a typo stored today should not surface as a failed
    search in a fortnight. `known` is what the machine actually has, when the
    caller was able to find out; a name that is not among them is refused with
    them as the choices, which is what makes a picker possible.
    """
    if role not in ROLES:
        raise errors.BadRequest(
            f"no such role {role!r}", choices=list(ROLES), role=role)

    wanted = (value or "").strip() if isinstance(value, str) else value
    if wanted in (None, "", "none", "off"):
        with conn:
            conn.execute("DELETE FROM meta WHERE key = ?", (f"model.{role}",))
        db.record_event(conn, "default", f"{role} = none")
        return defaults_payload(conn)

    if role == "reranker":
        check_reranker(wanted)
    elif known is not None and wanted not in known and f"{wanted}:latest" not in known:
        raise errors.ModelUnavailable(
            f"nothing installed is named {wanted!r}", choices=list(known), role=role)

    with conn:
        db.set_meta(conn, f"model.{role}", wanted)
    db.record_event(conn, "default", f"{role} = {wanted}")
    return defaults_payload(conn)


# One metadata read per file, kept across requests. Opening a picker should not
# re-read the same six files every time, and the key carries the size so a file
# replaced in place is read again rather than remembered wrongly.
_DESCRIBED: dict[tuple[str, int], dict] = {}


def _describes(path: str, size: int) -> dict:
    """`{architecture, rerank}` for one file. Imported lazily like every other
    use of `rerank` here, so listing models does not pull in llama_cpp until
    something actually reads a file."""
    key = (path, size)
    if key not in _DESCRIBED:
        from dyprys.rerank import can_rerank, declares

        said = declares(path)
        _DESCRIBED[key] = {"architecture": said.get("architecture"),
                           "rerank": can_rerank(path)}
    return _DESCRIBED[key]


def available_models(directories) -> dict:
    """The model files on this machine, under directories the caller chose.

    Which directories is the *frontend's* question — a terminal has a shell and
    a tab-completing path, a browser has neither — so this takes them rather
    than knowing where anyone keeps their weights.

    The point is `--reranker`. Unlike the expander and summariser, whose choices
    the index remembers, a reranker must be named on every search, and bare
    `--rerank` is an error rather than a downgrade. "Type an absolute path each
    time" is not a design in a browser, so something has to list what is there.

    Nothing here guesses what a file *is*, but it does report what the file
    says. `rerank` is read from the GGUF's own `pooling_type`: a cross-encoder
    declares RANK where an embedding model declares MEAN or CLS, and that is a
    fact from the file rather than an inference from its name — which is worth
    the distinction, because `embeddinggemma-300M-Q8_0.gguf` and
    `qwen3-reranker-0.6b-q8_0.gguf` are the same shape and one of them, used as
    a reranker, ranks silently and plausibly wrong.

    `rerank` is **null when the file did not say**, and a caller must not read
    that as "no": a `.gguf` converted before the key existed would otherwise be
    locked out of a job it can do.
    """
    seen: dict[str, dict] = {}
    for directory in directories:
        where = Path(directory).expanduser()
        if not where.is_dir():
            continue
        # One level down as well: models are commonly kept one directory per
        # model. Not deeper — this runs on every request that opens a picker.
        for pattern in (f"*{WEIGHTS_SUFFIX}", f"*/*{WEIGHTS_SUFFIX}"):
            for found in sorted(where.glob(pattern)):
                try:
                    size = found.stat().st_size
                except OSError:
                    continue
                if str(found) in seen:
                    continue
                said = _describes(str(found), size)
                seen[str(found)] = {
                    "path": str(found), "name": found.name, "bytes": size,
                    "directory": str(found.parent),
                    "architecture": said["architecture"],
                    "rerank": said["rerank"],
                }
    return {"models": sorted(seen.values(), key=lambda row: row["name"].lower()),
            "searched": [str(Path(d).expanduser()) for d in directories]}


def name_model(conn, wanted: str, alias: str) -> dict:
    """Give a model a short name to type. The weights identity is unchanged.

    `hf_ggml-org_embeddinggemma-300M-Q8_0@b5ce9d77a3fc` identifies weights
    exactly and tells a person nothing, which is what the alias is for — and it
    is stored in the index, so a name chosen in a browser is one the terminal
    can type too.
    """
    chosen = (alias or "").strip()
    if not chosen:
        raise errors.BadRequest("an alias needs to be something")
    if chosen.endswith(WEIGHTS_SUFFIX) or "/" in chosen:
        raise errors.BadRequest(
            f"{chosen!r} looks like a path; an alias is a short word to type "
            f"in place of one")
    row = db.find_model(conn, wanted)
    if row is None:
        raise errors.ModelMissing(
            f"no model matches {wanted!r}",
            choices=[m["name"] for m in
                     conn.execute("SELECT name FROM models ORDER BY id")])
    taken = conn.execute(
        "SELECT name FROM models WHERE alias = ? AND id != ?",
        (chosen, row["id"])).fetchone()
    if taken is not None:
        # The column is uniquely indexed, so this would otherwise surface as an
        # IntegrityError -- a 500 for something the caller can fix.
        raise errors.BadRequest(f"{chosen!r} is already {taken['name']}'s alias")
    db.set_alias(conn, row["id"], chosen)
    return {"name": row["name"], "alias": chosen}


def models_payload(conn, directory) -> dict:
    """What has embedded this index, how far, and at what cost on disk."""
    from dyprys.library import models as inspect

    return {"models": [
        {
            "name": m.name,
            "alias": m.alias,
            "dim": m.dim,
            "store": m.quantisation,
            "embedded": m.embedded,
            "live_chunks": m.live_chunks,
            "coverage": m.coverage,
            "disk_bytes": m.bytes_on_disk,
            "failures": m.failures,
            "carries": m.carries,
            "routing": {"profiled_books": m.centroid_books,
                        "stale_books": m.stale_books},
            "file_path": m.file_path,
            "file_present": bool(m.file_path) and Path(m.file_path).exists(),
            # Null until this model has embedded something. A frontend that
            # offers a chunk size needs it: bound, the choice is already
            # made and cannot be changed; unbound, it is a real choice.
            "chunking_id": m.chunking_id,
        }
        for m in inspect(conn, directory)]}


# Below this, the difference between `seconds` and `wall_seconds` is the
# machine having been asleep rather than a run having been slow. Under a minute
# it is clock jitter and not worth reporting.
SLEPT_AT_LEAST = 60


def asleep_seconds(run) -> float:
    """Seconds this run spent with the machine asleep, or 0 if not knowable.

    `seconds` is monotonic and stops while the machine sleeps; `wall_seconds`
    does not. Their difference is the sleep, and it is the only way to tell a
    run that worked for thirty minutes from one that was begun thirty minutes of
    work ago -- which on a laptop are routinely hours apart.
    """
    try:
        wall = run["wall_seconds"]
    except (IndexError, KeyError):
        return 0.0
    if wall is None:
        return 0.0
    gap = wall - run["seconds"]
    return gap if gap >= SLEPT_AT_LEAST else 0.0


def history_payload(conn, limit: int = 20) -> dict:
    """What has been done to this index, and what embedding it cost."""
    events = conn.execute(
        "SELECT at, action, detail FROM events ORDER BY id DESC LIMIT ?",
        (limit,)).fetchall()
    runs = conn.execute(
        "SELECT r.*, m.name FROM embed_runs r JOIN models m ON m.id = r.model_id "
        "ORDER BY r.id DESC LIMIT ?", (limit,)).fetchall()
    total = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(seconds),0), COALESCE(SUM(embedded),0) "
        "FROM embed_runs").fetchone()
    return {
        "events": [
            {"at": e["at"], "action": e["action"], "detail": e["detail"]}
            for e in reversed(events)],
        "runs": [
            {"started_at": r["started_at"], "seconds": r["seconds"],
             "wall_seconds": (r["wall_seconds"] if "wall_seconds" in r.keys() else None),
             "embedded": r["embedded"], "copied": r["copied"], "failed": r["failed"],
             "rate": (r["embedded"] / r["seconds"]) if r["seconds"] else 0.0,
             "stopped": r["stopped"], "model": r["name"],
             "asleep_seconds": asleep_seconds(r)}
            for r in reversed(runs)],
        "totals": {"runs": total[0], "seconds": total[1], "chunks": total[2],
                   "mean_rate": (total[2] / total[1]) if total[1] else 0.0},
    }


def _asked_row(row) -> dict:
    import json

    return {"id": row["id"], "at": row["at"], "question": row["question"],
            "mode": row["mode"], "ms": row["ms"],
            "detail": json.loads(row["detail"])}


def asked_payload(conn, limit: int = 15, match: str | None = None,
                  which: int | None = None):
    """The question log -- the list, or one question in full, or None.

    `None` rather than a raise for a number that names nothing: `dyp asked 9999`
    exits 1 with a `null` body today, and an API turns the same absence into a
    404. Both are answers about a row, not failures of the command.
    """
    if which is not None:
        rows = db.questions_asked(conn, limit=10 ** 9)
        one = next((r for r in rows if r["id"] == which), None)
        return _asked_row(one) if one else None
    return {"questions": [_asked_row(r)
                          for r in db.questions_asked(conn, limit, match)]}


# --- naming a library ------------------------------------------------------
#
# `dyp library add|remove|use` write the registry; the API had no equivalent, so
# a browser could read every library and name none. These three wrap
# `registry` so both frontends refuse the same things for the same reasons —
# and so the refusals arrive as typed errors rather than as a `print` and a 1.


def _registry_name(name: str) -> str:
    """A name that can be a path segment and a shell word.

    The API carries the name in the URL, so a `/` would silently change which
    route matched; the CLI carries it as an argument. Refused in the service so
    both agree, and so a name registered from the browser is one the terminal
    can still type.
    """
    cleaned = (name or "").strip()
    if not cleaned:
        raise errors.BadRequest("a library needs a name")
    if any(c in cleaned for c in "/\\ \t\n") or cleaned in (".", ".."):
        raise errors.BadRequest(
            f"{cleaned!r} cannot be a library name — no slashes or spaces, "
            f"because the name is a path segment in the API and an argument to "
            f"`dyp -L`")
    return cleaned


def register_library(name: str, path, make_default: bool | None = None) -> dict:
    """Give a directory a name. The library's files are never created here.

    The directory must already exist. `dyp library add` allows one that does
    not, because the next command in a terminal usually creates it; a browser
    has no such next command and a typo would register a path that nothing ever
    reports as wrong.
    """
    from dyprys import registry

    named = _registry_name(name)
    where = Path(path).expanduser()
    if not str(path).strip():
        raise errors.BadRequest("a library needs a directory")
    if not where.exists():
        raise errors.NoSuchLibrary(f"nothing at {where} — the directory must exist")
    if not where.is_dir():
        raise errors.BadRequest(f"{where} is a file; a library is a directory")
    taken = {entry.name for entry in registry.libraries()}
    if named in taken:
        raise errors.BadRequest(
            f"{named!r} is already registered — forget it first, or pick "
            f"another name", choices=sorted(taken))

    registry.add(named, where.resolve(), make_default)
    return libraries_payload()


def forget_library(name: str) -> dict:
    """Drop a name. The index and the text behind it are not touched.

    Deliberately *only* the name. `dyp library remove --delete` will erase an
    index directory; over HTTP that is a button one misclick away from days of
    embedding, so the API does not offer it and the message says where it lives.
    """
    from dyprys import registry

    if not registry.remove((name or "").strip()):
        raise errors.NoSuchLibrary(
            f"no library named {name!r}",
            choices=[entry.name for entry in registry.libraries()])
    return libraries_payload()


def default_library(name: str) -> dict:
    """Which library a bare `dyp` command means."""
    from dyprys import registry

    if not registry.use((name or "").strip()):
        raise errors.NoSuchLibrary(
            f"no library named {name!r}",
            choices=[entry.name for entry in registry.libraries()])
    return libraries_payload()


def libraries_payload() -> dict:
    """Every registered library, whether its index is there, and how far it got.

    Reads each index rather than the registry alone, because a registry entry is
    a name and a path and says nothing about whether either still means
    anything. `registry.summarise` opens read-only and returns None for a path
    that is gone, which is the difference between "not registered" and "the
    drive is not mounted".
    """
    from dyprys import registry
    from dyprys.library import notes_path

    libs = []
    for entry in registry.libraries():
        exists = db.index_exists(entry.path)
        note = notes_path(entry.path) if exists else None
        row = {"name": entry.name, "path": str(entry.path),
               "default": getattr(entry, "is_default", False),
               "exists": exists, "notes": str(note) if note else None}
        held = registry.summarise(entry.path) if exists else None
        if held is not None:
            row.update(books=held.books, chunks=held.chunks, models=[
                {"name": name, "embedded": done, "total": whole,
                 "coverage": (done / whole) if whole else 0.0}
                for name, done, whole in held.models])
        libs.append(row)
    return {
        "libraries": libs,
        "registry_path": str(registry.registry_path()),
        "registry_readable": registry.readable(),
    }
