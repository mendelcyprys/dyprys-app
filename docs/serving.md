# The HTTP frontend — reference

`dyp serve` puts everything `dyp ask` does behind an API, so a browser can call
it. Optional: FastAPI and uvicorn are an extra, and core `dyp` keeps its two
runtime dependencies.

```sh
uv pip install -e '.[api]'      # or: pip install 'dyprys[api]'
dyp serve --port 8765
```

`--host` (default `127.0.0.1`), `--origin URL` (repeatable; default is Vite's
`http://localhost:5173`), `--web DIR` to also serve a built frontend, `--keep N`
for how many embedding models to hold warm per library (default 2, each
300 MB–1 GB resident). Interactive docs at `/docs`.

**There is no authentication.** This is one person's library on one machine, so
binding anything but a loopback address puts an unauthenticated reader of your
files on the network. `dyp serve` says so on stderr if you do it anyway.

## The routes

```
GET    /api/health                            is it up, and which models are warm
GET    /api/libraries                         every registered library
POST   /api/libraries                         name a directory on this machine
DELETE /api/libraries/{name}                  forget a name; files untouched
POST   /api/libraries/{name}/default          which library a bare `dyp` means
GET    /api/libraries/{name}/status           totals, coverage, the notes path
GET    /api/libraries/{name}/check?deep=      what drifted, what is outstanding
GET    /api/libraries/{name}/books?pattern=   what is in the library
GET    /api/libraries/{name}/models           what embedded it, how far
GET    /api/libraries/{name}/models/available the .gguf files on this machine
POST   /api/libraries/{name}/models/{m}/alias a short name to type
GET    /api/libraries/{name}/history?limit=   embed runs and index operations
GET    /api/libraries/{name}/asked?limit=&find=&which=
GET    /api/libraries/{name}/notes            the library's NOTES.md, raw
POST   /api/libraries/{name}/ask              search
POST   /api/libraries/{name}/ask/stream       search, as NDJSON
GET    /api/libraries/{name}/source?path=&offset=&span=
POST   /api/libraries/{name}/warm             load a model before the first question
GET    /api/libraries/{name}/jobs             every job kind at once
GET    /api/libraries/{name}/jobs/{kind}?tail=      one, with its log tail
POST   /api/libraries/{name}/jobs/{kind}      start embed|add|route|lexical|compact
DELETE /api/libraries/{name}/jobs/{kind}?force=     stop it
```

The six read-only payloads are **the same objects `dyp X --json` prints** —
literally: one builder in `dyprys.service` feeds both frontends, and
`tests/test_api.py::test_the_api_returns_what_the_cli_prints_for_json` compares
them. So `docs/maintenance.md` and CLAUDE.md's `--json` notes describe these
responses too, and there is nothing extra to learn.

## Naming a library

`dyp library add|remove|use` had no HTTP equivalent, so a browser could read
every library and name none. Three routes close that, and all three return the
same `GET /api/libraries` payload so a client can seed its cache from the reply.

```json
POST /api/libraries   {"name": "neuro", "path": "/Users/you/dyprys/neuro"}
```

The path is **server-side text, not an upload** — a browser cannot pick a
directory, and this server is loopback-only and already reads the whole
filesystem through `dyp add`. Registering creates nothing; it is the registry
entry alone.

Three refusals the CLI does not make, all of them because the caller is now a
browser:

- **the directory must exist.** `dyp library add` allows one that does not,
  because the next command in a terminal usually creates it. A browser has no
  such next command, so a typo would become an entry that nothing reports as
  wrong and every use fails on.
- **no slashes or spaces in the name.** It is a path parameter on every other
  route, so a `/` would silently change which route matched, and a space would
  stop `dyp -L` from saying it — the two frontends have to be able to mean the
  same library.
- **a name already taken is a 400 carrying every taken name in `choices`**,
  rather than `registry.add`'s silent overwrite, which over HTTP is a lost
  library.

`DELETE` forgets the name and touches nothing on disk. The CLI's
`--delete`, which erases the index directory, deliberately has **no route**:
over HTTP that is one misclick from days of embedding, and a terminal is the
right place to confirm it. Deleting also drops the library's warm session,
because the name has stopped meaning that directory.

## Searching

`POST /ask` takes any field of `service.SearchOptions` plus `question`:

```json
{"question": "what is long-term potentiation",
 "k": 5, "mode": "hybrid", "model": "gemma", "collection": "neuroscience/",
 "route": 8, "rerank": 10, "reranker": "/path/qwen3-reranker-0.6b.gguf",
 "expand": true, "summarise": "gemma3:4b", "dedupe": false}
```

An option this dataclass does not name is a 400 that lists the ones it does,
rather than being ignored — a silently dropped `k` is worse than a refusal.

**`expand` and `summarise` are `bool | str`.** `true` means "whatever this
library remembers", a string names a model. That is exactly what argparse's
`nargs="?", const=True` gives the CLI, and typing the field `str | None` would
reject the bare form the CLI has always accepted.

The response is `dyp ask --json`'s payload — `query`, `mode`, `routed`,
`scanned_fraction`, `elapsed_ms`, `results[]` with `rank`, `chunk_id`, `book`,
`chapter`, `path`, `offset`, `cos`, `provenance`, `state`, `text` — plus four
fields the CLI prints rather than serialises: `warnings`, `answer`, `expansion`,
`models`.

`answer` carries three fields beyond the prose and its checked quotations, and
each says something a reader cannot otherwise recover:

- **`refused`** — the rephrasing that *also* found nothing. `--summarise` says
  "these passages do not answer the question" and retries once with the model's
  own words before giving up. A refusal that survived that is strong evidence
  the library lacks the answer; one that was never rephrased is not, and stored
  prose reading `NO ANSWER IN PASSAGES` looks identical either way.
- **`retried`** — the rephrasing that worked.
- **`drawn_from`** — present only after a successful retry, and then it matters
  a great deal: the answer is about the *retry's* passages while `results` still
  holds the original search, and `verified[].cited` indexes into these. Without
  it a reader is shown an answer citing passages that are not on the screen,
  with citations that look correct.

Read `results` the way `docs/searching.md` says to read the terminal output.
Three of those rules matter more over HTTP, because a browser hides what a
terminal shows:

- **`text` is `null` when the passage could not be proved** against its stored
  content hash, and `state` says why. Never render it as a quotation. A UI that
  coerces null to `""` erases the only signal saying the tool cannot vouch for
  the words.
- **`cos` is not a quality score.** It is comparable only within one response,
  and a higher number is not a better answer. Show it; do not sort on it.
- **`warnings` is where routing's blind spot goes.** A book embedded since the
  last `dyp route` has no profile, so `--route` cannot return it at *any* rank —
  and the response is still a full `k` results at 200, which looks exactly like a
  complete search. On a terminal that warning is a line on stderr; here it is
  `warnings: [{"kind": "unprofiled_books", "message": "..."}]` and nothing else
  will tell you.

`warnings` is always present, empty when there is nothing to say, so a caller
can read it unconditionally.

### Streaming

`POST /ask/stream` returns `application/x-ndjson`: any number of stage frames,
then exactly one final frame.

```
{"stage": {"kind": "note", "label": "routed to 8 of 225 books", "detail": "", "indent": 0}}
{"stage": {"kind": "note", "label": "Sacrificial_Procedure", "detail": "", "indent": 1}}
{"result": { ...the same payload as /ask... }}
```

Everything the CLI narrates on stderr arrives as a stage frame: the rephrasings
an expansion model chose, the books routing picked, "rescoring 20 passages",
the summariser's retry. A ten-second search with no output is indistinguishable
from a hang, which is why they exist.

A failure after the headers are sent cannot be a status code, so the final frame
is `{"error": ..., "detail": ..., "choices": [...]}` instead. **The last frame is
always one or the other** — a stream that simply stops is a bug, not an empty
result.

## The reader pane

`GET /source?path=&offset=&span=` returns `{"path", "offset", "text"}` — bytes
around an offset, snapped to sentence boundaries, for showing a passage in place.

The path is checked against this index's `sources` table and nothing else. A
prefix check is **not** equivalent and must not be substituted: symlinks, `..`,
and a second library's files all pass one. Anything unlisted is a 404. `span` is
capped at `service.MAX_SPAN` (200 KB) rather than refused, because a UI asking
for too much should get what it may have.

## Choosing a model

`GET /models/available` lists the `.gguf` files on the machine, so a browser can
offer them instead of asking for an absolute path.

**It exists for `--reranker`.** The expander and summariser default to whatever
the library last used, and the index stores it. The cross-encoder is not
remembered anywhere, must be named on every search, and bare `rerank` is an
error rather than a downgrade — so a browser needs somewhere to get one from,
and a UI keeping it per library in `localStorage` is the whole of that design.

Where to look is the **frontend's** question and is answered in `api.py`: the
directories of weights this index already remembers (the likeliest home, and
needing no configuration), `$DYPRYS_MODEL_DIR` for anywhere else, and
`~/.cache/qmd/models`. One level down each, since models are commonly kept one
directory per model. `searched` comes back with the list, because an empty
result is only readable next to where it looked.

Nothing here guesses what a file *is*. A cross-encoder and a chat model are both
a `.gguf` and the difference is not in the name; a guess reported as a fact
would be worse than nothing, since a chat model used as a reranker rescores
silently and plausibly.

`POST /models/{model}/alias` gives a model a short name.
`hf_ggml-org_embeddinggemma-300M-Q8_0@b5ce9d77a3fc` identifies weights exactly
and tells a person nothing. The alias is stored in the index rather than the
client, so one chosen in a browser is one `dyp --model` accepts in a terminal.
An alias that looks like a path is refused — `--model` takes either, so the
confusion is real — and so is one already belonging to another model, which the
unique index would otherwise raise as a 500.

## Jobs

Searching runs in this process; every mutation runs as a detached `dyp`
subprocess. Three reasons, all of them about time:

- the warm model is the whole point of the API — reloading a 300 MB–1 GB GGUF
  per query costs more than the search does — so `ask` holds it between requests;
- `lock.exclusive` is a per-process advisory lock, so a server that embedded
  in-process could not answer a question while it did;
- a run can last days and must outlive a server restart. Nothing here holds a
  handle on it: progress is committed to the index per batch, so `GET /jobs` is a
  *read of the index*, not a pipe.

So there is one run, seen from two places. `dyp watch` in a terminal follows a
run the browser started; `GET /jobs` reports on one a terminal started.

```jsonc
// GET /api/libraries/neuro/jobs/embed
{"kind": "embed", "running": true, "pid": 43666,
 "model": "hf_ggml-org_embeddinggemma-300M-Q8_0@b5ce9d77a3fc",
 "done": 64, "live": 960, "share": 0.067,
 "rate": 38.3, "eta_seconds": 23, "book_in_flight": "doc0",
 "log": "/path/.jobs/embed-20260908T112926.log"}
```

`rate` comes from the gap between *your own* polls, because a request cannot
sleep to take a second sample. Until a second poll arrives it falls back to
`db.observed_rate` — the median this model has actually managed on this machine,
the same figure `dyp check` estimates from — so the first reading is a decent
guess and the second is measured.

"Nothing is running" is a state, not a 404: a UI polls this, and an error code
for the ordinary case makes every poll look like a failure.

**Starting**: `POST /jobs/{kind}` with the flags as a JSON object —
`{"for": "45m", "duty": 80, "model": "..."}` for `embed`, `{"paths": [...]}` for
`add`. Only options the kind actually has are accepted; anything else never
reaches the subprocess. Returns 202 with the pid and the log path. **409** if
something already holds the lock — refused here rather than in the child,
because `dyp embed` loads its GGUF *before* it locks, so letting the subprocess
discover the clash would show a job that starts and dies a minute later for no
visible reason.

**Stopping**: `DELETE /jobs/{kind}` sends SIGINT, which `dyp embed` handles by
finishing the batch in flight and committing it — so nothing is lost and the next
run picks up exactly where this one left off. That is the only reason a stop
button is safe to offer. `?force=true` escalates to SIGTERM.

Logs land in `INDEX/.jobs/<kind>-<stamp>.log`; `GET /jobs/{kind}?tail=N` returns
the end of the most recent one.

## Failures

One handler, one shape:

```json
{"error": "model_ambiguous", "detail": "this index has 2 models — say which…",
 "choices": ["gemma", "jina"], "role": null}
```

`error` is a stable slug to match on, `detail` is the sentence the CLI prints —
the same sentence, from the same exception — and `choices` is what makes a picker
possible. `role` names which model is missing (`reranker`, `expander`,
`summariser`, `model`) where that applies.

| status | when |
|---|---|
| 404 | unknown library, unmatched `-c` pattern, a path this index does not own |
| 409 | something already holds the index's lock |
| 422 | an empty query — refused, never answered with whatever the vector half drifts to |
| 503 | a model the request needs is not on this machine (reranker, expander, weights) |
| 400 | everything else, including `model_ambiguous` and an unknown option |

503 rather than 400 for a missing model is deliberate: the request was fine and
the server could do this. In particular **`rerank` with no reranker is an error,
never a quiet downgrade** — a 200 carrying unreranked results would be the worst
outcome of the three, because the caller asked for a rescoring and would believe
it happened.

## Concurrency

Each library gets one `ThreadPoolExecutor(max_workers=1)` and one session on it,
created lazily. `db.connect` does not pass `check_same_thread`, so a connection
belongs to the thread that opened it, and `llama_cpp.Llama` is not re-entrant
either — one thread per library makes both facts moot instead of managed.

A second simultaneous search against the same library queues. That is correct
rather than a limitation: the model is the bottleneck, so running two at once
would only make both slower. Different libraries run in parallel.

`--keep` caps warm models per library and evicts least-recently-used.
`GET /api/health` lists what is currently resident, which is the number to watch
if the server's memory matters.

## Corpus-independence

Nothing about any particular library is in the server. Library identity is always
a path parameter resolved through the registry; there is no default `--model`,
`--reranker` or `--expander`; and corpus knowledge stays *data* — `GET /notes`
transports the library's own `NOTES.md`, which CLAUDE.md tells a reader to read
before searching, and never interprets it.

## Where the seam is held

The API and the CLI are both thin. Everything between them is `dyprys.service`,
and three test files keep it that way:

| file | holds |
|---|---|
| `tests/test_layering.py` | no module below the frontends prints, exits, reads `argv`, resolves a model from the environment, or catches a typed error |
| `tests/test_ask_pipeline_contract.py` | what the retrieval core must still do — a verbatim phrase first, `-c` confining the ranking half, provenance attached |
| `tests/test_api.py` | status-code mapping, `text: null` surviving, warnings on the payload, `?path=` bounded by `sources`, stream frames, jobs 409 — and that the API and CLI serialise the same object |

If you add an endpoint, add it as a call into `service` plus a serialisation. A
route that reaches into `db` or `search` directly is how the two frontends start
disagreeing.
