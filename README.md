# dyprys

Local semantic search over a personal library of book-length texts. Ask a
question, get back the **passage that answers it** — the text itself, with the
file and byte it came from — not a ranked list of books to go and read.

Everything runs on your machine. Nothing is sent anywhere. Built to stay fast
from one book to several thousand.

Command: `dyp`.

---

## What it feels like

```sh
dyp ask "why can a nerve impulse only travel in one direction"
```

```
1. [cos 0.52 · vec 1 · phrase 1] Principles.of.Neural.Science  (chunk 21482)
   … the refractory period following an action potential leaves the stretch of
   membrane behind it briefly unexcitable, so the impulse cannot turn back …
   ~/books/Principles.of.Neural.Science.txt:523017
```

Two halves run on every query and their results are merged: a **meaning** search
(describe a thing you can't name) and a **keyword** search (quote a line you
half-remember). You never choose between them.

**Read more than the first result.** On a reference library of 3,453 books the
answer is the very first hit about 6 times in 10, and somewhere in the top five
about 8 in 10 — so a single result is often wrong where five would be right. Ask
for a handful and read them.

---

## Install

```sh
git clone git@github.com:mendelcyprys/dyprys.git
cd dyprys
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

`uv` is optional — `python3 -m venv .venv` and `pip` work identically. `-e`
installs in editable mode. Check it worked:

```sh
dyp            # prints help
pytest         # 755, all green
```

Two runtime dependencies only: `llama-cpp-python` (the embedding model, run
locally against a GGUF file) and `numpy`.

---

## The core loop

```sh
dyp add ~/library              # 1. record where each book's text is and how it splits
dyp embed --model model.gguf   # 2. turn the text into vectors (slow, resumable, once)
dyp route                      # 3. profile each book so search can skip most of them
dyp ask "why do neurons fire"  # 4. search — best passages first
```

Only step 2 is slow, and you run it once. The rest are instant.

---

## Searching well

An agent driving `dyp` reads `CLAUDE.md`, a short decision manual for exactly
this. For a person, the same in prose:

**Phrase the question for the half that will answer it.**

```sh
dyp ask "what stops the brain being flooded by blood-borne toxins"   # describe it
dyp ask "move by saltatory conduction from the Latin saltare"        # quote it
```

Describe a thing you cannot name and the meaning half finds it. Quote a line you
remember and the keyword half finds it verbatim — so do **not** paraphrase a
remembered quote, and spell a rare name exactly, because a name only ever matches
literally.

**Reading a result:**

- `cos 0.52` — closeness in meaning, 0–1. Comparable *within one search only*,
  and a higher number is **not** a better answer. Worth seeing, not worth ranking
  on.
- `vec 1 · phrase 1` — which half found the passage and at what rank. `phrase`
  means it holds your words in order (a literal hit); `words` means it shares
  vocabulary; `vec` means it matched on meaning.
- the last line — the file and byte offset, so you can open the source directly.

**The flags worth knowing:**

```sh
dyp ask "..." -k 10            # return ten passages, not the default five
dyp ask "..." --route          # ~6x faster: score every book cheaply, search the best five
dyp ask "..." -c neuroscience  # scope to a shelf, an author, or a glob
dyp ask "..." --full           # the whole passage, not just its query-densest part
dyp ask "..." --json           # machine-readable output, for a program to parse
```

`--route` reads about 1% of the library and costs roughly one answer in
twenty-five — right for a quick look, worth dropping when the answer matters more
than the second it saves.

**Two optional stages** need a language model and are **off by default**, because
each costs seconds and the defaults are for typing at a prompt:

```sh
dyp ask "..." --expand gemma3:4b     # rewrite the question into the library's words first
dyp ask "..." --summarise gemma3:4b  # draft a written answer from the passages
```

`--expand` helps most when you ask in everyday words about a technical library,
and not at all when the answer is a rare literal name. `--summarise` drafts prose
but **trusts nothing**: every quotation is checked byte-for-byte against the
source and cut if it does not match, and the book/chunk/offset of each citation
come from the index, never from the model. A separate `--rerank` reorders results
with a cross-encoder; it finds a few more answers for far more time. **Do not
stack `--expand` and `--rerank`** — measured, they recover the same answers, so
pay for one, not both.

---

## Building an index

**What a library looks like.** Plain text, `.txt` by default; `--ext .md,.text`
widens it. If nothing matches, `add` says so and exits non-zero rather than
quietly doing nothing.

```
library/                         library/
  Some Book.txt   ← one book       The Long Book/   ← one book with --chapters,
  Another.txt     ← one book         chapter01.txt     its files as chapters in
                                     chapter02.txt     filename order
```

One-book-or-a-shelf is stated (`--chapters`), never guessed — a wrong guess would
mis-model the library and only show up as wrong results later. Extraction from PDF
or EPUB happens upstream; dyprys indexes the text you give it.

**Embedding** is the one slow step — about 5 passages a second, so a few hundred
books is an overnight job. It is resumable at the granularity of a single passage,
so stopping never loses work:

```sh
dyp embed --for 45m    # stop after three quarters of an hour
dyp embed --duty 80    # spend 80% of the time working, keep the machine usable
dyp watch              # follow a run from another terminal
```

**Choosing a model matters.** A small model (e.g. jina-embeddings-v2-small, 512
dims) builds its index several times faster and *finds* the answer nearly as
often, but ranks it a little worse. A larger one (e.g. EmbeddingGemma-300M, 768
dims) ranks best out of the box at several times the build cost. Prefer the small
model for a large or growing library and pair it with `--rerank`; prefer the
larger when the corpus is small and the first result matters most. Each model
keeps its own vector file, coverage and routing profile in the one index, and you
can hold several side by side.

Run `dyp route` after embedding so search can skip most of the library.

---

## Several libraries

```sh
dyp library add neuro ~/indexes/neuro   # name a directory
dyp library use neuro                    # make it the default
dyp -L neuro ask "..."                   # or address one by name per command
```

Which index a command uses is resolved `--data DIR` → `-L NAME` → the default →
`./data`; explicit always beats remembered, so a stale default can never silently
redirect a command that named its target. The registry holds names and paths
only — losing it costs nothing, every library still opens with `--data`.

---

## Looking after a library

```sh
dyp books [PATTERN]    # what is in the library, or one book in detail
dyp status             # totals, and what to do next
dyp models             # embedding models: coverage, disk, routing profile
dyp check              # what drifted, what work is outstanding — changes nothing
```

`check` is the one to run when something looks off: source files that moved or
vanished, chunks awaiting embedding, whether the keyword index and routing
profile are current.

```sh
dyp backup lib.tar.gz                              # index + vectors + text, one archive
dyp restore lib.tar.gz --into ~/idx --sources ~/books --as name
dyp relocate OLD NEW                               # the text moved; rewrite the paths
dyp remove PATTERN --yes && dyp compact --yes      # forget books, reclaim their space
dyp models --drop NAME --yes                       # remove a model and its vectors
dyp models --quantise NAME                         # convert vectors to int8, a quarter the disk
```

An index is three things that travel together — the database, the vectors, and
the source text the offsets point into — and `backup`/`restore` keep them
together so nothing is re-embedded. A backup taken mid-embed is still consistent.

**A model is identified by a hash of its weights, not its filename**, so someone
else's restored index reuses your own copy of the same GGUF automatically. If you
lack the weights, the manifest records their name, size and SHA-256, and
`dyp models --verify FILE` checks a candidate before you spend hours embedding
under the wrong identity by mistake.

If the source text is edited after indexing, dyprys still finds each passage where
it can prove — by a stored content hash — that the bytes are the ones it embedded;
a passage it cannot prove is withheld, never shown stale. An edit that only shifts
a file recovers automatically; `dyp add` is idempotent and a moved file keeps its
vectors.

---

## From a browser

`dyp serve` puts the same searches behind an HTTP API, for a web UI to call.
It is an optional extra — core `dyp` keeps its two dependencies.

```sh
uv pip install -e '.[api]'
dyp serve --port 8765            # 127.0.0.1 only, no authentication
```

```
GET    /api/health · /api/libraries
GET    /api/libraries/{name}/status | check | books | models | history | asked
GET    /api/libraries/{name}/notes            the library's NOTES.md, raw
POST   /api/libraries/{name}/ask              the `--json` payload, plus warnings
POST   /api/libraries/{name}/ask/stream       NDJSON: stage frames, then one result
GET    /api/libraries/{name}/source?path=&offset=&span=
POST   /api/libraries/{name}/warm             load a model before the first question
GET    /api/libraries/{name}/jobs             what is running, how far, how fast
POST   /api/libraries/{name}/jobs/{kind}      start embed | add | route | lexical | compact
DELETE /api/libraries/{name}/jobs/{kind}      stop it, gracefully
```

Three things are worth knowing before building against it.

**Searching happens in this process; every mutation is a subprocess.** The warm
model is the whole reason the API exists — reloading a 300 MB–1 GB GGUF per query
costs more than the search does — so `ask` holds it between requests. Writes go
the other way and spawn a real `dyp`, because an embed can run for days and has to
outlive a server restart. `dyp watch` in a terminal follows a run the browser
started, and `GET /jobs` reports on one a terminal started; there is one run,
seen from two places. Stopping one is safe because embedding is resumable: the
signal lets it finish the batch in flight and commit.

**Each library gets one worker thread.** A sqlite connection belongs to the
thread that opened it and llama.cpp is not re-entrant, so a second simultaneous
search against the same library queues — which is right, since the model is the
bottleneck and running two at once only makes both slower.

**It reports rather than interprets.** `cos`, `provenance`, `state`,
`scanned_fraction` and `warnings` pass through untouched, and `text` stays `null`
for a passage that could not be proved against its stored hash. Nothing about a
corpus is encoded in the server: `GET /notes` transports the library's own notes,
and every route is scoped by a library name resolved through the registry.

Bind `127.0.0.1` and keep it there. There is no authentication — this is one
person's library on one machine — so a public interface would put an
unauthenticated reader of your files on the network.

→ **`docs/serving.md`** — the payload shapes, the job lifecycle, the status
codes, and how the two frontends are kept from drifting.

---

## How it works, in one screen

A handful of decisions carry the design. They are stated here so the tool is
understandable on its own.

- **The chunk is the unit of everything.** A passage of a few thousand bytes has
  an id, and that id *is* its row in the vector file. Ids are dense and allocated
  per book, so a book, a chapter or a shelf is a contiguous slice of one array —
  routing to it is one memory-mapped slice, one matrix multiply. There is no
  separate index to keep in sync, and therefore no recall ceiling.
- **Text lives in the source files, never the database.** The index stores
  `(path, offset, length)` and reads the window by seek. Results carry ids,
  offsets and scores — never text — until the moment a passage is displayed,
  embedded or scored.
- **Two-stage search.** Stage 1 profiles each book into a few spherical k-means
  centroids and scores a query against every book's *best* centroid; stage 2 runs
  exact cosine over only the top few books. On the reference corpus that is ~90 ms
  against ~600 ms, reading ~1% of the passages.
- **Hybrid, fused by rank.** Vector and keyword (BM25) results are combined by
  reciprocal rank, not by score — a cosine and a BM25 score are not comparable
  numbers. The exact-phrase probe searches the whole library even when routing is
  on, so a remembered sentence stays findable.
- **int8 vectors by default.** A per-row scale and one byte per dimension, a
  quarter the size of float32 for no measured loss in what search returns.
- **Resumable and crash-safe.** A vector is written and flushed *before* the
  count that claims it exists, so an interruption re-embeds one batch rather than
  corrupting the store.

---

## Layout

```
src/dyprys/
  chunker.py     text -> byte spans; a pure function, no I/O
  db.py          the schema, connection, migrations and derivations
  ingest.py      walk files, hash, chunk, allocate ids, salvage vectors
  embed.py       the resumable embedding loop
  embedder.py    the GGUF model, behind one interface
  vectors.py     the sparse per-model memmap
  search.py      exact vector search, scoping, resolving passages
  routing.py     spherical k-means; stage 1
  lexical.py     BM25 and reciprocal-rank fusion
  rerank.py      cross-encoder reordering (optional)
  expand.py      query rephrasing (optional)
  summarise.py   answers with checked quotations (optional)
  evaluate.py    the retrieval harness
  text.py        the only place that reads a book's bytes back
  term.py        emphasis, extracts, bars — and none of it when piped
  errors.py      failures as typed values, with the prose attached
  progress.py    what a slow search is doing, before anyone renders it
  service.py     the ask pipeline, model resolution and the --json payloads
  jobs.py        the long mutations, as detached subprocesses
  cli.py         `dyp` resolves to dyprys.cli:main
  api.py         the same, over HTTP (optional: `pip install 'dyprys[api]'`)
tests/           pytest — 755 tests
CLAUDE.md        the operator's decision manual, loaded by an agent
docs/            deeper operational reference, linked on-demand from CLAUDE.md
```

**The split is by I/O, not by topic** — the pure functions (chunking, fusion,
the JSON shape) test without a model, a database or a temp directory. Add a
dependency in `pyproject.toml`, never with a bare `uv pip install`, or the project
will work here and fail on a fresh clone.

The index, vectors and source text live outside version control (`*.sqlite`,
`*.npy`, `*.f32`, `data/` are gitignored). Nothing that grows with the library
belongs in git.

## Measuring it

```sh
dyp eval --compare    # vector vs keyword vs hybrid on a question set
dyp eval --route      # the same, routed, with routing recall beside its chance baseline
```

The harness scores whether the returned *passage* contains the answer, not
whether the right book appeared — a document-level hit rate is blind to which
passage was chosen, which is the thing most likely to be wrong. Question sets live
in `eval/`; a routing measure is always reported next to the score random book
selection would get, because on a library where the answer is everywhere that
baseline is most of the number.

## Provenance

Architecture informed by [qmd](https://github.com/tobi/qmd) (MIT, © 2024–2026
Tobi Lutke) — by diagnosing where its design runs out at book scale. No qmd code
is reused.
