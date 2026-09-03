# dyprys

Semantic search across a personal library of book-length texts, built to scale
to 1,000–5,000 volumes. Returns the passage that answers the question, not a
list of books to go and read.

Command: `dyp`

## Status

Working end to end: ingest, embedding, routing, hybrid vector + BM25 search, and
the optional model-backed stages (query expansion, reranking, answer synthesis),
with compaction, int8 storage, backup/restore and a multi-library registry.

**dyprys is a distilled fork of [cyprys](https://github.com/mendelcyprys/cyprys),
built for an agent that operates the tool to search on a user's behalf.** It takes
cyprys's proven code — the full test suite passes unchanged — and leaves behind
the research apparatus, keeping a short operating manual in its place.

The measurements behind every default were taken in cyprys, on a 3,453-book /
284,627-chunk index: the answer is the first result about 6 times in 10 and
somewhere in the top five about 8 in 10, so results are meant to be read a handful
at a time; routing reads roughly 1% of the library for about one lost answer in
twenty-five. cyprys holds the full record and the reasoning; this repo holds the
tool.

## Documentation

- **`CLAUDE.md`** — the operating manual: which search to run for which kind of
  request, how to read a result, how to embed and manage a library. Short, and
  read automatically by an agent on load.
- **[cyprys](https://github.com/mendelcyprys/cyprys)** — the design argument, the
  measurement discipline, and the record of what was tried and reversed. Read it
  when you want to know *why*, not *how*.

## Getting started

```sh
git clone git@github.com:mendelcyprys/dyprys.git
cd dyprys
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

`-e` installs in editable mode, so edits to `src/` take effect without
reinstalling. `uv` is optional — `python3 -m venv .venv` and `pip` work
identically.

Check it worked:

```sh
dyp          # prints help
pytest       # all green
```

## Usage

```sh
dyp add ~/library                  # chunk every .txt found, record the boundaries
dyp add ~/library --ext .md,.txt   # widen what counts as a book
dyp add ~/library --chapters       # each directory is one book, files are chapters
dyp embed --model model.gguf       # resumable; Ctrl-C keeps everything computed
dyp route                          # profile each book for two-stage search
dyp ask "why do neurons fire"      # hybrid search, best passages first
```

Embedding is the one slow step — 5 passages a second, so a few hundred books is
an overnight job and the 5,000-book target is about a month. It is resumable at
the granularity of one passage, so stopping it costs nothing:

```sh
dyp embed --for 45m                # stop after three quarters of an hour
dyp embed --duty 80                # use 80% of the GPU, keep the machine usable
dyp watch                          # follow it from another terminal
```

`--duty` is not free and the measurement says so: 80% of wall time buys back a
responsive machine and costs 21% of the schedule. There is no thermal headroom
to reclaim — sampled over seven continuous minutes the rate does not decline.

### Asking better questions

```sh
dyp ask "..." --route              # 6.6x faster, about ten questions less accurate
dyp ask "..." -k 10                # the answer is in the top five 90% of the time
dyp ask "..." -c neuroscience      # one shelf, one author, or a glob
dyp ask "..." --full               # the whole passage, not the query-dense part
```

Two optional stages need a language model and are **off by default**, because
both cost seconds and the defaults are for typing at a prompt:

```sh
dyp ask "..." --expand gemma3:4b     # rephrase first: +9 answers in 110, ~11x slower
dyp ask "..." --summarise gemma3:4b  # draft an answer, quoting only what verifies
```

Nothing in `--summarise` is trusted. Every quotation is checked byte-for-byte
against the source and cut from the prose if it does not match; locations never
come from the model at all — it picks among numbered passages and the book,
chunk and offset are attached afterwards from the index.

### What the library has to look like

Plain text files, `.txt` by default. Pass `--ext .md,.text` to widen that. If
nothing matches, `add` says so, lists the extensions that *are* present, and
exits non-zero — it will not quietly report success having done nothing.

```
library/
  Some Book.txt                    ← one file, one book
  Another Book.txt
```

With `--chapters`, each directory named on the command line is **one book** and
the files inside it are its chapters, in filename order:

```
library/
  The Long Book/                   ← one book, four chapters
    chapter01.txt
    chapter02.txt
```

This is deliberately explicit rather than inferred. Guessing whether
`library/neuroscience/` is one book or a shelf of them would silently mis-model
the library, and the mistake would not show up until search results were wrong.

Extraction from PDF or EPUB happens upstream; dyprys indexes the text you give
it. If the library moves — a new disk, another machine — `dyp relocate OLD NEW`
rewrites the stored paths. Chunk offsets are into a file's bytes, so nothing
else changes and no re-embedding is needed.

### Many libraries, one installation

```sh
dyp library add neuro ~/indexes/neuro   # name a directory
dyp library list                        # what this installation knows
dyp library use neuro                   # make it the default
dyp -L neuro ask "..."                  # act on it by name
```

Which index a command uses is resolved in this order: `--data DIR`, then
`-L NAME`, then whichever library is the default, then `./data`. Explicit always
beats remembered, so a stale default cannot silently redirect a command that
named its target.

The registry holds names and paths only. Losing it loses nothing — every library
is still openable with `--data`.

**A library is one directory and nothing else.** The database, every model's
vector file and the routing profile all live inside it; nothing about it is
recorded anywhere but the registry entry, which is a name and a path. So testing
someone's backup and then throwing it away is exactly what it sounds like:

```sh
dyp restore theirs.tar.gz --into ~/tmp/theirs --sources ~/tmp/their-books --as theirs
dyp -L theirs eval ...
dyp library remove theirs --delete --yes   # the index goes; the text does not
```

`--delete` removes the index directory only. The source text is usually somewhere
you chose and may be shared with another library, so it is never deleted for you
— the command says where it is and leaves it.

### Moving the whole thing

```sh
dyp backup library.tar.gz
dyp restore library.tar.gz --into ~/index --sources ~/books --as theirs
```

An index is three things that have to travel together: the database, the vector
files, and the text the offsets point into. Vectors without their sources can
rank a passage but not show it; sources without vectors are hundreds of hours
from being searchable again. `backup` takes all three, and `restore` unpacks
them and repoints the paths, so nothing is re-embedded.

The database is copied through SQLite's online backup, so a backup taken while
`dyp embed` is running is still consistent. `--no-sources` makes a smaller
archive when the text already exists wherever it is going, and `--as NAME`
registers the result so it is immediately usable with `-L NAME`.

Several models can cover one library, at different widths and formats. The index
these numbers come from holds two over the same 284,627 passages —
EmbeddingGemma-300M at dimension 768 and jina-embeddings-v2-small-en at 512,
both int8, 220 MB and 214 MB. Each keeps its own vector file, its own coverage
and its own routing profile, and a backup carries all of them.

They are not interchangeable, and the difference is worth knowing before
choosing: the model nine times smaller builds its index in a sixth of the time
and *finds* the answer nearly as often (94/110 against 99/110 in the top five,
p = 1.000), but **ranks it worse** — 63/110 against 78/110 first, p = 0.023. An
agent reading five passages loses almost nothing; a person reading the first
loses fifteen questions in a hundred.

**Someone else's backup works on your models.** A model is identified by a hash
of its weights, not by its filename, so restoring an index and pointing `--model`
at your own copy of the same file reuses every vector in it.

If you *don't* have the weights, the archive says what they were. The manifest
records each model's filename, size, full SHA-256 and — if it was recorded — where
to obtain it, and can be read without unpacking anything:

```sh
tar xzOf theirs.tar.gz manifest.json
dyp models --needed                  # the same, from a restored index
dyp models --verify ~/Downloads/candidate.gguf
dyp models --source NAME hf:org/repo/file.gguf   # record where yours came from
```

`--verify` is the one that saves the afternoon. The wrong weights file does not
fail — it embeds the whole library again under a new identity, which looks
completely normal while it happens. Checking a candidate first costs one hash.

`add` is idempotent — a file whose contents have not changed is skipped, and a
file that merely *moved* keeps its vectors. `embed` can be stopped by `--limit`,
`--for 30m` or Ctrl-C, and always commits the batch in flight before stopping.

### Looking at what is there

```sh
dyp books                    # every book, with chunk and BM25 coverage
dyp books Kandel             # one book in full: sources, chunkings, coverage per model
dyp models                   # embedding models: coverage, disk, routing profile
dyp status                   # totals, and each chunking when there is more than one
dyp check                    # what drifted, and what work is outstanding
```

`check` is the one to run when something looks wrong: it reports source files
that moved or vanished, chunks awaiting embedding split into "copy" (near-free)
and "embed" (hours), whether the BM25 index and routing profile are current, and
how much dead space a re-ingest left behind. It never changes anything.

### Managing it

```sh
dyp models --drop NAME --yes # remove a model, its vectors and its centroids
dyp models --quantise NAME   # convert vectors to int8: a quarter the disk
dyp relocate OLD NEW         # the library moved; rewrite the stored paths
dyp backup library.tar.gz    # index, vectors and text in one archive
dyp restore library.tar.gz --into ~/idx --sources ~/books
dyp remove PATTERN --yes     # forget books
dyp compact --yes            # reclaim the ids they left behind
dyp lexical                  # rebuild the BM25 index
dyp route                    # rebuild the routing profile
```

Dropping a model is clean because each model's vectors live in their own file —
books, chunks and the BM25 index are untouched. Dropping a *book* is not offered:
its chunk ids sit inside a dense range that other offsets depend on, so removing
one is a compaction — `dyp remove` forgets the book, `dyp compact` reclaims the
ids across every store.

### Measuring it

```sh
dyp eval --compare           # vector vs BM25 vs hybrid, on the same questions
dyp eval --route             # the same, routed, with routing recall
```

Three question sets, and they measure different things:

| set | n | the answer sits in | measures |
|---|---:|---|---|
| `eval/questions.json` | 110 | a median of 81 books of 3,453 | end-to-end answer recall |
| `eval/questions-rare.json` | 28 | a median of 2 books | **book selection — routing** |
| `eval/questions-gutenberg.json` | 24 | a median of 1 book | routing on a *homogeneous* library |

Use the rare or Gutenberg set for anything touching routing. On the main set the
answer is in most of the library, so routing recall is largely answered by luck
— `dyp eval --route` therefore prints the chance baseline beside it, and prints
how often routing kept the book flat search itself chose.

That baseline is not a formality. At 117 books it was 92.3%, which meant a
reported 109/110 said almost nothing and was quoted for two build steps anyway.

Measurement is the part of this project most worth understanding before changing
anything: cyprys's `docs/dev/measurement.md` sets out five questions to ask of a
number, each of which caught a wrong conclusion that had already been written down
as a finding.

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
  evaluate.py    the harness every claim rests on
  text.py        the only place that reads a book's bytes back
  term.py        emphasis, extracts, bars — and none of it when piped
  cli.py         `dyp` resolves to dyprys.cli:main
tests/           pytest — 547 tests across 25 files
pyproject.toml   dependencies and the console-script entry point
```

**The split is by I/O, not by topic.** `chunk_bytes` touches nothing, so its
tests need no fixtures, no temp directory and no database.

Two runtime dependencies, `llama-cpp-python` and `numpy`. Add anything new to
`pyproject.toml` under `dependencies` — never with a bare `uv pip install`, or
the project will work here and fail on a fresh clone.

## Data

The index, vectors and source library live outside version control and are
gitignored (`*.sqlite`, `*.npy`, `data/`). Nothing that grows with the library
belongs in git.

## Provenance

Architecture informed by [qmd](https://github.com/tobi/qmd) (MIT, © 2024–2026
Tobi Lutke) — by diagnosing where its design runs out at book scale. No qmd code
is reused.
