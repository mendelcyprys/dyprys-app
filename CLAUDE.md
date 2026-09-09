# dyprys — operating manual

You are running `dyp`, a local semantic search tool over a personal library of
book-length texts. Your job is to **find the passage that answers what the user
asked and return it to them** — not a list of books, the passage itself, with
where it came from.

This file is loaded every session and kept short: it tells you *which search to
run when*. For exact flag syntax, `dyp COMMAND -h`. For fuller detail on anything
below — the complete flag list, the `--json` schema, what a specific error means,
how to choose an embed model — the `docs/` pages listed at the end go deeper; read
one only when a task needs it.

---

## Before the first search of an unfamiliar library

Run **`dyp status`**. If it names a notes file, **read it before searching.** A
corpus carries facts the index cannot tell you and you will not guess: which
shelf `-c` cuts along cleanly and which it does not, the words this collection
uses for the thing you are about to call something else, two subjects whose
vocabulary overlaps enough that a query for one returns the other, a title that
means two different books. Nobody discovers these from the results — they only
notice, several bad searches later, that they should have.

**`dyp status` reports only the library it is run against**, so a bare `dyp
status` speaks for the default library and stays silent about the notes of every
other one. Run `dyp -L NAME status` for the library you are about to search — or
read the `notes:` line of `dyp library list`, which names them all at once.

There may not be one. Its absence is not a problem; skipping it when it exists
is, and if a session teaches you such a fact, offer to add it.

---

## The one habit that matters most

**Read the top five, not the top one, and quote from them.** The answer is the
very first result about 6 times in 10, and somewhere in the top five about 8 times
in 10. So a single result is often wrong when five would have been right. Ask for
`-k 5` (the default) or more, read them, and quote the one that actually answers —
its rank is a hint, not a verdict.

---

## Choosing the search for the request

| the user… | do this | why it helps |
|---|---|---|
| remembers a phrase / exact words | pass the words **verbatim**, don't paraphrase | the keyword half finds literal text; rewording destroys the one thing it had |
| describes a thing they can't name | plain description, ordinary words | the meaning half is built for this |
| names a rare term / proper noun | include it **exactly**, spelling matters | a name only ever matches literally |
| wants a quick look | add `--route` | ~6× faster, reads ~1% of the library, costs a little accuracy |
| needs the best possible answer | **no** `--route` | routed drops roughly one answer in twenty-five |
| asks in everyday words about a technical library | add `--expand MODEL` | rewrites the query into the library's vocabulary; ~4 s |
| wants a written answer, not passages | add `--summarise MODEL` | drafts prose; every quote is checked against the source |
| means one shelf / author / book | add `-c PATTERN` | matches title, path or the name you gave it, case-insensitive |
| means several books they picked | repeat `-c` | the patterns are a union; each must match something |
| wants the right passage **ranked first** | add `--rerank` | a cross-encoder rescores what was already found; buys rank, not recall |

Defaults are deliberately fast and literal-safe. Reach for `--expand` and
`--summarise` only when the request is worth the seconds they cost; they are off
by default for that reason.

**If the index holds more than one model, every one of these needs `--model
NAME`** — search refuses rather than guess which vectors to answer from, so on
such an index the whole table above fails until you say. You never have to type
the full name: **any unique part of it resolves** (`--model gemma`), `dyp models`
prints a handle you can paste, `dyp models --name NAME ALIAS` sets a short alias
once and for all, and `$DYPRYS_MODEL` sets a default for the session. Check with
`dyp models` before assuming an index has only one.

**Do not add `--expand` when the answer is a rare literal** (a name, a place, an
odd spelling). It bridges vocabulary, and a rare token has no vocabulary to
bridge — it only adds noise.

**`--route` can only reach books that have been profiled.** A book embedded
since the last `dyp route` has no profile, so routed search cannot return it at
*any* rank — and it still returns a full `k` results, so nothing about the output
says a part of the library was skipped. That is a different and much larger risk
than the one-in-twenty-five above, which assumes a current profile. Re-run
`dyp route` after every `add` + `embed`; `dyp check` says how many books are
waiting, and search warns when it is routing around some.

**`--rerank` needs a second model, and bare `--rerank` errors rather than
falling back to anything.** It takes a cross-encoder GGUF — not an ollama chat
name — via `--reranker PATH`, `$DYPRYS_RERANKER`, or a default stored in the
index. **Not every `.gguf` will do**: an embedding model is refused, because
llama.cpp loads one as a reranker without complaint and returns numbers that are
not relevance. The refusal reads the file's own declared `pooling_type`, so it
is a fact rather than a guess from the name — and it fires when the default is
*set*, not weeks later when a search uses it.

**No optional model is remembered from being used.** The expander, summariser
and reranker are each stored only when set on purpose:
`dyp models --summariser gemma3:4b`, `--expander`, `--reranker PATH.gguf`
(`none` forgets one). `dyp models` shows what a library currently remembers, and
`dyp history` the setting of it. Until one is set, every search must name the
model or say `$DYPRYS_*`; if none is set and the user has not named one, ask
rather than guessing a path.

**Reach for `--rerank` whenever the rank matters and you can spare the seconds.**
It rescores results already found, so it cannot recover an answer the search
missed — it buys rank, not recall, and rank is usually the complaint. Since
`cos` is a poor guide to which of five results is right, this is the cheapest
correction available. How much it gains varies by corpus; `docs/searching.md`
carries the measurements and says what each was measured on.

**Do not stack `--expand` and `--rerank`.** Measured, they are substitutes, not
complements: together they recover the same answers as the better one alone, at
the sum of the costs. Pick one — reranking finds a few more answers, expansion is
several times faster.

---

## Reading a result

```
1. [cos 0.52 · vec 1 · phrase 1] Principles.of.Neural.Science  (chunk 21482)
   … the passage text …
   ~/books/Principles.of.Neural.Science.txt:523017
```

- **`cos`** — closeness in meaning, 0–1. Comparable **only within one search**,
  and a higher number is *not* a better answer. Show it, don't rank on it.
- **`vec` / `phrase` / `words`** — which half found it and at what rank. `phrase`
  means the passage holds the query's words in order (a literal hit); `words`
  means it shares vocabulary; `vec` means it matched on meaning.
- **the last line** — the file and byte offset. This is what you hand the user so
  they can open the source themselves. It is also the only thing that says
  *which* book: titles are not unique, and one library can hold several distinct
  books under the same title on different shelves. **Attribute from the path,
  never from the title alone** — `dyp add` warns when titles collide, and
  `dyp books PATTERN` shows what shares one.

When you need the data rather than the display, use `--json` (see below).

→ **`docs/searching.md`** — the full flag list, the `--json` schema, `--summarise`
trust rules, and how to reason about a search that returns nothing or the wrong
passage.

---

## Administering a library

Occasional, but yours to do when asked. Never do the slow ones unprompted.

**Add text** — `dyp add DIR`. Plain `.txt` by default; `--ext .md,.txt` to widen;
`--chapters` when each directory is one book and its files are chapters.
`--target BYTES` sets the chunk size, and this is the **only** place it is set:
re-adding paths already in the index at a different target is not a re-ingest —
it records a second chunking alongside the first ("rechunked"), disturbing
nothing and losing no vector. Embedding that second chunking then needs a
**second model**, since a model embeds one chunking and cannot be moved.

**Embed** (turns text into vectors — the one slow, expensive step):
- **You need a model file first.** An embedding model is a local GGUF file passed
  as `--model PATH.gguf` (remembered after the first run) or set in
  `$DYPRYS_MODEL`. If none is configured and none is remembered in the index, do
  not guess a path — ask the user where their model is, or look for `*.gguf` under
  `~/.cache` (e.g. `~/.cache/qmd/models/`). `dyp models --json` shows what an
  index already knows.
- Choose the model deliberately. A **light model** (e.g. jina) builds the index
  several times faster and ranks answers slightly worse — pair it with
  `--rerank` to recover the ranking. A **heavier model** ranks best out of the
  box but costs far more to build. Prefer the light model when the library is
  large or growing; the heavy one when the corpus is small and rank-1 matters.
- It is **resumable** — stopping loses nothing. `--for 45m` bounds a session;
  `--duty 80` leaves the machine usable.
- **Never start a full embed unasked.** It can run for days on a large library.
  A bounded test run on a few books is fine.
- A model is bound to one chunk size on its first embed and cannot change it.
- Run `dyp route` afterwards so search can skip most of the library — **every
  time**, not just the first. Books embedded since the last run are unreachable
  under `--route` until you do.

**Several libraries** — `dyp library add NAME DIR`, `dyp library use NAME`, or
`-L NAME` per command. `-L` and `--data` are **global flags and go before the
command** — `dyp -L NAME ask "…"`, not `dyp ask "…" -L NAME`, which fails with
`unrecognized arguments`.

**Shelves** — `dyp shelves`. A library holds shelves; a shelf holds books. A
shelf is the directory the text was added from, derived rather than stored, and
it is the level `-c` has always cut along: `dyp ask "…" -c papers/` is one shelf.
`dyp shelves` names them with their sizes, which on a 3,453-book library is
three lines instead of three thousand — and knowing which of the three a
question belongs to is usually worth more than any flag below.

**Set a book or a shelf aside** — `dyp books PATTERN --aside`,
`dyp shelves --aside SHELF` (`--restore` puts them back). This takes them out of
**every** search while leaving every vector, passage and BM25 row exactly where
it is: nothing is deleted and putting them back costs one command. It is the
right answer for a book whose extraction lost its word boundaries, or a shelf
that swamps every answer — `dyp remove` throws away the hours that embedded it
and this does not. A set-aside book is still listed by `dyp books`, marked, and
`-c` still matches it: a search says so in `warnings` and refuses outright if
the whole scope is set aside.

**Name a book, and note what it is** — `dyp books PATTERN --label NAME
--note TEXT`. The label is a display name and something `-c` can match; the file
is not renamed and keeps its own title, because ingest derives that and rewrites
it whenever the file moves. The note rides on every result from that book — the
place for what the index cannot tell you and a reader will not guess: which
edition, why the sentences run together, whose vocabulary this is. Neither is
identity; the path still is. **This is the fix for colliding titles** — one
library can hold two different works called Arakhin, and naming one is how you
tell them apart and how you scope to just one.

**Health & upkeep** — `dyp status` (totals, what to do next), `dyp check` (what
drifted, what work is outstanding — never changes anything), `dyp books` /
`dyp shelves` (what is in the library). Then `backup` / `restore` / `relocate` /
`remove` + `compact` / `lexical` as needed.

**Taking things out** has two settings at every level, and the difference is
whether the embedding survives:

| | reversible | for good |
|---|---|---|
| book | `dyp books PATTERN --aside` | `dyp remove PATTERN --yes` (+ `compact`) |
| shelf | `dyp shelves --aside SHELF` | `dyp remove SHELF --yes` |
| library | `dyp library remove NAME` | `dyp library remove NAME --delete --yes` |

**No source text is ever deleted by any of them** — the right column removes the
index's memory of the books, or the index directory, and never the files.
Prefer the left column: it costs nothing, and it is one command back.

→ **`docs/indexing.md`** (add, embed, model choice, detecting bad extraction),
**`docs/libraries.md`** (registry, several models, backup/restore, moving),
**`docs/maintenance.md`** (check, remove/compact, what each command keeps).

---

## Hard rules

- **Return where text came from, always.** A passage without its file and offset
  is not a finished answer.
- **Never invent a citation.** Only quote text `dyp` actually returned; the byte
  offsets are real, use them.
- **Never start a full embedding run without being asked.** It is long and costly.
- **Say when a search failed, and know how to tell.** Search *always* returns
  `k` results — the meaning half fills them even when the library holds nothing
  relevant — so exit 0 is not "found it". A miss looks like: the top hit is
  `vec`-only (no `phrase`/`words` agreement), its `cos` is low *for this search*,
  and — the deciding test — **its `text` does not actually address the question**.
  Read the top passage before trusting its rank. When a topic may simply be
  absent, `--summarise` is the surest check: it says "these passages do not answer
  the question" outright and retries once with a rephrasing before giving up. A
  refusal that survives rephrasing is strong evidence the library lacks it; a
  single weak result is not, so retry with different words first.
- **Prefer the fast path for casual questions, the exact path when it matters.**
  Tell the user which you used when it affects how much to trust the result.

---

## Getting the machine-readable form

Parse `--json`, don't scrape the pretty output (it carries ANSI codes and
middle-elided titles). Available on `ask` **and on every read-only command** —
`status`, `check`, `books`, `models`, `library list`, `history`, `asked` — so you
can drive both searching and administration structurally:

- `dyp status --json` → is it embedded, and how far? (`.models[].coverage`, which
  is against `.models[].live_chunks` — the chunking that model embeds, not the
  whole library, so a finished model reads 1.0)
- `dyp check --json` → anything drifted? (`.drift.clean`, `.models[].outstanding`)
  and is routing whole? (`.models[].routing.unprofiled_books` — books `--route`
  cannot reach at all; non-zero means run `dyp route`)
- `dyp books --json` / `dyp models --json` → what is here, per-book / per-model
  (`set_aside` is non-null on a book no search will return; `shelf` is which
  shelf it sits on)
- `dyp shelves --json` → the shelves, their sizes and how many books on each are
  set aside, plus the `root` their paths are relative to
- `dyp library list --json` → the libraries, sizes and defaults
- `dyp history --json` → embed runs (rate, how each ended) and index operations
- `dyp asked --json` → what this library was already asked and what came back
  (`dyp asked N --json` for one in full) — cross-session memory of prior searches

`dyp ask "…" --json` returns each result as
`{rank, chunk_id, book, book_note, chapter, path, offset, cos, provenance,
state, text}`. `book` is the name someone gave the book if they gave one and the
derived title otherwise; `book_note` is what they wrote about it, usually null —
read it, it is there because something about that book is worth knowing.
`provenance` is the rank signal (e.g. `"vec 1 · phrase 1"`); `text` is `null` when
the passage could not be proved against its stored hash, and `state` says why —
**never quote a passage whose `text` is null**. Full schema in `docs/searching.md`.

If a JSON tool like `jq` is on the machine, piping through it is fine
(`dyp status --json | jq .models[0].coverage`); the output is plain JSON on
stdout, one object, exit code 0 on success and 1 on an empty/no-result listing.

---

## Deeper reference — read on demand

Short pages, consulted only when a task needs more than the rules above. Do not
read them pre-emptively; each is a lookup for one kind of work.

| page | read it when |
|---|---|
| `docs/searching.md` | a search misbehaves, or you need the `--json` schema or the `--summarise` trust rules |
| `docs/indexing.md` | building a new index — `add`, `embed`, chunk size, choosing a model, spotting bad extraction |
| `docs/libraries.md` | several libraries or models, backup/restore, moving text, model-by-weights and `--verify` |
| `docs/maintenance.md` | `check`, removing books, compaction, and exactly what each command keeps vs removes |
| `docs/serving.md` | `dyp serve` — the HTTP API, its payloads, jobs and status codes |
| `docs/troubleshooting.md` | an error string or symptom — a lookup table from what you saw to what to do |
