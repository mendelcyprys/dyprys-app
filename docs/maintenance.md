# Upkeep and inspection — reference

The read-only inspection commands, and the lifecycle commands that change an
index. Most days nothing here is needed.

Every read-only command takes `--json` — `status`, `check`, `books`, `models`,
`history`, `asked` (and `library list`) — so parse that instead of the human
tables when driving the tool programmatically. Each prints one JSON object on
stdout, safe to pipe through `jq` if it is on the machine.

## `dyp check` — what has drifted

Compares the index against the filesystem and against every model. **Reports,
never acts** — a file missing because a drive is unmounted must not silently
discard hours of embedding.

```sh
dyp check           # trusts size + mtime
dyp check --deep    # re-hash every file; run before discarding any work
```

What it may report, and the action:

| report | meaning | action |
|---|---|---|
| files missing | moved, or a drive unmounted | `dyp relocate OLD NEW`, or mount it |
| files changed | edited since indexed | re-`dyp add`; survivors keep vectors |
| chunks to copy | vectors that survive an edit | next `dyp embed` carries them (seconds) |
| no word boundaries | bad extraction | `dyp books … --aside` while you re-extract; `dyp remove` only once you are sure (see indexing.md) |
| routing stale | books added since `dyp route` | re-`dyp route` (seconds) |
| BM25 incomplete | keyword index behind | `dyp lexical` |

## `dyp status` — the short version

A summary that ends with the single command that comes next, whatever state the
library is in. The one to run when unsure.

## `dyp books [PATTERN]` — what is in there

```sh
dyp books           # every book: chunks, files, BM25 coverage
dyp books Kandel    # matching books in full: sources, chunkings, coverage per model
```

## `dyp shelves` — the level between a library and a book

```sh
dyp shelves                 # every shelf: books, chunks, size, how many set aside
dyp shelves --json
```

A shelf is the directory a book's text sits in, derived from the book keys
rather than stored, so it cannot disagree with the filesystem. It is the level
`-c` has always cut along — `dyp ask "…" -c papers/` is one shelf — and the one
worth looking at first: `neuro` is 3,453 books and three shelves, and which of
the three a question belongs to matters more than any search flag.

The root the paths are relative to is the deepest directory every book shares,
which is **not** the registered library path: an index often lives beside its
text rather than above it.

A book is on exactly one shelf. `papers` and `papers/old` are two shelves rather
than a parent and a child, so that setting one aside means one thing.

## Setting books aside — removal you can undo

```sh
dyp books PATTERN --aside      # out of every search; nothing is deleted
dyp books PATTERN --restore    # back into the library
dyp shelves --aside SHELF      # the same, a whole shelf at a time
dyp shelves --restore SHELF
```

Every vector, passage and BM25 row stays exactly where it is and costs nothing
to keep; what changes is that no search reads them. `search.embedded_ranges` is
the single place every retrieval path derives its scope from — the vector scan,
BM25's OR of words, and BM25's exact-phrase attempt, which otherwise
deliberately escapes `-c` — so one filter there is a filter everywhere.

Reach for this when a book's extraction lost its word boundaries (it can never
match a query and costs a full share of every scan) or when a shelf swamps every
answer. `dyp remove` would throw away the hours that embedded it; this does not,
and `--restore` is instant.

A set-aside book is still listed by `dyp books`, marked, because one nothing
shows is one nobody can put back. `-c` still matches it: a search that scopes to
some set-aside books says so in `warnings`, and one whose whole scope is set
aside refuses rather than answering from elsewhere.

## Removing books and reclaiming space

```sh
dyp remove PATTERN          # PREVIEW: prints exactly what it would take, refuses
dyp remove PATTERN --yes    # actually forget the books
dyp compact --yes           # reclaim the dead ids across every store
```

- `remove` without `--yes` **is** the dry run — there is no separate flag. It
  lists the books and chunk count and notes the ids become dead space until
  compacted. **Source files are never touched.**
- `compact` closes the gaps across vectors, passage records, offsets and the
  keyword index at once, in place (no temp copy of a large file). On a real index,
  removing 11 books and reclaiming 3,704 ids took 0.7 s, and re-embedded passages
  matched their stored vectors exactly afterward.
- **Do not compact while vectors are waiting to be carried across an edit** —
  `dyp check` says if any are; compacting would move rows those carries point at.

## Rebuilding parts that go stale

```sh
dyp route      # after adding books — re-profile for the fast path (seconds)
dyp lexical    # rebuild the keyword index from source files (rarely needed)
```

`lexical` is built as you add; it is only for recovering if it ever falls behind.

## What each command removes, and keeps

| command | removes | keeps |
|---|---|---|
| `dyp books PATTERN --aside` | nothing — the books leave every search | every vector, passage and BM25 row; `--restore` undoes it |
| `dyp remove PATTERN` | book records; their passages become dead space | every source file, every other book, all vectors |
| `dyp compact` | dead passage ids, the tail of each vector file | every live vector |
| `dyp models --drop NAME` | one model, its vectors, its profile | books, passages, keyword index, other models |
| `dyp library remove NAME` | the registry name only | everything; still opens with `--data` |
| `dyp library remove --delete` | the index directory | the source text |

**Nothing here deletes a source file.** Text is read by seek and never copied in,
so no command owns it.

## The record of what has happened

```sh
dyp history            # every embed run (rate, how it ended) + index operations
dyp history -n 40
dyp asked              # every question asked, what came back, which models
dyp asked 7            # one in full, with its answer
dyp asked --find myelin
dyp asked --forget 2026-08-01   # drop entries older than a date (or `all --yes`)
```

`history` records both a wall clock and a working clock per run, so a run that
spanned a sleeping laptop shows the sleep rather than an impossible rate.
