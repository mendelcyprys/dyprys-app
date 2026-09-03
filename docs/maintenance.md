# Upkeep and inspection — reference

The read-only inspection commands, and the lifecycle commands that change an
index. Most days nothing here is needed.

`status`, `check`, `books` and `models` all take `--json` — parse that instead of
the human tables when driving the tool programmatically. Each prints one JSON
object on stdout (exit 0, or 1 for an empty listing), safe to pipe through `jq` if
it is on the machine.

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
| no word boundaries | bad extraction | re-extract or `dyp remove` (see indexing.md) |
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
