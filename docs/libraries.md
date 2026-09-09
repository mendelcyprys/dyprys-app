# Libraries, models, backup — reference

Managing several indexes, several models in one index, and moving them between
machines.

## Which index a command uses

Resolved in order, first wins:

1. `--data DIR` — an explicit path
2. `-L NAME` — a registered name
3. the default library
4. `./data` in the current directory

**Explicit always beats remembered** — a command that named its target can never
be redirected by a stale default.

**A registered name pointing nowhere is an error, not an invitation.** Only
`dyp library add` and `dyp restore` create an index; `-L neuro status` on a
missing path fails rather than writing an empty index into the mount point.

## The registry

```sh
dyp library add neuro ~/indexes/neuro   # name a directory (first one becomes default)
dyp library list                        # names, sizes, models; * marks the default
dyp library use neuro                   # change the default
dyp library remove neuro                # forget the name only; files untouched
dyp library remove neuro --delete --yes # also delete the index files (NOT the source text)
```

Holds names and paths only — a small JSON file, written atomically and moved
aside if ever found corrupt. Losing it costs the names and nothing else; every
library still opens with `--data DIR`.

`dyp library list --json` emits the libraries structurally
(`[{name, path, default, exists, notes, books, chunks, models[]}]`) for
programmatic use.

## Notes on a library

Drop a **`NOTES.md`** (or `README.md`) in the library directory and `dyp status`
prints a line pointing at it, `dyp library list` names it, and both `--json`
forms carry it as `notes`. dyprys never reads, parses or writes the file — the
problem it solves is that a reader does not know to look.

Worth putting in one: which shelf `-c` cuts along cleanly and which it does not,
terms the corpus renders differently from how you would search for them, subjects
whose vocabulary overlaps enough that a query for one returns the other, and any
title that means two different books. These are facts about the text that the
index cannot expose and that each new reader otherwise rediscovers — or doesn't,
and searches worse without knowing it.

```
~/books/NOTES.md          # dyp status → "notes on this library — read it first"
```

## Several libraries over shared text

Two indexes can point at one folder of text; nothing is copied, so it costs only
the index. Each keeps its own passages, vectors and progress:

```sh
dyp -L work     add ~/books/shared
dyp -L personal add ~/books/shared
```

## Several models in one library

Usually better than two libraries over the same books — the text is read and
split once.

```sh
dyp models                                   # coverage, disk, names, per model
dyp models --name <long-hash-name> jina      # give one a short name to type
dyp ask "..." --model jina                   # search with one of them
```

## Backup and restore

An index is three things that must travel together: the database, the vectors,
and the source text the offsets point into. Any two without the third is not a
library.

```sh
dyp backup lib.tar.gz                # all three, one file
dyp backup index-only.tar.gz --no-sources   # smaller; needs the text present at the far end
dyp restore lib.tar.gz --into ~/idx --sources ~/books --as neuro
```

- Safe to `backup` **while embedding** — the database is copied through SQLite's
  online backup; vectors may run slightly ahead of the progress counts, which is
  harmless (those passages just re-embed).
- gzip because a mostly-unwritten vector file is nearly all zeros (unwritten rows
  compress ~1000×), so a half-embedded library backs up cheaply.
- `restore` repoints paths automatically and refuses to write into a directory
  that already holds an index (a silent merge would be unrecoverable).
- **Model weights are not in the archive** (they are an input, like a compiler).

## Models are identified by their weights, not their filename

A model's identity is a hash of its weights, so someone else's restored index
reuses *your* copy of the same GGUF automatically. The consequence: **the wrong
weights file does not fail** — the index silently re-embeds itself under a new
identity, looking normal for hours. Guard against it:

```sh
dyp models --needed                  # what this index requires (name, size, SHA-256, source)
dyp models --verify FILE.gguf        # is this the right file? costs one hash
dyp models --source NAME hf:org/repo/file.gguf   # record where yours came from
```

`--verify` before a restore-and-embed is the check that saves hours.

## Moving the text

Stored paths are absolute. If the text moves:

```sh
dyp relocate /old/prefix /new/prefix   # rewrites the prefix, checks it is readable
```

No vector is recomputed — offsets are byte positions into a file. A single book
that moved needs nothing: `dyp add` recognises it by content and updates the path.

## Environment variables

| variable | sets |
|---|---|
| `DYPRYS_MODEL` | embedding model, if not remembered in the index |
| `DYPRYS_EXPANDER` | model for `--expand` |
| `DYPRYS_SUMMARISER` | model for `--summarise` |
| `DYPRYS_RERANKER` | cross-encoder for `--rerank` |
| `NO_COLOR` | turn off colour and emphasis |

The three optional models can also be remembered per library
(`dyp models --expander …`), which is usually easier than an env var.
