# dyprys — operating manual

You are running `dyp`, a local semantic search tool over a personal library of
book-length texts. Your job is to **find the passage that answers what the user
asked and return it to them** — not a list of books, the passage itself, with
where it came from.

This file tells you *which search to run when*. For exact flags, `dyp COMMAND -h`.
For why any of it is true, that reasoning lives in the sibling `cyprys` repo; you
do not need it to operate well.

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
| means one shelf / author / book | add `-c PATTERN` | matches title or path, case-insensitive |

Defaults are deliberately fast and literal-safe. Reach for `--expand` and
`--summarise` only when the request is worth the seconds they cost; they are off
by default for that reason.

**Do not add `--expand` when the answer is a rare literal** (a name, a place, an
odd spelling). It bridges vocabulary, and a rare token has no vocabulary to
bridge — it only adds noise.

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
  they can open the source themselves.

When you need the data rather than the display, use `--json` (see below).

---

## Administering a library

Occasional, but yours to do when asked. Never do the slow ones unprompted.

**Add text** — `dyp add DIR`. Plain `.txt` by default; `--ext .md,.txt` to widen;
`--chapters` when each directory is one book and its files are chapters.

**Embed** (turns text into vectors — the one slow, expensive step):
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
- Run `dyp route` afterwards so search can skip most of the library.

**Several libraries** — `dyp library add NAME DIR`, `dyp library use NAME`, or
`-L NAME` per command.

**Health & upkeep** — `dyp status` (totals, what to do next), `dyp check` (what
drifted, what work is outstanding — never changes anything), `dyp books` (what is
in the library). Then `backup` / `restore` / `relocate` / `remove` + `compact` /
`lexical` as needed.

---

## Hard rules

- **Return where text came from, always.** A passage without its file and offset
  is not a finished answer.
- **Never invent a citation.** Only quote text `dyp` actually returned; the byte
  offsets are real, use them.
- **Never start a full embedding run without being asked.** It is long and costly.
- **Say when a search failed.** If nothing good comes back, retry with different
  words before concluding the library lacks the answer — a miss under one phrasing
  is not a miss under all.
- **Prefer the fast path for casual questions, the exact path when it matters.**
  Tell the user which you used when it affects how much to trust the result.

---

## Getting the machine-readable form

`dyp ask "…" --json` returns each result as
`{chunk_id, book, path, offset, cos, vec_rank, bm25_rank, text}` — parse this
rather than scraping the pretty output when you are acting on results
programmatically.
