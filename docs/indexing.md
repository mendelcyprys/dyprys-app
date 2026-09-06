# Building an index — reference

`add` → `embed` → `route`. Read this when setting up a new library or choosing an
embedding strategy.

## `dyp add PATHS…` — record the text

Never copies, moves, or modifies files. Stores `(path, offset, length)` and reads
by seek.

```sh
dyp add ~/books                 # every .txt under the folder, each file one book
dyp add ~/notes --ext .md,.rst  # widen the extensions
dyp add ~/books/kandel --chapters   # one directory = one book, files = chapters
dyp add ~/books --target 1200   # a second, finer chunking alongside the first
dyp add ~/books --deep          # re-hash instead of trusting size+mtime
```

- **Nothing matches** → exits non-zero and lists what *is* there (`--ext` to
  include it). It never reports success on an empty run.
- **`--chapters`** is explicit because guessing "one book or a shelf of forty"
  mis-models the library and only shows up later as misattributed results. It
  also limits edit damage — an edit only disturbs the file it is in (a 5 KB
  deletion kept 55% of vectors as one file, 94% across 8 chapters, 99% across 40).
- **Idempotent** — an unchanged file is skipped; a *moved* file is recognised by
  content and its path updated with no re-embedding; an *edited* file is re-read
  and only its changed passages need re-embedding (survivors keep their vectors,
  matched by content not position).

### Detecting bad extraction — check this on every add

A book whose text extracted badly (spaces lost, `Thepresentchapter…`) can never
match a query and costs the same to embed as good text. `add` warns, and
`dyp check` re-checks any time:

```
3 book(s) whose text has no word boundaries — 436 chunks
  the PDF's font map defeated the extractor. Embedding these costs the same as
  real text and can never match a query; re-extract them or `dyp remove` them.
```

The test is longest-word length (real English ~12 chars at the 90th percentile;
lost-space text ~80). **The only fix is re-extracting with a different tool, or
`dyp remove`.** Nothing downstream repairs it.

### Chunk size (`--target`)

Default ~3,600 bytes (~a page, ~900 tokens). Two constraints before changing it:

- **The model must fit it.** Most small embedding models cap at 512 tokens and
  would silently truncate 99% of default passages — that measures truncation
  damage, not retrieval.
- **A library can hold several chunkings.** Adding at a new `--target` adds a
  second passage set; each model binds to one on its first embed. If split more
  than one way, `dyp embed` refuses to guess and lists the choices.

## `dyp embed` — turn text into vectors

The one slow step; run once. Resumable at single-passage granularity — a vector
is written and flushed *before* the count that claims it, so an interruption
re-embeds one batch, never corrupts (verified with SIGKILL).

```sh
dyp embed --model FILE.gguf   # remembered after the first run
dyp embed --for 45m           # stop after wall-clock time
dyp embed --limit 5000        # stop after N passages
dyp embed --duty 80           # work 80% of the time, keep the machine usable
dyp embed --int8              # 1 byte/dim instead of 4 — ¼ the disk, no measured loss
dyp embed -c PATTERN          # embed one shelf now, the rest later
dyp embed --target 3600       # which chunking, when split more than one way
```

All bounds take effect **between** batches, so the batch in flight always
finishes and commits. One writer per index (file lock) — a second `embed`
refuses rather than burning the same hours twice.

**Never start a full embed unasked** — it can run for days. A bounded test run on
a few small books is fine.

### Throughput and model choice

Measured on an Apple M4, per embedding model:

| model | dim | passages/s | 285k passages | ranking |
|---|---:|---:|---:|---|
| jina-embeddings-v2-small-en | 512 | 30.7 | ~3 h | slightly worse |
| nomic-embed-text-v1.5 | 768 | 6.0 | ~13 h | — |
| EmbeddingGemma-300M | 768 | 5.0 | ~19 h | best |

Rate depends on passage length too (a 300M model runs ~6/s on 3,600-byte
passages, ~20/s on 1,200-byte), so a rate without a passage size means little.

**Choosing:** the small model (jina) builds several times faster and *finds* the
answer nearly as often (top-five 94/110 vs 99/110, not significant) but ranks it
first less often (63/110 vs 78/110, significant). Prefer it for a large or
growing library and pair with `--rerank`; prefer the larger when the corpus is
small and the first result matters most. Several models coexist in one index,
each with its own vectors, coverage and routing profile — choose per query with
`--model NAME`.

**Those figures are one corpus, and the shape of the trade can invert.** They
come from a 117-book English non-fiction library. On a translated religious-law
corpus the same pair came out the other way round in a small (6-question) check:
jina ranked first more often when it found the answer at all, but *missed* two
outright — one at rank 13, one absent from 40 results — where the larger model
found all six. Recall and precision traded places. So treat the numbers above as
evidence that the two differ, not as a prediction for your library; if the
choice matters, measure it on your own questions. Whichever you pick,
`--rerank` improved both.

### Smaller storage

- `--int8` (or `dyp models --quantise NAME` after the fact) — ¼ disk, no measured
  loss. Set only when a model is first registered, or convert later.
- `dyp models --truncate 512` — for models trained for it (EmbeddingGemma is),
  keep the first N dims. 768→512 is a third less disk for no significant loss;
  256 halves it again for ~8 questions in 100 at rank one (top-five unaffected).

### Running it over sessions, keeping the machine awake

```sh
for i in {1..12}; do caffeinate -i dyp embed --for 45m; done
```

`caffeinate -i` stops idle-sleep; without it a laptop sleeps mid-run (harmless —
it resumes — but a 45-min run then spans hours of wall clock). `--duty 80` gives
~79% of full speed for 80% of the time; there is no thermal headroom to reclaim.
`taskpolicy -b dyp embed` is a non-adjustable ~88%-speed alternative.

## `dyp route` — enable the fast path

Profiles each book into spherical k-means centroids so `--route` can skip most of
the library. Takes seconds. **Re-run after adding books** (`dyp check` says when
it is stale).

## Watching a run

`dyp watch` (from any terminal, or over ssh — reads only) shows the whole job and
the book in flight, because progress is committed to the database as it goes.
Progress is measured against the chunking the model is bound to, and with more
than one model it follows the unfinished one and names it.

An embed loads its weights *before* it locks the index, so for the first seconds
of a run there is nothing to attach to. `watch` waits `--wait` seconds (default
15) for a run to appear rather than reporting that none exists; `--wait 0` checks
once and returns.

`dyp history` afterward records every run with a wall clock *and* a working clock,
so a run that spanned a laptop sleeping shows the sleep, not an impossible rate.
