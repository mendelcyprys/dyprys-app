# Troubleshooting — error to action

Scan for the symptom or the error string; each row is what it means and what to
do. `dyp check` is the first thing to run for anything unclear — it compares the
index against the filesystem and every model and reports without changing
anything (`dyp check --deep` re-hashes rather than trusting timestamps).

## Search returns nothing or the wrong thing

| symptom | cause | action |
|---|---|---|
| `nothing embedded yet — run dyp embed first` | text added, no vectors | `dyp embed` (check `dyp status`) |
| `no passage matched` | lexical-only search found no words, or `-c` matched nothing | rephrase; drop or widen the scope |
| `-c` returns nothing | pattern typo | `dyp books PATTERN` shows what it matches |
| wrong book, plausible passage | `--route` narrowed past it | drop `--route` and retry |
| a book you know is embedded never appears under `--route` | it was embedded after the last `dyp route`, so it has no profile and stage 1 cannot reach it | `dyp route`; `dyp check` counts the books waiting, and search warns when it is routing around any |
| `this index has N models` | several models, none named | `--model` with any unique part of a name; `dyp models` lists handles, `dyp models --name NAME ALIAS` shortens one |
| only result is wrong | reading just the first | `-k 10`; answer is first ~6/10, top-five ~8/10 |
| a low `cos` with `words N` | literal match, not topical | expected — the passage shares words without being about the subject |
| one/two-word query, thin results | too short to form a phrase | add words, or quote an exact term |

## A passage will not display

The text is verified against a stored hash before display; an unprovable passage
is withheld, never shown stale.

| message | meaning | action |
|---|---|---|
| `file was edited and this passage is no longer in it` | text changed since indexing | re-`dyp add` the book; survivors keep vectors |
| `file was edited above this passage; found intact further on` | nothing wrong — found at a moved offset | re-`dyp add` to tidy offsets (optional) |
| `file is gone or its drive is not mounted` | source unreachable | mount it, or `dyp relocate OLD NEW` |

## Embedding

| symptom | cause | action |
|---|---|---|
| seems stuck | large book, long stretch on one title | `dyp watch` — if the counter advances, it is working |
| far below 3–6 passages/s (300M model) | GPU contention | stop whatever else is using the GPU |
| a `--for 45m` run took hours | the laptop slept | `caffeinate -i dyp embed --for 45m`; `dyp history` shows the sleep |
| `another dyp embed is running against this index (pid N)` | one-writer lock | wait for it; two runs would burn the same hours twice |

## The index will not open

| message | meaning | action |
|---|---|---|
| `index is schema vN, this build is vM` | index newer than the build (or too old to migrate) | use a matching/newer build; migrations are forward-only |
| `no index at PATH` | a registered name points nowhere | typo, stale entry, or unmounted drive — `dyp library list`; dyp never creates an index just to look at one |

## Models

| message / symptom | meaning | action |
|---|---|---|
| `weights /path/x.gguf MISSING` | the GGUF moved; vectors are fine | `dyp embed --model /new/path` (path is then remembered) |
| a finished model looks unfinished, or >100% | old build with a fixed counter bug | update; coverage is measured against the bound chunking |
| wrong results after a restore | possibly the wrong weights re-embedded silently | `dyp models --verify FILE` **before** embedding a restore |

## Optional models (expand / summarise / rerank)

| message | action |
|---|---|
| `no expansion model. Pass one: --expand MODEL or set $DYPRYS_EXPANDER` | pass a model; `dyp models` lists what is installed |
| `ollama has no model named 'X'` | `ollama pull X`, or use an installed one |
| `cannot reach ollama at http://localhost:11434` | `ollama serve`, or pass a `.gguf` path to `--expander` |
| reranking rejects a chat model | it needs a **cross-encoder** `.gguf`, not a chat model |

`--summarise` answering "no answer in these passages" is a search-failure signal,
not a library-lacks-it verdict: re-search with different words. A refusal that
survives rephrasing is strong evidence the answer is genuinely absent.
