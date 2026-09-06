# Searching — reference

Deeper detail than `CLAUDE.md`. Read this when a search is not returning what the
user needs and you have to reason about why.

## Phrasing, by what the user gave you

- **A remembered phrase or exact wording** → pass it verbatim. The keyword half
  matches literal text; rewording destroys it. A verbatim hit shows `phrase N`.
- **A concept described, not named** → plain description in ordinary words. The
  meaning half is built for this.
- **A rare proper noun / specific term** → include it exactly; spelling matters,
  because a name only matches literally. Do not `--expand` it.
- **Everyday words about a technical library** → `--expand MODEL` rewrites the
  question into the library's vocabulary. Helps most on "oblique" phrasing; adds
  nothing when the answer is a rare literal.

Both halves run on every query and are merged; you never pick one. `--mode
vector` / `--mode lexical` force a single half and are rarely useful except to
see what each contributes.

## Reading a result line

```
1. [cos 0.52 · vec 1 · phrase 1] Principles.of.Neural.Science  (chunk 21482)
   … extract …
   ~/books/Principles.of.Neural.Science.txt:523017
```

- **`cos`** — cosine to the query, 0–1. Comparable *within one search only*, and
  higher is **not** better: a passage literally containing the answer scored 0.42
  while an unrelated book scored 0.63. Report it, never rank on it.
- **provenance** — which half found it and at what rank:
  - `vec N` — meaning search, rank N. `words N` — shares vocabulary, rank N.
  - `phrase N` — holds the query's words **in order** (a literal hit).
  - `vec 1 · phrase 1` — both halves agree strongly.
- **last line** — `path:byteoffset`. Hand this to the user; it opens the source.

## Flags

```sh
dyp ask "..." -k 10          # return N passages (default 5). Read several.
dyp ask "..." --route        # ~6x faster; reads ~1% of the library
dyp ask "..." -c PATTERN     # scope to books whose title/path matches (glob ok)
dyp ask "..." --full         # whole passage, not just its query-dense part
dyp ask "..." --json         # machine-readable (schema below)
dyp ask "..." -q             # results only, no stderr commentary
dyp ask "..." --expand MODEL     # rewrite the query first (~4s; vocabulary bridge)
dyp ask "..." --summarise MODEL  # draft prose, quotations verified (see below)
dyp ask "..." --rerank --reranker FILE.gguf  # cross-encoder reorder (~9s)
```

`--route` costs about one answer in twenty-five for a ~6x speedup — use it for a
quick look, drop it when the answer matters more than the second it saves. `-c`
matching nothing returns nothing (exit 1), never a silent full-library search.

`--rerank` rescores what the search already found, with a cross-encoder that
reads query and passage together instead of comparing two vectors after the
fact. It cannot recover an answer the search missed, so it raises rank rather
than recall — but that is usually the complaint. Add it whenever the ordering
matters and seconds are affordable.

Two measurements, on different corpora and of very different weight:

- the 110-question set behind the rest of these figures (117-book English
  non-fiction): reranking finds a few more answers than the plain order.
- a 24-run check on one 225-book library of translated religious law: the right
  passage came first twice as often (2/6 → 4/6), lifted from rank 3 and rank 4,
  and no answer was ranked worse than the plain order.

**Six questions on one corpus is a small sample.** Take the direction as
reliable and the size of the gain as anecdotal — and note that "never worse" is
what those runs happened to show, not a property of reranking: a cross-encoder
is a model and can be wrong about a passage. If ranking matters enough to
budget for, measure it on your own questions.

**Do not stack `--expand` and `--rerank`.** Measured, they recover the same
answers; pay for one. Reranking finds a few more; expansion is several times
faster.

## `--json` schema

```json
{
  "query": "...", "mode": "hybrid", "routed": false,
  "scanned_fraction": 0.0141, "elapsed_ms": 91.2,
  "results": [
    {"rank": 1, "chunk_id": 21482, "book": "Principles", "chapter": null,
     "path": "/abs/path.txt", "offset": 523017, "cos": 0.5199,
     "provenance": "vec 1 · phrase 1", "state": "exact",
     "text": "the passage bytes"}
  ]
}
```

- `provenance` is the native rank signal, not a fused score (a fused RRF score
  spans only 0.016–0.033 and is uninterpretable — never surface one).
- `offset` is the **chunk's** own start. The printed output shows a different
  byte for the same result, and both are real: a chunk boundary is a byte
  budget, so a passage usually begins mid-sentence, and the display widens it
  to the enclosing sentences and cites where the text it actually shows you
  begins. The two therefore differ by up to a sentence — typically a hundred
  bytes or so, always inside the same passage. Cite whichever matches the text
  you are quoting, and do not treat the pair as two different locations when
  reconciling a printed result against a parsed one.
- `text` is `null` when the passage could not be proved against its stored hash;
  `state` says which: `exact`, `shifted` (found intact at a moved offset),
  `changed` (file edited, bytes gone), `missing` (file gone/unmounted). **Never
  quote a passage whose `text` is null.**
- Empty results are still valid JSON (`results: []`); exit code is 1 on a miss.

## `--summarise` — what is and isn't trusted

Drafts prose from the passages, but **trusts nothing the model says**:

- Every quotation is checked byte-for-byte against the source and **cut from the
  prose if it does not match** (~1 in 10 gets cut).
- Locations never come from the model: it picks among numbered passages, and the
  book, chunk and byte offset are attached afterward from the index.
- It may answer "no answer in these passages" — a signal the *search* failed, not
  the library. Re-search with different words before concluding the library lacks
  it; a refusal that survives rephrasing is strong evidence it is absent.

## When nothing good comes back

**Hybrid search always returns `k` results at exit 0** — the meaning half fills
them from the nearest vectors even when the library holds nothing relevant. So a
non-empty result is not proof of an answer. The deciding test is the **text**: read
the top passage and judge whether it addresses the question. A probable miss is a
`vec`-only top hit with a low `cos` *for that search* whose text is off-topic
(a copyright page, a methods table). When a topic may simply be absent, reach for
`--summarise` — it says "these passages do not answer the question" and retries
once with a rephrasing, which is a far stronger absence signal than one weak hit.

1. **`no passage matched`** (exit 1, embedded index) — a lexical-only search found
   no words, or `-c` matched nothing. Rephrase, or drop the scope.
2. **`nothing embedded yet`** — the index has text but no vectors; it needs
   `dyp embed`.
3. **Wrong book, plausible passage** — you used `--route`; drop it and retry, the
   answer may be just outside the routed set.
4. **Only one hit, and it's wrong** — read more (`-k 10`); the answer is first
   ~6/10 and in the top five ~8/10.
5. **A one- or two-word query** returns meaning-half results only — too short to
   form a phrase for the keyword half. Add words, or quote an exact term.
