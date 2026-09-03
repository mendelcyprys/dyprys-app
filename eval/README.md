# Evaluation set

`questions.json` is a list of `{q, a, kind}` objects. A question counts as
answered when the *returned passage* contains the term `a` — not when the right
book appears, which is a measure blind to chunk selection. The older
`["question", "answer"]` pair format still loads.

## Kinds

Retrieval fails differently depending on how a question is worded, so the kinds
are the axis worth reporting on:

| kind | what it does | example |
|---|---|---|
| `lookup` | asks in the vocabulary a textbook would use | *what myelinates axons in the central nervous system* |
| `descriptive` | describes the thing without naming it | *the tough outermost membrane covering the brain* |
| `mechanism` | asks how or why, not what | *why can a nerve impulse travel in only one direction* |
| `oblique` | everyday wording, no technical terms at all | *what is the brain's rubbish collection service* |

Two rules held throughout: **no question contains its own answer term**, and each
must stand alone — a question that only makes sense after the previous one is a
broken question, and both are checked.

## Difficulty is measured, not asserted

Labelling one's own questions "hard" proves nothing. `overlap_with()` reports the
share of a question's content words that appear in a passage that really does
answer it, and recall is reported split on that. The current set:

| | questions | recall@1 | recall@5 |
|---|---:|---:|---:|
| shares wording with the answer | 11 | 82% | 100% |
| shares little | 99 | 59% | 81% |

That gap is the point. The previous 35-question set was almost entirely the top
row, which is why BM25 scored level with vector search on it and the harness
could not tell the two apart.

## Terms are verified

Terms absent from the embedded corpus are reported as unanswerable and excluded
from the score rather than counted as misses, so the number never silently
measures the corpus instead of the retrieval.

## The sets

| file | n | what it is for |
|---|---:|---|
| `questions.json` | 110 | **frozen.** Every number recorded in this project is "of 110" |
| `questions-extended.json` | 300 | the same 110 plus 190, balanced 75 per kind — use this |
| `questions-rare.json` | 28 | answers in a median of 2 books: routing can fail here |
| `questions-gutenberg.json` | 24 | answers in 1 book of 3,453, same-domain competition |

`questions.json` is not edited. Growing it in place would make every recorded
figure uninterpretable — 78/110 and 78/300 are not the same claim, and nothing
in a git history tells you which set a number in a commit message meant.
`questions-extended.json` is a strict superset, so the original 110 can still be
scored on their own and the two stay comparable.

### Why 300

At 110 the harness could separate strategies and not settle small ones. Every
paired test in the reranking measurement came in at 4 to 29 discordant pairs,
where `+4/-0` is p = 0.125 — indistinguishable from luck. The noise floor from
the k-means seed alone is about two questions. Most remaining improvements in
this project are that size, so the instrument, not the ideas, had become the
limit.

The 190 new questions were written from domain knowledge rather than from the
passages that answer them: a question drafted while reading its own answer
shares that passage's vocabulary, which is the caveat the Gutenberg set already
carries. They are more specific than the original 110 — a median of 33 books
holding the answer against 78, and a maximum of 345 against 1,255.

## What is checked

`tests/test_questions.py` runs over every set in this directory. Until it was
written, this file claimed two of these rules were checked and none of them were.

- **No question contains its own answer.** Otherwise BM25 answers it for free.
- **No duplicates**, within a set or against the frozen one.
- **Each stands alone.** Search sees one question at a time, so a back-reference
  has no referent.
- **Kinds are one of the four**, and no field is empty.
- **Answers use the corpus's spelling.** This corpus is American — `fiber` 2828
  against `fibre` 542, `edema` 379 against `oedema` 13. A term spelled the other
  way scores as a miss however good the passage is, which measures the speller
  rather than the retrieval. The check caught one that survived a manual pass.
- **The extended set contains the frozen one.**

Terms are separately verified against the index before a question is admitted.
One candidate was dropped for being absent (`glymphatic`, a concept postdating
these textbooks) and twenty answers were replaced for being ordinary English
words — a passage containing "error" certifies nothing, and `error` sat in 1,901
books of 3,453.

### Answer terms match on a word boundary

`contains_term` matches `\b` + the term, not a bare substring. A plain substring
test counted "Saxon" as containing "axon", "produce" as containing "rod",
"ganglia" as containing "glia" and "biographer" as containing "raphe" — **23% of
all book-hits on this corpus**, inflating the metric in the one direction a
metric must never be inflated.

Correcting it moved the headline numbers by about one question each (flat
recall@1 78 → 77, recall@5 99 → 99; routed 68 → 67 and 93 → 92) and the chance
baseline for routing from 16.2% to 14.6%. Small, because a spurious match only
matters if it lands in the five passages returned — but the redundancy figure it
also feeds was overstated by three books.

The boundary is on the left only, because these sets deliberately use prefixes:
`electroencephalog` is meant to match `electroencephalography`, and `compensat`
to match `compensatory`.

Written by hand for a neuroscience library; no part derives from another tool.
