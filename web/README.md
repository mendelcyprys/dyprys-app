# The browser client

A place to run a library — not a search box. `docs/plans/frontend.md` is the
plan and its phase order; this is how to run what exists.

```sh
dyp serve --port 8765      # the API, in one terminal
npm install                # in web/
npm run dev                # http://localhost:5173, proxying /api to :8765
```

`:5173` is already `dyp serve`'s default allowed origin, and Vite proxies
`/api`, so the client never holds a server address.

To serve it from `dyp` itself:

```sh
npm run build
dyp serve --web web/dist
```

## What is here

**Libraries** — the rail's picker (books, per-model coverage, an index that is
registered but not mounted, which one is the default), naming and forgetting a
library, and `NOTES.md` rendered, with a one-line nudge on the first search of a
session because notes only help when read *before* searching.

**Books** — a library holds shelves; a shelf holds books. The middle level was
always there (`-c papers/` has always meant one shelf, `dyp embed -c` fills one at
a time) and only the flat list did not show it: `neuro` was 3,453 rows with no
structure, and is now three — 1,679 scanned texts, 1,657 Gutenberg books, 117
extracted neuroscience PDFs. Which of the three a question belongs to is the most
useful thing about that library, and it was invisible.

Shelf headers and books are **one** flat virtualised list of two row kinds
rather than a list of lists, so a shelf of 1,657 books costs what one of three
costs. Shelves start folded above 60 books and always open while filtering,
because the filter is the thing being looked at. A header's counts stay the
library's, with a `5 of 117` prefix when a filter is narrowing what is under it —
a header that shrank with the filter would say the Talmud shelf holds two books
because two of them matched.

A shelf enters the scope as **one pattern**, its directory: that is what a
terminal would write, it survives books being added to the shelf, and it is one
chip in the rail rather than 1,657. A selected *book* still contributes its
**path**, because titles collide and sefaria holds two books called `Arakhin`.

**Removing things** has two settings at every level, and which one a control gets
follows from whether the embedding survives. **Set aside** is a plain button —
every vector, passage and BM25 row stays, no search reads them, one click back —
and it is what to reach for when a book's extraction lost its word boundaries or
a shelf swamps every answer. **Remove** and **delete the index** are previewed by
the server and then typed to confirm, because they are not recoverable. No source
text is deleted by any of them, and the index-deletion dialog says where the text
lives so that is checkable rather than a promise.

**Models** — a picker that is required on a multi-model index and warms on
selection, `model_ambiguous` rendered as the picker that resolves it, weights
that have gone missing said before the question rather than as a 503 during it,
and a settings sheet holding the per-library choices the index cannot remember.

**Ask and the reader** — the streamed search, the drafted answer with its quotes
checked, and `GET /source` as a window around a passage. Stopping a search
really stops it: the server is told when the stream closes, and the search gives
up at its next checkpoint rather than holding the library's one worker thread
while the next question queues behind a closed tab.

**Search settings** — three sections, because a search makes three independent
choices and the sheet had been running two of them together. *Where to look*
is routing; *how hard to look* is the effort control; *what comes back* is the
drafted answer. Each option carries the model it needs inside it, so a reranker
cannot be chosen while the search is set to expand.

Routing had no control at all until this: `route` was declared on the browser's
`SearchBody` and never sent, so every search from the browser read the whole
library — on `neuro`'s 3,453 books as readily as on a 20-book one. It is the
tool's largest speedup (~6x, about 1% of the library read) for roughly one
answer in twenty-five, and it composes with everything else: `_pipeline` takes
the router and the reranker as separate arguments, and `eval` runs a routed
rerank on purpose. Folding it into the effort control would have made the
biggest speedup unreachable exactly when a large library also wanted a better
ranking. It is offered only where the chosen model has a profile, because a
search asked to route without one is refused rather than quietly run flat —
better said at the switch than in a red box after the question.

`tests/test_api.py` pins the whole request: every field of `SearchOptions` is
either in the body `ask.tsx` builds or listed with the reason it is not offered.
A typed field nobody sends is invisible from both sides — Python saw a valid
request, TypeScript saw an optional property — and that is how routing stayed
missing.

Reranking's depth is a slider, because it is the whole cost — one cross-encoder
pass per candidate. On `neuro` with a 0.6B reranker: retrieval alone is 0.2s,
depth 5 is 4.3s, depth 20 is 25.7s, and the top passage was the same at 5 as at
20.

**Jobs** — a card per kind with the log tail, polled on an interval so `rate` is
measured rather than guessed, a stop button that is safe because embedding is
resumable, and a `route` prompt badged from `unprofiled_books` — the one failure
that is invisible from a search's output.

`embed` and `route` refuse to start on a multi-model index until a model is
chosen, because a detached subprocess refuses by exiting a second after it
starts, which over HTTP looks like a job that never ran.

`embed` and `add` are **set up rather than run**: they are the two whose flags
change what the run is, so neither is one click. `embed` offers `--for`,
`--duty`, `--limit`, `--batch`, `-c` and the model — the browser sent none of
these before, so every run it started was unbounded, full duty, whole library,
which is the wrong default for the one command that can take days. `add` offers
`--ext`, `--chapters`, `--deep` and the chunk size.

Two controls are **reported rather than offered** once they are settled, since
both fail as a subprocess that exits a second after it starts. A model bound to
a chunking cannot be moved to another (a second size needs a second model), and
a registered model's quantisation is fixed — the form says so and names
`dyp models --quantise`. A model with no binding yet gets both as real choices,
which is how a second chunk size is reached from the browser at all: `add` the
same paths at a new `--target`, then embed that chunking with a second model.

**Light and dark** — three states in the rail: light, dark, or follow the
system, remembered per browser. An inline script in `index.html` applies the
stored choice before the first paint; the provider keeps it right afterwards,
including when the system flips under a page that is already open.

**Books** — every book, virtualised, filtered on the server through the matcher
`-c` uses. `Open` gives one book's card: read it from the top, scope the next
question to it, give it a name, or write down what it is.

The naming is worth explaining. `title` is derived from the filename and ingest
rewrites it whenever the file moves, so a chosen name is a separate field and
neither overwrites the other — the file keeps its name and the book gets one.
Neither is identity: the key is, because titles collide. `sefaria` holds two
different works called Arakhin, and the list showed them as two identical rows,
which is exactly the mistake CLAUDE.md's "attribute from the path, never the
title" rule exists to prevent. A colliding title now carries the shelf it came
from, and naming one is the permanent fix. A label is matched by `-c`, so a book
you have named is one you can scope to by name.

A book's note rides back on every result from it, under the passage — the
library's `NOTES.md` one book down, in the place where "which edition is this?"
is actually asked.

**The reader** — one continuous document, not a series of windows. The first
version paged: Earlier and Later re-requested a window at a new offset and
replaced what was on screen, which is a defensible way to *look at* a citation
and a bad way to *read* — every move threw away where you were. Now the loaded
byte range grows at whichever edge you scroll to, each stretch continuing from
the previous one's `end` (asking again at `offset + span` would skip whatever
the sentence-snap trimmed), and a prepend corrects the scroll position by the
height it added, in a layout effect, before the browser paints.

**Jobs** also carries what a search will never tell you: books whose extraction
failed and so can never match a query, files gone or edited since indexing, an
incomplete BM25 index, chunks that failed to embed. `dyp check` has always
computed all of it; the browser read two fields and typed the rest `unknown[]`.
Every entry shares one shape — the search still returns `k` results and still
exits 0 — which is exactly why they need saying somewhere.

**Asked, and the keyboard** — the rail lists what this library has already been
asked, from either frontend: a question typed in a terminal shows up there, and
so does one an agent asked through `--json`. Clicking runs it again. `Cmd-K`
opens a palette over libraries, books, tabs and past questions; `j`/`k` move
through results and `enter` opens one in its book.

## The rules this client is built to keep

Three of them come straight from `docs/serving.md`, and every one is easy to
break by writing the obvious UI:

- **`text: null` is never rendered as a quotation.** The passage could not be
  proved against its stored hash.
- **`cos` is displayed, never sorted on.** It is comparable only inside one
  response, and a higher number is not a better answer.
- **`warnings` is rendered unconditionally.** `unprofiled_books` means routing
  could not reach part of the library at *any* rank, while returning a full `k`
  at 200 — the one failure invisible from the output.

And one that is structural: **the library name is a path parameter on every
call.** There is no server-side "current library", which is why two tabs can
run two libraries, so nothing in `lib/api.ts` remembers one.
