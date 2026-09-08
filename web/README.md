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

**Books** — virtualised (3,453 in `neuro`), filtered on the server through the
same matcher `-c` uses, per-model coverage, missing source files called out.
Checkboxes feed the rail's scope, and a selected book contributes its **path**:
titles collide, and sefaria holds two books called `Arakhin`.

**Models** — a picker that is required on a multi-model index and warms on
selection, `model_ambiguous` rendered as the picker that resolves it, weights
that have gone missing said before the question rather than as a 503 during it,
and a settings sheet holding the reranker per library.

**Ask and the reader** — the streamed search, the drafted answer with its quotes
checked, and `GET /source` as a window around a passage.

**Jobs** — a card per kind with the log tail, polled on an interval so `rate` is
measured rather than guessed, a stop button that is safe because embedding is
resumable, and a `route` prompt badged from `unprofiled_books` — the one failure
that is invisible from a search's output.

`embed` and `route` refuse to start on a multi-model index until a model is
chosen, because a detached subprocess refuses by exiting a second after it
starts, which over HTTP looks like a job that never ran.

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
