# The web frontend — plan

`dyp serve` already exposes everything the CLI can do (`docs/serving.md`). This
plans the browser client that sits on it: not a search box, but **a place to run
a library** — see what is in it, choose what to ask, watch what is being built,
and read a passage where it actually lives.

The order below is deliberate and is not "search first". Search is the part that
already works; the parts that don't exist yet are the ones that decide *what a
search is asked of*. A results list is easy to build twice. A library manager
that was bolted on afterwards is not.

---

## The shape

Two long-lived panes and a transient one:

```
┌─ rail ────┬─ workspace ─────────────────────┬─ reader ────────┐
│ library ▾ │  ask · books · models · jobs    │  the passage    │
│ model  ▾  │                                 │  in its book,   │
│ scope     │  (the tab you are in)           │  scrollable     │
│ ───────── │                                 │                 │
│ asked     │                                 │  opens on a     │
│ (history) │                                 │  result click   │
└───────────┴─────────────────────────────────┴─────────────────┘
```

- **The rail is the state of the question**: which library, which model, which
  books. It persists across tabs, because those three choices outlive any one
  search and every tab means something different depending on them.
- **The workspace is four tabs**: Ask, Books, Models, Jobs.
- **The reader slides in** over the right third when a result is clicked, and
  can be pinned. It is the whole reason for `GET /source`.

One rule that shapes the entire UI: **the library name is a path parameter on
every call**. There is no server-side "current library". The rail's selection is
client state, and every request carries it. That is what makes two browser tabs
on two libraries work at once, and it is worth not breaking.

---

## The stack

**Vite + React + TypeScript, Tailwind, and shadcn/ui.** Recommended rather than
surveyed, because "modern UI that handles buttons, menus, dialogs" is exactly
what that combination is for: shadcn/ui is Radix primitives (menus, dialogs,
comboboxes, popovers, tooltips, command palette) vendored into the repo as
source you own, so there is no component library to fight when a picker needs to
show coverage bars inside its options — which two of the pickers below do.

- **TanStack Query** for server state. Every read endpoint here is a poll or a
  cache-with-invalidation, and jobs need refetch-on-interval that stops when the
  tab is hidden. Writing that by hand is where the bugs go.
- **TanStack Virtual** for the book list and the reader. `neuro` has 3,453 books
  and books are megabytes; neither can be a plain `.map()`.
- **`cmdk`** (ships with shadcn) for a `⌘K` palette — switch library, jump to a
  book, start a search. On a tool with this many nouns it earns its place early.
- No state library beyond Query + a small context for the rail. There is no
  client state worth a store.
- The build lands in `web/dist` and `dyp serve --web web/dist` serves it, which
  the API already supports. Dev is Vite on :5173 against the server's CORS
  default, which is already `DEFAULT_ORIGINS`.

---

## Phase 1 — Libraries: choose one, see its condition  ✅ built

The rail's library picker, and a first-run screen when nothing is registered.

Reads `GET /api/libraries`, which is richer than a name list and should be shown
as such: `exists` (registered but the drive is not mounted is a real and common
state), `books`, `chunks`, per-model `coverage`, `default`, and `notes`.

- The picker shows **name · books · coverage**, and greys out `exists: false`
  with "index not found at {path}" rather than letting it be chosen and fail.
- **`notes` is the field to take seriously.** CLAUDE.md tells a reader to read a
  library's `NOTES.md` before searching it — which shelf `-c` cuts along, two
  subjects whose vocabulary collides, a title that means two books. Nobody will
  open a second tab to read it. So: a **Notes** button in the rail, badged when
  the library has one, opening `GET /notes` as rendered markdown in a sheet; and
  on the first search of a session against a library that has notes, a single
  dismissible line above the results linking to it.
- `registry_readable: false` is its own screen, not a toast. Nothing works.

**The gap this phase opened, and how it was closed.** There was no API for
registry mutation — `dyp library add|use|remove` had no routes — so a browser
could read every library and name none. Taken as recommended: `POST
/api/libraries`, `DELETE /api/libraries/{name}` and `POST
/api/libraries/{name}/default`, with the path as server-side text, since a
browser cannot pick a directory and this server is loopback-only and already
reads the whole filesystem through `dyp add`.

Three refusals the CLI does not make, each of them because the caller is now a
browser rather than someone with a next command to type: the directory must
already exist, the name may not hold a slash or a space (it is a path parameter
on every other route), and a name already taken is a refusal carrying the taken
names rather than `registry.add`'s silent overwrite. `--delete`, which erases an
index directory, deliberately has no route. See `docs/serving.md`.

## Phase 2 — Books: what is in here, and what am I asking  ✅ built

The Books tab, and the scope control in the rail. This is the phase that changes
what a search *is*, so it comes before making search pretty.

`GET /books?pattern=` returns per book: `title`, `key` (the path), `chunks`,
`lexical_indexed`, `sources[]` with `present`, `chunkings[]`, and `embedded` /
`live_chunks` **per model**. That is a table, virtualised, with:

- a filter box that debounces into `?pattern=` (the same glob/substring matcher
  `-c` uses, so what the box does and what the scope means are one thing);
- a coverage column **per model**, since a book fully embedded by one model and
  untouched by another is normal and invisible today;
- `present: false` rendered loudly — the file is gone, the chunks remain, and
  search will return `text: null` for it;
- **attribution from `key`, never `title`.** Titles collide; one library can
  hold two different books under one title. The table shows the path under the
  title, and the reader header shows the path, always.

**Selecting what to query.** Checkboxes in this table feed the rail's scope
chips. A selected book contributes its **path**, which is unique where a title
is not; the filter box can also be promoted to a scope as one pattern, which is
the difference between "these four books" and "the Talmud shelf, whatever is on
it". The chips live in the rail rather than on this tab because a scope you
cannot see from where you type is one you forget you set.

This was the one place the plan asked for a **service change rather than a UI
trick**, and it was taken: `collection` is now `str | list[str]`, `search.scope`
unions the matches, and `-c` is repeatable. The API needed no new field.

One thing the union introduced that a single pattern never had: a mistyped
pattern beside a good one contributes no books and changes no result, so it
would narrow the search invisibly, at exit 0, with citations that look correct.
So `scope` returns the patterns that matched nothing and the search refuses on
**any** miss, not only when everything missed.

**And the scope must be honest in the UI.** `test_ask_pipeline_contract.py`
pins deliberate behaviour: the exact-phrase leg searches the whole library even
under a scope, because confining it took lexical safety from 20/20 to 9/20. So a
scoped search *can* return a book you did not select, and the citation will look
correct. Render those rows with an "outside your scope — exact phrase match"
badge. Without it, the first person to notice will file it as a bug, and the
second will stop trusting the scope.

## Phase 3 — Models: which vectors answer, and where the files are  ✅ built

The Models tab plus the rail's model picker. `GET /models` gives `name`, `alias`,
`dim`, `store`, `coverage`, `disk_bytes`, `failures`, `routing`, `file_path` and
`file_present`.

- The rail picker is a combobox showing **alias or short handle · coverage ·
  dim**, and it is *required* on a multi-model index. The API already refuses
  with `model_ambiguous` carrying `choices` — so the picker's empty state is
  populated **from the error**, which is why `choices` is on the error shape at
  all. Wire that path explicitly; it is the first-run experience on `neuro`.
- `file_present: false` is a warning row: this model embedded the index but its
  GGUF is no longer on the machine, so you can read its coverage and cannot
  search with it. Today that surfaces as a 503 at query time.
- `POST /warm` on selection, with a spinner in the picker. Loading a 300 MB–1 GB
  GGUF is the dominant cost of the first question, and doing it when the user
  picks rather than when they ask moves the wait somewhere they expect it.
  `GET /health` reports what is resident, so the picker can show a warm dot.

**The reranker is the awkward one and needs designing, not defaulting.** It is a
cross-encoder GGUF path the index does not remember, it must be passed on every
`ask`, and bare `rerank` is an error rather than a downgrade. In a browser
"type an absolute path each time" is not a design. Proposal: a **Settings sheet
holding per-library reranker/expander/summariser choices in `localStorage`**,
attached to every request from there, with the reranker as a path field that
remembers what worked. To make it choosable rather than typed, add a small
`GET /api/models/available` that lists `*.gguf` under the configured model
directories — a lookup the CLI does informally today and the UI cannot do at all.

Also here: **`--expand` and `--rerank` must not both be offered as "on".**
Measured, they are substitutes, and together they cost the sum for the gain of
the better one. Make them a segmented control — *fast · expand · rerank* — not
two checkboxes.

## Phase 4 — Ask, and the reader  ✅ built

Only now the search itself, and it should use **`POST /ask/stream`** from the
start rather than being retrofitted. A ten-second routed search with no output is
indistinguishable from a hang, and the stage frames are already there: the
rephrasings an expander chose, the books routing picked, "rescoring 20 passages",
the summariser's retry. Render them as a live line above the results, then
collapse them into a "how this was found" disclosure when the result frame lands.

The result list obeys three rules that come straight from `docs/serving.md` and
are easy to violate in a UI:

1. **`text: null` is never rendered as a quotation.** The passage could not be
   proved against its stored hash; `state` says why. It renders as a struck-out
   card explaining that, with the path still shown.
2. **`cos` is displayed, never sorted on.** It is comparable only inside one
   response and a higher number is not a better answer. Show it as a small dim
   figure beside `provenance`, and keep server rank order.
3. **`warnings` is rendered unconditionally.** `unprofiled_books` means routing
   could not reach part of the library *at any rank*, while returning a full `k`
   at 200 — the one failure invisible from the output. It belongs at the top of
   the results, with a "run `route`" button that starts the job from Phase 5.

Each result shows `provenance` as chips (`phrase` / `words` / `vec`), the book
title with its path beneath, and opens the reader on click.

**The reader** is `GET /source?path=&offset=&span=`, and it is why the passage is
worth opening at all: the span is snapped to sentence boundaries and capped at
200 KB, so scrolling means re-requesting at a new offset. Build it as a windowed
scroller — up and down buttons that refetch at `offset ± span`, the matched span
highlighted, the path and byte offset in the header, and a copy-citation button
that yields exactly `path:offset`. Never invent a citation; the offsets are real.

`--summarise` renders above the passages, with every claim's quote linking to the
result it was checked against, and the "these passages do not answer the
question" refusal rendered as a first-class answer rather than an error — it is
the surest evidence the library lacks something.

## Phase 5 — Jobs: build the library from the browser  ✅ built

`GET /jobs`, `POST /jobs/{kind}`, `DELETE /jobs/{kind}`, and the log tail.

- A card per kind (`embed`, `add`, `route`, `lexical`, `compact`) showing
  `running`, `done/live`, `share`, `rate`, `eta_seconds`, `book_in_flight`.
- **Poll on an interval and let `rate` be measured.** `rate` is computed from the
  gap between *your own* polls, falling back to the median this machine has
  managed. A UI that polls once and stops shows the guess forever.
- **A stop button is safe and should be offered**, because `DELETE` sends SIGINT
  and `dyp embed` finishes the batch in flight and commits it. Say that on the
  button's confirm: "stops after the current batch; nothing is lost".
- **409 is a normal outcome, not a failure toast.** Something already holds the
  lock — quite possibly the user's own terminal. Show it as "a run is already in
  progress" with the log tail, which is the useful thing.
- **A prompt to run `route` after every embed.** `dyp check`'s
  `unprofiled_books` is the number to badge the Jobs tab with. This is the single
  most valuable nudge the UI can offer, because the failure it prevents is silent.
- Never start `embed` implicitly. It can run for days. The button says how much
  is outstanding and roughly how long, from `check` and `observed_rate`.

## Phase 6 — Asked, and the polish

`GET /asked` is cross-session, cross-*interface* memory — searches run from the
terminal appear here, and since the service layer landed, `--json` searches do
too. In the rail as a history list: click to re-run with the same options, or
open the stored answer. Then the `⌘K` palette, keyboard nav through results
(`j`/`k`, `enter` to open the reader), and dark mode.

---

## What the API still needs

Phases 1 to 3 are built and everything they needed has been added — registry
writes, `collection` as a union, `GET /models/available`, and model aliases.
What is left belongs to phases still to come:

| need | phase | shape |
|---|---|---|
| cancel an in-flight search | 4 | the stream's disconnect is currently not observed; the worker runs on |
| book table of contents | 4 | chapter offsets for a book, so the reader can jump |

## What is deliberately not in this plan

- **No auth, and no non-loopback bind.** The server has none by design, and a
  frontend is not the place to invent one.
- **No client-side re-ranking, filtering, or sorting of results.** The server's
  order is the answer. A UI that lets you sort by `cos` is a UI that teaches its
  user something false.
- **No corpus knowledge in the client.** Which shelves exist, what the vocabulary
  is, which titles collide — that is `NOTES.md`, transported and rendered, never
  interpreted. The server is corpus-independent and the client should be too.
