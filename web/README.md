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

**Phase 1 — libraries.** The rail's picker (books, per-model coverage, an
index that is registered but not mounted, which one is the default), naming and
forgetting a library, and `NOTES.md` rendered — with a one-line nudge on the
first search of a session, because notes only help when read *before*
searching.

Books, models, ask and jobs are phases 2 to 5 and are stubs.

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
