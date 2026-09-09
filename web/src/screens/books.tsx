import * as React from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  Archive,
  ChevronDown,
  ChevronRight,
  FileX2,
  FolderOpen,
  Info,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Tooltip } from "@/components/ui/tooltip";
import { CoverageBar, shortModel } from "@/components/coverage";
import { RemoveBooks } from "@/components/remove-dialog";
import type { BookRow, Shelf } from "@/lib/api";
import { useDebounced } from "@/lib/debounce";
import { useBooks, useShelves } from "@/lib/queries";
import { useSelection } from "@/lib/selection";
import { bytes, cn, count } from "@/lib/utils";
import { BookSheet } from "./book-sheet";
import { Reader, type Reading } from "./reader";

const BOOK_ROW = 56;
const SHELF_ROW = 60;

/** Past this many books, shelves start closed: the point of the level is that
 *  a 3,453-book library is three rows before it is three thousand. */
const FOLD_ABOVE = 60;

type Row = { kind: "shelf"; shelf: Shelf; showing: number } | { kind: "book"; book: BookRow };

/**
 * A library holds shelves; a shelf holds books.
 *
 * The middle level was always there — `-c papers/` has always meant one shelf,
 * `dyp embed -c` fills one at a time — and only the flat list did not show it.
 * On `neuro` that flat list is 3,453 rows with no structure at all; as shelves
 * it is three: 1,679 scanned texts, 1,657 Gutenberg books, 117 extracted
 * neuroscience PDFs. Which of those three a question should be asked of is the
 * most useful thing anyone can know about that library, and it was invisible.
 *
 * Rows are one flat virtualised list of two kinds, rather than a list of lists,
 * so a shelf of 1,657 books costs the same to render as one of three.
 */
export function Books({ library }: { library: string }) {
  const [filter, setFilter] = React.useState("");
  const pattern = useDebounced(filter);
  const books = useBooks(library, pattern);
  const shelves = useShelves(library);
  const { scope, setScope } = useSelection();
  const viewport = React.useRef<HTMLDivElement>(null);
  // Which book is open, by key rather than by value: the row is re-fetched
  // after a rename, and a copy taken at click time would show the old name.
  const [opened, setOpened] = React.useState<string | null>(null);
  const [reading, setReading] = React.useState<Reading | null>(null);
  const [removing, setRemoving] = React.useState<Shelf | null>(null);
  const [shut, setShut] = React.useState<Set<string> | null>(null);

  const found = books.data?.books ?? [];
  const shelfRows = shelves.data?.shelves ?? [];
  const chosen = React.useMemo(() => new Set(scope), [scope]);

  // Closed by default on a big library, open on a small one — and always open
  // while filtering, because the filter is the thing being looked at and
  // hiding its matches inside a closed row is answering a different question.
  const closed = React.useMemo(() => {
    if (pattern) return new Set<string>();
    if (shut) return shut;
    const many = found.length > FOLD_ABOVE && shelfRows.length > 1;
    return many ? new Set(shelfRows.map((shelf) => shelf.path)) : new Set<string>();
  }, [shut, pattern, found.length, shelfRows]);

  const rows: Row[] = React.useMemo(() => {
    const byShelf = new Map<string, BookRow[]>();
    for (const book of found) {
      const list = byShelf.get(book.shelf);
      if (list) list.push(book);
      else byShelf.set(book.shelf, [book]);
    }
    const known = new Map(shelfRows.map((shelf) => [shelf.path, shelf]));
    const out: Row[] = [];
    // Shelf order comes from the shelves query, so it does not reshuffle as a
    // filter narrows the books. A shelf a filter emptied is simply not shown.
    for (const path of [...known.keys()].sort()) {
      const here = byShelf.get(path);
      if (!here?.length) continue;
      const shelf = known.get(path)!;
      out.push({ kind: "shelf", shelf, showing: here.length });
      if (!closed.has(path)) {
        for (const book of here) out.push({ kind: "book", book });
      }
    }
    return out;
  }, [found, shelfRows, closed]);

  const virtual = useVirtualizer({
    count: rows.length,
    getScrollElement: () => viewport.current,
    estimateSize: (index) => (rows[index]?.kind === "shelf" ? SHELF_ROW : BOOK_ROW),
    overscan: 12,
  });

  /**
   * What "everything listed" is, as patterns rather than as keys.
   *
   * Ticking this used to put every visible book key in the scope: on `neuro`
   * that is 3,453 patterns, 2.4 seconds of work to build, a rail reading
   * "+3,447 more", and a request asking the server to resolve three thousand
   * patterns one at a time — to say what three shelf names say, or what an
   * empty scope says. The shelf checkboxes had already settled the right unit;
   * this is the same unit, one row up.
   */
  const shownPatterns = React.useMemo(
    () =>
      pattern
        ? [pattern]
        : rows.flatMap((row) => (row.kind === "shelf" ? [row.shelf.directory] : [])),
    [pattern, rows],
  );
  const allShown = shownPatterns.length > 0 && shownPatterns.every((each) => chosen.has(each));
  const someShown = !allShown && shownPatterns.some((each) => chosen.has(each));

  function toggle(key: string) {
    setScope(chosen.has(key) ? scope.filter((each) => each !== key) : [...scope, key]);
  }

  function toggleShown() {
    setScope(
      allShown
        ? scope.filter((each) => !shownPatterns.includes(each))
        : [...new Set([...scope, ...shownPatterns])],
    );
  }

  /**
   * A shelf enters the scope as **one pattern** — its directory — rather than
   * as its books. `-c` matches a path, so `…/Talmud` is exactly the scope a
   * terminal would write, it survives books being added to that shelf, and it
   * is one line in the rail instead of 1,657.
   */
  function toggleShelf(shelf: Shelf) {
    setScope(
      chosen.has(shelf.directory)
        ? scope.filter((each) => each !== shelf.directory)
        : [...new Set([...scope, shelf.directory])],
    );
  }

  function fold(path: string) {
    setShut((current) => {
      const next = new Set(current ?? closed);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }

  const models = Object.keys(found[0]?.live_chunks ?? {});

  /**
   * Titles that more than one book here answers to.
   *
   * `sefaria` holds Arakhin twice — once from the Mishnah, once from the Talmud
   * — and the list showed them as two identical rows. CLAUDE.md's standing rule
   * is to attribute from the path and never from the title alone, and this is
   * the screen where that rule is easiest to break: the path is there, in grey,
   * three points smaller than the name.
   *
   * With shelves shown, the shelf is usually what tells them apart, so that is
   * what the badge says.
   */
  const collides = React.useMemo(() => {
    const seen = new Map<string, number>();
    for (const row of found) {
      const name = row.label ?? row.title;
      seen.set(name, (seen.get(name) ?? 0) + 1);
    }
    return new Set([...seen].filter(([, n]) => n > 1).map(([name]) => name));
  }, [found]);

  const aside = found.filter((book) => book.set_aside).length;

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder="Title or path — a shelf (papers/), part of a title, a glob (*Atlas*)"
            className="pl-8"
            spellCheck={false}
          />
          {filter && (
            <button
              onClick={() => setFilter("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="size-3.5" />
            </button>
          )}
        </div>

        {pattern && found.length > 0 && (
          <Tooltip label="scope to every book this filter matches, as one pattern rather than a list">
            <Button
              size="sm"
              variant="outline"
              onClick={() => setScope([...new Set([...scope, pattern])])}
            >
              Scope to “{pattern}”
            </Button>
          </Tooltip>
        )}
      </div>

      <div className="flex items-center gap-3 px-1 text-xs text-muted-foreground">
        <Checkbox
          checked={allShown}
          indeterminate={someShown}
          onCheckedChange={toggleShown}
          aria-label="ask only what is listed here"
        />
        <span>
          {books.isLoading
            ? "reading…"
            : `${count(found.length, "book")}${pattern ? " matching" : ""} on ${count(
                rows.filter((row) => row.kind === "shelf").length,
                "shelf",
                "shelves",
              )}`}
        </span>
        {aside > 0 && (
          <Tooltip label="set aside: still here, still embedded, and returned by no search">
            <Badge variant="outline" className="font-normal">
              <Archive className="size-2.5" /> {aside} set aside
            </Badge>
          </Tooltip>
        )}
        {scope.length > 0 && (
          <>
            <span className="ml-auto">{scope.length} in scope</span>
            <Button size="sm" variant="ghost" className="h-6 px-2" onClick={() => setScope([])}>
              Clear
            </Button>
          </>
        )}
      </div>

      <div ref={viewport} className="min-h-0 flex-1 overflow-y-auto rounded-lg border">
        {rows.length === 0 && !books.isLoading && (
          <p className="p-8 text-center text-sm text-muted-foreground">
            No book’s title or path matches that.
          </p>
        )}
        <div style={{ height: virtual.getTotalSize(), position: "relative" }}>
          {virtual.getVirtualItems().map((item) => {
            const row = rows[item.index];
            return (
              <div
                key={row.kind === "shelf" ? `shelf:${row.shelf.path}` : row.book.key}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: item.size,
                  transform: `translateY(${item.start}px)`,
                }}
              >
                {row.kind === "shelf" ? (
                  <ShelfHead
                    shelf={row.shelf}
                    showing={row.showing}
                    filtered={Boolean(pattern)}
                    closed={closed.has(row.shelf.path)}
                    scoped={chosen.has(row.shelf.directory)}
                    models={models}
                    onFold={() => fold(row.shelf.path)}
                    onScope={() => toggleShelf(row.shelf)}
                    onRemove={() => setRemoving(row.shelf)}
                  />
                ) : (
                  <Book
                    book={row.book}
                    models={models}
                    chosen={chosen.has(row.book.key)}
                    onToggle={() => toggle(row.book.key)}
                    onOpen={() => setOpened(row.book.key)}
                    shared={collides.has(row.book.label ?? row.book.title)}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      <BookSheet
        library={library}
        book={found.find((row) => row.key === opened) ?? null}
        onClose={() => setOpened(null)}
        onRead={(book) => {
          const first = book.sources.find((source) => source.present) ?? book.sources[0];
          // Offset 0 and no text: the reader window is fetched from the file,
          // and there is no passage to anchor on -- this is the top of the book.
          setReading({ path: first.path, offset: 0, book: book.label ?? book.title, text: null });
          setOpened(null);
        }}
        onScope={(book) => {
          setScope([...new Set([...scope, book.key])]);
          setOpened(null);
        }}
      />
      {removing && (
        <RemoveBooks
          library={library}
          what={{ shelf: removing.directory }}
          // The relative path, not the last segment: "the old shelf" could be
          // `papers/old` or `notes/old`, and this is the sentence someone
          // confirms an irreversible action against.
          subject={`the ${removing.path || removing.name} shelf`}
          books={removing.books}
          setAside={removing.set_aside}
          open
          onOpenChange={(next) => !next && setRemoving(null)}
          onRemoved={() => setScope(scope.filter((each) => each !== removing.directory))}
        />
      )}
      <Reader library={library} reading={reading} onClose={() => setReading(null)} />
    </div>
  );
}

/**
 * One shelf, and everything a person decides about a shelf from one row.
 *
 * The counts are the library's, not the filter's: a filter narrows what is
 * listed under the header, and a header that shrank with it would say the
 * Talmud shelf holds two books because two of them matched "Arakhin".
 */
function ShelfHead({
  shelf,
  showing,
  filtered,
  closed,
  scoped,
  models,
  onFold,
  onScope,
  onRemove,
}: {
  shelf: Shelf;
  showing: number;
  filtered: boolean;
  closed: boolean;
  scoped: boolean;
  models: string[];
  onFold: () => void;
  onScope: () => void;
  onRemove: () => void;
}) {
  const allAside = shelf.set_aside >= shelf.books && shelf.books > 0;
  return (
    <div
      onClick={onFold}
      className={cn(
        "flex h-full cursor-pointer items-center gap-2.5 border-b bg-muted/50 px-3 transition-colors hover:bg-muted",
        scoped && "bg-accent/60 hover:bg-accent/70",
        allAside && "opacity-60",
      )}
    >
      <Checkbox
        checked={scoped}
        onCheckedChange={onScope}
        aria-label={`ask only the ${shelf.name} shelf`}
      />
      {closed ? (
        <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" />
      ) : (
        <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" />
      )}
      <FolderOpen className="size-3.5 shrink-0 text-muted-foreground" />

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          {/* The relative path, not the last segment: `gutenberg` and
              `scale_texts/gutenberg` are one word for two shelves, and telling
              them apart is the whole point of showing this level. */}
          <span className="truncate text-sm font-semibold">{shelf.path || shelf.name}</span>
          {shelf.set_aside > 0 && (
            <Tooltip label="set aside: still here, still embedded, and returned by no search">
              <Badge variant="outline" className="shrink-0 font-normal">
                <Archive className="size-2.5" />
                {allAside ? "set aside" : `${shelf.set_aside} set aside`}
              </Badge>
            </Tooltip>
          )}
        </div>
        <p className="truncate text-[10px] text-muted-foreground">
          {filtered ? `${showing.toLocaleString()} of ` : ""}
          {count(shelf.books, "book")} · {shelf.chunks.toLocaleString()} chunks ·{" "}
          {bytes(shelf.bytes)}
        </p>
      </div>

      <Tooltip label="set the whole shelf aside, or remove it">
        <Button
          size="icon"
          variant="ghost"
          className="size-7 shrink-0 text-muted-foreground hover:text-destructive"
          aria-label={`set aside or remove the ${shelf.path || shelf.name} shelf`}
          onClick={(event) => {
            event.stopPropagation();
            onRemove();
          }}
        >
          <Trash2 className="size-3.5" />
        </Button>
      </Tooltip>

      <div className="hidden w-44 shrink-0 flex-col gap-0.5 lg:flex">
        {models.map((model) => {
          const live = shelf.live_chunks[model] ?? 0;
          const done = shelf.embedded[model] ?? 0;
          return (
            <span key={model} className="flex items-center gap-1.5">
              <span className="w-20 truncate font-mono text-[10px] text-muted-foreground">
                {shortModel(model)}
              </span>
              <CoverageBar fraction={live ? done / live : 0} />
            </span>
          );
        })}
      </div>
    </div>
  );
}

function Book({
  book,
  models,
  chosen,
  onToggle,
  onOpen,
  shared,
}: {
  book: BookRow;
  models: string[];
  chosen: boolean;
  onToggle: () => void;
  onOpen: () => void;
  /** Another book on this screen answers to the same name. */
  shared: boolean;
}) {
  // The file is gone and the chunks remain: search will still rank this book
  // and hand back `text: null`, so it is worth saying here rather than there.
  const missing = book.sources.filter((source) => !source.present);
  const size = book.sources.reduce((total, source) => total + source.size_bytes, 0);

  return (
    <div
      onClick={onToggle}
      className={cn(
        "flex h-full cursor-pointer items-center gap-3 border-b pl-9 pr-3 text-sm transition-colors hover:bg-accent/40",
        chosen && "bg-accent/60",
        // Present, listed, and out of every search. Greying it is the whole
        // signal: a set-aside book that looked like the others would be the
        // second-best way to make results inexplicable.
        book.set_aside && "opacity-50",
      )}
    >
      <Checkbox checked={chosen} onCheckedChange={onToggle} />

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className={cn("truncate font-medium", book.set_aside && "line-through")}>
            {book.label ?? book.title}
          </span>
          {book.set_aside && (
            <Tooltip label="set aside: still here, still embedded, and returned by no search. Open it to put it back.">
              <Badge variant="outline" className="shrink-0 font-normal">
                <Archive className="size-2.5" /> set aside
              </Badge>
            </Tooltip>
          )}
          {shared && (
            // The shelf it came from, which is what actually tells these apart.
            // A name alone is not enough to cite by and not enough to pick by.
            <Tooltip label="another book here has this name — these are different works, and only the path says which is which. Give one a name to tell them apart.">
              <Badge variant="warning" className="shrink-0 font-mono">
                {book.shelf || "root"}
              </Badge>
            </Tooltip>
          )}
          {book.note && (
            <Tooltip label={book.note}>
              <Info className="size-3 shrink-0 text-muted-foreground" />
            </Tooltip>
          )}
          {missing.length > 0 && (
            <Tooltip label="the file is gone; its chunks remain, so a search can rank it and return no text">
              <Badge variant="danger">
                <FileX2 className="size-2.5" /> file missing
              </Badge>
            </Tooltip>
          )}
        </div>
        {/* The path, not the title: titles collide, and one library can hold
            two different books under one. This is what attribution is from. */}
        <p className="truncate font-mono text-[10px] text-muted-foreground">{book.key}</p>
      </div>

      <div className="hidden w-24 shrink-0 text-right text-[11px] text-muted-foreground md:block">
        <div className="tabular-nums">{count(book.chunks, "chunk")}</div>
        <div className="tabular-nums">{bytes(size)}</div>
      </div>

      <Tooltip label="read it, name it, set it aside">
        <Button
          size="sm"
          variant="ghost"
          className="h-7 shrink-0 px-2"
          onClick={(event) => {
            // The row toggles the scope checkbox; this must not do both.
            event.stopPropagation();
            onOpen();
          }}
        >
          Open
        </Button>
      </Tooltip>

      <div className="hidden w-44 shrink-0 flex-col gap-0.5 lg:flex">
        {models.map((model) => {
          const live = book.live_chunks[model] ?? 0;
          const done = book.embedded[model] ?? 0;
          return (
            <span key={model} className="flex items-center gap-1.5">
              <span className="w-20 truncate font-mono text-[10px] text-muted-foreground">
                {shortModel(model)}
              </span>
              <CoverageBar fraction={live ? done / live : 0} />
            </span>
          );
        })}
      </div>
    </div>
  );
}
