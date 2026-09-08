import * as React from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { FileX2, Search, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Tooltip } from "@/components/ui/tooltip";
import { CoverageBar, shortModel } from "@/components/coverage";
import type { BookRow } from "@/lib/api";
import { useDebounced } from "@/lib/debounce";
import { useBooks } from "@/lib/queries";
import { useSelection } from "@/lib/selection";
import { bytes, cn } from "@/lib/utils";

const ROW = 56;

/**
 * What is in the library, and what the next question will be asked of.
 *
 * Virtualised because `neuro` holds 3,453 books and this is a list a person
 * scrolls rather than pages. Filtered on the **server**, through the same
 * matcher `-c` uses, so that what the box finds and what selecting it means are
 * one thing rather than two that agree most of the time.
 */
export function Books({ library }: { library: string }) {
  const [filter, setFilter] = React.useState("");
  const pattern = useDebounced(filter);
  const books = useBooks(library, pattern);
  const { scope, setScope } = useSelection();
  const viewport = React.useRef<HTMLDivElement>(null);

  const rows = books.data?.books ?? [];
  const chosen = React.useMemo(() => new Set(scope), [scope]);

  const virtual = useVirtualizer({
    count: rows.length,
    getScrollElement: () => viewport.current,
    estimateSize: () => ROW,
    overscan: 12,
  });

  const shownKeys = rows.map((row) => row.key);
  const allShown = shownKeys.length > 0 && shownKeys.every((key) => chosen.has(key));
  const someShown = !allShown && shownKeys.some((key) => chosen.has(key));

  function toggle(key: string) {
    setScope(chosen.has(key) ? scope.filter((each) => each !== key) : [...scope, key]);
  }

  function toggleShown() {
    setScope(
      allShown
        ? scope.filter((each) => !shownKeys.includes(each))
        : [...new Set([...scope, ...shownKeys])],
    );
  }

  const models = Object.keys(rows[0]?.live_chunks ?? {});

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder="Title or path — a shelf (Talmud/), a work (Kandel), a glob (*Imaging*)"
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

        {pattern && rows.length > 0 && (
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
        <Checkbox checked={allShown} indeterminate={someShown} onCheckedChange={toggleShown} />
        <span>
          {books.isLoading
            ? "reading…"
            : `${rows.length.toLocaleString()} book${rows.length === 1 ? "" : "s"}${
                pattern ? " matching" : ""
              }`}
        </span>
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
          {virtual.getVirtualItems().map((item) => (
            <div
              key={rows[item.index].key}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: item.size,
                transform: `translateY(${item.start}px)`,
              }}
            >
              <Book
                book={rows[item.index]}
                models={models}
                chosen={chosen.has(rows[item.index].key)}
                onToggle={() => toggle(rows[item.index].key)}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Book({
  book,
  models,
  chosen,
  onToggle,
}: {
  book: BookRow;
  models: string[];
  chosen: boolean;
  onToggle: () => void;
}) {
  // The file is gone and the chunks remain: search will still rank this book
  // and hand back `text: null`, so it is worth saying here rather than there.
  const missing = book.sources.filter((source) => !source.present);
  const size = book.sources.reduce((total, source) => total + source.size_bytes, 0);

  return (
    <div
      onClick={onToggle}
      className={cn(
        "flex h-full cursor-pointer items-center gap-3 border-b px-3 text-sm transition-colors hover:bg-accent/40",
        chosen && "bg-accent/60",
      )}
    >
      <Checkbox checked={chosen} onCheckedChange={onToggle} />

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate font-medium">{book.title}</span>
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
        <div className="tabular-nums">{book.chunks.toLocaleString()} chunks</div>
        <div className="tabular-nums">{bytes(size)}</div>
      </div>

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
