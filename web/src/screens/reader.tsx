import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Copy, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Tooltip } from "@/components/ui/tooltip";
import { api, type SourceWindow } from "@/lib/api";

/** One stretch to fetch. A screenful and a bit, so scrolling is never starved. */
const STRETCH = 6000;
/** How close to an edge before the next stretch is fetched. */
const NEAR = 800;

export interface Reading {
  path: string;
  offset: number;
  book: string;
  /** The passage as the search returned it, so it can be found in the window. */
  text: string | null;
}

/**
 * A passage where it actually lives, read continuously.
 *
 * The first version paged: Earlier and Later re-requested a 4,000-byte window
 * at a new offset and replaced what was on screen. That is a defensible way to
 * *look at* a citation and a bad way to *read*, because every move throws away
 * where you were — you cannot scroll from the passage into the paragraph after
 * it, and the text jumps rather than continues.
 *
 * So this keeps one growing byte range and extends it at whichever edge you
 * reach. Two details make that honest rather than approximately right:
 *
 *   * Each stretch continues from the previous one's **`end`**, which the server
 *     now reports. `read_window` snaps its edges to sentence boundaries, so
 *     asking again at `offset + span` skips exactly what the snap trimmed — a
 *     sentence lost at every join, silently. `end` is where the bytes actually
 *     stopped.
 *   * Growing upward would push the text down under the reader's eye, so the
 *     scroll position is corrected by the height that was added, in a layout
 *     effect, before the browser paints.
 *
 * It is still a window and not a document: `MAX_SPAN` caps the server at 200 KB
 * per request, which is why the range grows a stretch at a time instead of
 * asking for the book.
 */
export function Reader({
  library,
  reading,
  onClose,
}: {
  library: string;
  reading: Reading | null;
  onClose: () => void;
}) {
  // Every stretch loaded, in file order and contiguous. Not a single string:
  // each carries its own byte range, which is what the next request is built
  // from and what the footer counts.
  const [loaded, setLoaded] = React.useState<SourceWindow[]>([]);
  const [pending, setPending] = React.useState<"up" | "down" | null>(null);
  const [failed, setFailed] = React.useState<string | null>(null);
  const [copied, setCopied] = React.useState<"yes" | "no" | null>(null);
  const [copiedPath, setCopiedPath] = React.useState<"yes" | "no" | null>(null);

  const viewport = React.useRef<HTMLDivElement>(null);
  const passage = React.useRef<HTMLElement>(null);
  // Height of the scrolling content before a prepend, so the correction after
  // one is a measurement rather than an estimate.
  const grew = React.useRef<number | null>(null);

  const first = loaded[0];
  const last = loaded[loaded.length - 1];
  const atStart = !first || first.offset <= 0;
  const atEnd = !last || (last.bytes !== null && last.end >= last.bytes);

  // The opening read: a stretch *before* the passage as well as at it, so the
  // passage lands in the middle of something rather than at the top of nothing.
  const opening = useQuery({
    queryKey: ["source", library, reading?.path, reading?.offset] as const,
    queryFn: async () => {
      const from = Math.max(0, (reading?.offset ?? 0) - STRETCH);
      const head = await api.source(library, reading!.path, from, STRETCH);
      // At the very start of a file the two coincide; one read is the whole of it.
      if (head.end >= (reading?.offset ?? 0)) return [head];
      return [head, await api.source(library, reading!.path, head.end, STRETCH * 2)];
    },
    enabled: Boolean(reading),
    staleTime: Infinity,
  });

  React.useEffect(() => {
    setLoaded(opening.data ?? []);
    setFailed(null);
  }, [opening.data]);

  // Put the passage under the eye once its stretch is on screen, and only then.
  React.useEffect(() => {
    if (!loaded.length) return;
    const id = window.setTimeout(
      () => passage.current?.scrollIntoView({ block: "center", behavior: "auto" }),
      0,
    );
    return () => window.clearTimeout(id);
  }, [loaded.length > 0, reading?.offset]);

  // Restore the reading position after a prepend, before paint. Without this,
  // reaching the top pushes everything down by however much arrived and the
  // sentence being read walks off the bottom of the pane.
  React.useLayoutEffect(() => {
    const pane = viewport.current;
    if (!pane || grew.current === null) return;
    pane.scrollTop += pane.scrollHeight - grew.current;
    grew.current = null;
  }, [loaded]);

  async function extend(where: "up" | "down") {
    if (pending || !reading) return;
    if (where === "up" ? atStart : atEnd) return;
    setPending(where);
    try {
      if (where === "down") {
        const next = await api.source(library, reading.path, last.end, STRETCH);
        // A stretch with nothing in it is the end of the file by another name.
        setLoaded((all) => (next.text ? [...all, next] : all));
      } else {
        const from = Math.max(0, first.offset - STRETCH);
        const next = await api.source(library, reading.path, from, first.offset - from);
        grew.current = viewport.current?.scrollHeight ?? null;
        setLoaded((all) => (next.text ? [next, ...all] : all));
      }
      setFailed(null);
    } catch (thrown) {
      setFailed((thrown as Error).message);
    } finally {
      setPending(null);
    }
  }

  function onScroll(event: React.UIEvent<HTMLDivElement>) {
    const pane = event.currentTarget;
    if (pane.scrollTop < NEAR) void extend("up");
    else if (pane.scrollHeight - pane.scrollTop - pane.clientHeight < NEAR) void extend("down");
  }

  const citation = reading ? `${reading.path}:${reading.offset}` : "";
  // Only to decide whether the footer can say anything true at all.
  const measured = last?.bytes != null;

  return (
    <Dialog open={Boolean(reading)} onOpenChange={(open) => !open && onClose()}>
      <DialogContent side="right" className="flex flex-col gap-0 overflow-hidden">
        <DialogTitle className="pr-8">{reading?.book}</DialogTitle>
        {/* The passage's own byte, always — this is the citation, and it is
            what Copy gives you. It does not move as you read on, because a
            citation names where the passage is, not where you scrolled to. */}
        <p className="mt-1 break-all font-mono text-[10px] text-muted-foreground">
          {reading?.path}:{reading?.offset}
        </p>

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() =>
              void copy(citation, (ok) => {
                setCopied(ok ? "yes" : "no");
                window.setTimeout(() => setCopied(null), 2500);
              })
            }
          >
            <Copy />{" "}
            {copied === "yes"
              ? "Copied"
              : copied === "no"
                ? "Could not copy — select it instead"
                : "Copy citation"}
          </Button>
          <Tooltip label="scroll back to the passage this result was">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => passage.current?.scrollIntoView({ block: "center" })}
              disabled={!reading?.text}
            >
              Back to the passage
            </Button>
          </Tooltip>
          <Tooltip label="the path alone, to open the file yourself">
            <Button
              size="sm"
              variant="ghost"
              onClick={() =>
                // The path, not `path:offset`: this one is for a terminal or a
                // file dialog. A `file://` link would be the obvious control
                // here and is inert — a page served over http may not navigate
                // to one, silently — so it is a copy button instead.
                void copy(reading?.path ?? "", (ok) => {
                  setCopiedPath(ok ? "yes" : "no");
                  window.setTimeout(() => setCopiedPath(null), 2500);
                })
              }
            >
              {copiedPath === "yes"
                ? "Path copied"
                : copiedPath === "no"
                  ? "Could not copy"
                  : "Copy path"}
            </Button>
          </Tooltip>
        </div>

        <div ref={viewport} onScroll={onScroll} className="mt-4 flex-1 overflow-y-auto pr-2">
          {opening.isLoading && <p className="text-sm text-muted-foreground">Reading…</p>}
          {opening.error && (
            <p className="text-sm text-destructive">{(opening.error as Error).message}</p>
          )}

          <Edge
            shown={loaded.length > 0}
            done={atStart}
            busy={pending === "up"}
            label="the beginning of this file"
          />

          {loaded.map((stretch, index) => (
            <p
              key={stretch.offset}
              className="whitespace-pre-wrap text-sm leading-relaxed"
              // Stretches are joined by the whitespace `read_window` stripped
              // from each edge; without this the last word of one runs into the
              // first of the next.
              style={{ marginTop: index === 0 ? 0 : "1em" }}
            >
              <Highlighted text={stretch.text} find={reading?.text ?? null} mark={passage} />
            </p>
          ))}

          <Edge
            shown={loaded.length > 0}
            done={atEnd}
            busy={pending === "down"}
            label="the end of this file"
          />

          {failed && <p className="py-3 text-xs text-destructive">{failed}</p>}
        </div>

        {measured && (
          // The range on screen, not a progress bar. "72% through the file" was
          // said while sitting at byte 0 of it -- true of the loaded range's far
          // edge and false of where the reader was.
          <p className="mt-2 shrink-0 border-t pt-2 text-[10px] text-muted-foreground">
            showing bytes {first.offset.toLocaleString()}–{last.end.toLocaleString()} of{" "}
            {last.bytes?.toLocaleString()}
            {atEnd && atStart ? " — the whole file" : ""}
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}

/** What is above or below the loaded text: more of it, or the end of the file. */
function Edge({
  shown,
  done,
  busy,
  label,
}: {
  shown: boolean;
  done: boolean;
  busy: boolean;
  label: string;
}) {
  if (!shown) return null;
  return (
    <p className="flex items-center justify-center gap-1.5 py-3 text-[10px] text-muted-foreground">
      {busy ? (
        <>
          <Loader2 className="size-3 animate-spin" /> reading on…
        </>
      ) : done ? (
        label
      ) : (
        "keep scrolling"
      )}
    </p>
  );
}

/**
 * Copy, and say so only if it worked.
 *
 * `writeText` rejects rather than throws, and an unhandled rejection is a
 * console error nobody reads plus a button that says "Copied" when nothing was.
 * It refuses on an unfocused document and on any non-secure origin that is not
 * localhost — so a server reached over a LAN address hits this every time.
 */
async function copy(text: string, said: (ok: boolean) => void): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
    said(true);
  } catch {
    said(false);
  }
}

/**
 * The passage, marked inside its surroundings.
 *
 * Located by matching the text the search returned rather than by arithmetic on
 * the offset: the window is snapped to sentence boundaries, so its start is not
 * the byte that was asked for, and a byte offset is not a string index once
 * anything is not ASCII. When it cannot be found, nothing is highlighted —
 * marking the wrong span would be worse than marking none.
 *
 * `mark` is attached to the highlight so the pane can scroll back to it. Only
 * the stretch that actually contains the passage claims the ref; the others
 * leave it alone rather than each pointing it at themselves.
 */
function Highlighted({
  text,
  find,
  mark,
}: {
  text: string;
  find: string | null;
  mark: React.RefObject<HTMLElement | null>;
}) {
  if (!find) return <>{text}</>;
  const needle = find.trim().slice(0, 300);
  const at = text.indexOf(needle);
  if (at < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, at)}
      <mark ref={mark} className="rounded bg-amber-400/25 px-0.5 text-foreground">
        {text.slice(at, at + needle.length)}
      </mark>
      {text.slice(at + needle.length)}
    </>
  );
}
