import * as React from "react";
import { ChevronDown, ChevronUp, Copy, ExternalLink } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Tooltip } from "@/components/ui/tooltip";
import { api } from "@/lib/api";

/** Matches `service.SOURCE_SPAN`'s spirit: a screenful, not a book. */
const SPAN = 4000;

export interface Reading {
  path: string;
  offset: number;
  book: string;
  /** The passage as the search returned it, so it can be found in the window. */
  text: string | null;
}

/**
 * A passage where it actually lives.
 *
 * `GET /source` returns bytes around an offset, snapped to sentence boundaries
 * and capped at 200 KB, so this is a window rather than a document: moving is
 * re-requesting at a new offset. That is also why it is a pane and not a
 * viewer — the point is to read *around* a result, not to replace the file.
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
  // How far the window has been moved from the passage, not an absolute
  // position. Derived rather than synchronised, so the first render already
  // asks for the right bytes — initialising it in state fetched offset 0 (the
  // start of a 16 MB file) and showed it for a beat before an effect corrected.
  const [moved, setMoved] = React.useState<number | null>(null);
  const [copied, setCopied] = React.useState(false);
  const offset = moved ?? reading?.offset ?? 0;

  React.useEffect(() => setMoved(null), [reading?.path, reading?.offset]);

  const window_ = useQuery({
    queryKey: ["source", library, reading?.path, offset] as const,
    queryFn: () => api.source(library, reading!.path, offset, SPAN),
    enabled: Boolean(reading),
  });

  const citation = reading ? `${reading.path}:${reading.offset}` : "";

  return (
    <Dialog open={Boolean(reading)} onOpenChange={(open) => !open && onClose()}>
      <DialogContent side="right" className="flex flex-col gap-0 overflow-hidden">
        <DialogTitle className="pr-8">{reading?.book}</DialogTitle>
        {/* The passage's own byte, always — this is the citation, and it is
            what Copy gives you. It stays put while the window moves, because a
            citation names where the passage is, not where you scrolled to. */}
        <p className="mt-1 break-all font-mono text-[10px] text-muted-foreground">
          {reading?.path}:{reading?.offset}
        </p>
        {moved !== null && (
          <p className="mt-0.5 text-[10px] text-muted-foreground">
            showing from byte {(window_.data?.offset ?? offset).toLocaleString()}
          </p>
        )}

        <div className="mt-3 flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              navigator.clipboard?.writeText(citation);
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1500);
            }}
          >
            <Copy /> {copied ? "Copied" : "Copy citation"}
          </Button>
          <Tooltip label="move the window back through the file">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setMoved(Math.max(0, offset - SPAN))}
              disabled={offset === 0}
            >
              <ChevronUp /> Earlier
            </Button>
          </Tooltip>
          <Button size="sm" variant="ghost" onClick={() => setMoved(offset + SPAN)}>
            <ChevronDown /> Later
          </Button>
          <Tooltip label="the file itself, in whatever opens .txt here">
            <Button size="sm" variant="ghost" asChild>
              <a href={`file://${reading?.path ?? ""}`} target="_blank" rel="noreferrer">
                <ExternalLink />
              </a>
            </Button>
          </Tooltip>
        </div>

        <div className="mt-4 flex-1 overflow-y-auto pr-2">
          {window_.isLoading && <p className="text-sm text-muted-foreground">Reading…</p>}
          {window_.error && (
            <p className="text-sm text-destructive">{(window_.error as Error).message}</p>
          )}
          {window_.data && (
            <p className="whitespace-pre-wrap text-sm leading-relaxed">
              <Highlighted text={window_.data.text} find={reading?.text ?? null} />
            </p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

/**
 * The passage, marked inside its surroundings.
 *
 * Located by matching the text the search returned rather than by arithmetic on
 * the offset: the window is snapped to sentence boundaries, so its start is not
 * the byte that was asked for, and a byte offset is not a string index once
 * anything is not ASCII. When it cannot be found, nothing is highlighted —
 * marking the wrong span would be worse than marking none.
 */
function Highlighted({ text, find }: { text: string; find: string | null }) {
  if (!find) return <>{text}</>;
  const needle = find.trim().slice(0, 300);
  const at = text.indexOf(needle);
  if (at < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, at)}
      <mark className="rounded bg-amber-400/25 px-0.5 text-foreground">
        {text.slice(at, at + needle.length)}
      </mark>
      {text.slice(at + needle.length)}
    </>
  );
}
