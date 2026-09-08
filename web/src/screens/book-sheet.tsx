import * as React from "react";
import { BookOpen, FileX2, Loader2, Pencil } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import type { BookRow } from "@/lib/api";
import { useDescribeBook } from "@/lib/queries";
import { bytes, cn, count } from "@/lib/utils";

/**
 * One book: what it is, what to call it, and a way into reading it.
 *
 * The naming is the part worth explaining. `title` is derived from the filename
 * and ingest rewrites it whenever the file moves, so a chosen name is stored
 * separately and neither overwrites the other — the file keeps its name and the
 * book gets one. Neither is identity: the **key** is, because titles collide and
 * one library can hold two different works under the same one.
 *
 * A label is not decoration. `-c` matches it, so naming a book is what makes it
 * possible to scope a search by saying what you call it rather than by finding
 * its path.
 */
export function BookSheet({
  library,
  book,
  onClose,
  onRead,
  onScope,
}: {
  library: string;
  book: BookRow | null;
  onClose: () => void;
  onRead: (book: BookRow) => void;
  onScope: (book: BookRow) => void;
}) {
  const describe = useDescribeBook(library);
  const [label, setLabel] = React.useState("");
  const [note, setNote] = React.useState("");

  // Reset to whatever the server holds each time a different book opens, so an
  // abandoned edit never leaks onto the next book.
  React.useEffect(() => {
    setLabel(book?.label ?? "");
    setNote(book?.note ?? "");
  }, [book?.key, book?.label, book?.note]);

  if (!book) return null;

  const missing = book.sources.filter((source) => !source.present);
  const readable = book.sources.find((source) => source.present);
  const size = book.sources.reduce((total, source) => total + source.size_bytes, 0);
  const dirty = label !== (book.label ?? "") || note !== (book.note ?? "");

  return (
    <Dialog open onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="max-w-2xl">
        <DialogTitle className="pr-6 text-base">{book.label ?? book.title}</DialogTitle>

        <div className="space-y-4 text-sm">
          <div className="space-y-1">
            <p className="break-all font-mono text-[10px] text-muted-foreground">{book.key}</p>
            <p className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <span className="tabular-nums">{count(book.chunks, "chunk")}</span>
              <span>·</span>
              <span className="tabular-nums">{bytes(size)}</span>
              <span>·</span>
              <span>{count(book.sources.length, "file")}</span>
              {missing.length > 0 && (
                <Badge variant="danger">
                  <FileX2 className="size-2.5" /> {missing.length} missing from disk
                </Badge>
              )}
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              variant="outline"
              disabled={!readable}
              onClick={() => onRead(book)}
              title={readable ? undefined : "every file of this book is gone from disk"}
            >
              <BookOpen /> Read it
            </Button>
            <Button size="sm" variant="outline" onClick={() => onScope(book)}>
              Ask only this book
            </Button>
          </div>

          <div className="space-y-3 border-t pt-4">
            <label className="flex flex-col gap-1">
              <span className="flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
                <Pencil className="size-3" /> Call it
              </span>
              <Input
                value={label}
                onChange={(event) => setLabel(event.target.value)}
                placeholder={book.title}
                className="h-9 text-sm"
              />
              <span className="text-[10px] leading-snug text-muted-foreground">
                A display name. The file is not renamed and keeps its own —{" "}
                <span className="font-mono">{book.title}</span>. <code>-c</code> matches this too,
                so a book you have named is one you can scope to by name.
              </span>
            </label>

            <label className="flex flex-col gap-1">
              <span className="text-[11px] font-medium text-muted-foreground">Context</span>
              <textarea
                value={note}
                onChange={(event) => setNote(event.target.value)}
                rows={3}
                placeholder="Which edition, why it is here, what its vocabulary is, how the text was extracted…"
                className={cn(
                  "w-full resize-y rounded-md border border-input bg-transparent px-3 py-2",
                  "text-xs shadow-sm placeholder:text-muted-foreground",
                  "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                )}
              />
              <span className="text-[10px] leading-snug text-muted-foreground">
                Shown on every result from this book. For what the index cannot tell you and a
                reader will not guess — the library’s own <code>NOTES.md</code>, one book down.
              </span>
            </label>

            {describe.error && (
              <p className="text-xs text-destructive">{(describe.error as Error).message}</p>
            )}

            <div className="flex items-center gap-2">
              <Button
                size="sm"
                disabled={!dirty || describe.isPending}
                onClick={() =>
                  describe.mutate({ key: book.key, label: label.trim(), note: note.trim() })
                }
              >
                {describe.isPending && <Loader2 className="animate-spin" />} Save
              </Button>
              {dirty && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setLabel(book.label ?? "");
                    setNote(book.note ?? "");
                  }}
                >
                  Undo
                </Button>
              )}
              {!dirty && (book.label || book.note) && (
                <span className="text-[11px] text-muted-foreground">Saved.</span>
              )}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
