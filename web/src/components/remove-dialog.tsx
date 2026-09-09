import * as React from "react";
import { AlertTriangle, Archive, ArchiveRestore, Loader2, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import type { Removal } from "@/lib/api";
import { api } from "@/lib/api";
import { useRemoval } from "@/lib/queries";
import { count } from "@/lib/utils";

/**
 * Taking books out of a library — the reversible way and the other one.
 *
 * Both in one dialog on purpose. They are the same intention arriving with
 * different amounts of certainty, and offering only the destructive one is how
 * a badly extracted book gets deleted along with the hours that embedded it.
 * So the reversible option is the one with the button, and the permanent one
 * has to be opened, previewed and typed.
 *
 * What each actually does is stated rather than implied:
 *
 * - **Set aside** — every vector, passage and BM25 row stays. No search reads
 *   them. One click back.
 * - **Remove** — the index forgets these books. Their chunk ids become dead
 *   space until `compact`, their vectors are gone, and putting one back means
 *   adding and embedding it again. **The files on disk are never touched.**
 *
 * Addressed by whole keys or a whole shelf directory, never by pattern: a glob
 * is right for a search, where a wrong guess costs a second look, and wrong
 * here.
 */
export function RemoveBooks({
  library,
  what,
  subject,
  books,
  setAside,
  open,
  onOpenChange,
  onRemoved,
}: {
  library: string;
  /** Exactly one of these. Keys for a selection, a directory for a whole shelf. */
  what: { keys?: string[]; shelf?: string };
  /** What to call the thing being removed, in a sentence. */
  subject: string;
  /** How many books that is, so the dialog can speak before the preview lands. */
  books: number;
  /** How many of them are already set aside — the difference between the verbs. */
  setAside: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called after a real removal, so a caller holding a selection can drop it. */
  onRemoved?: () => void;
}) {
  const { aside, remove } = useRemoval(library);
  const [preview, setPreview] = React.useState<Removal | null>(null);
  const [typed, setTyped] = React.useState("");
  const [failed, setFailed] = React.useState<string | null>(null);

  // A dialog that reopens holding the last preview and a filled-in confirmation
  // is one click from removing something nobody looked at.
  React.useEffect(() => {
    if (!open) {
      setPreview(null);
      setTyped("");
      setFailed(null);
    }
  }, [open]);

  const allAside = setAside >= books && books > 0;
  const someLive = books - setAside;

  async function look() {
    setFailed(null);
    try {
      setPreview(await api.removeBooks(library, what, false));
    } catch (thrown) {
      setFailed((thrown as Error).message);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogTitle className="pr-6 text-base">Remove {subject}</DialogTitle>
        <DialogDescription>
          {count(books, "book")}
          {setAside > 0 && `, ${setAside} already set aside`}.
        </DialogDescription>

        <div className="mt-4 space-y-4 text-sm">
          <section className="space-y-2 rounded-md border p-3">
            <h3 className="flex items-center gap-2 text-sm font-medium">
              {allAside ? (
                <ArchiveRestore className="size-3.5" />
              ) : (
                <Archive className="size-3.5" />
              )}
              {allAside ? "Put back" : "Set aside"}
              <span className="ml-auto rounded bg-secondary px-1.5 py-0.5 text-[10px] font-normal text-secondary-foreground">
                reversible
              </span>
            </h3>
            <p className="text-xs leading-relaxed text-muted-foreground">
              {allAside ? (
                <>Put {subject} back into the library. Every search sees it again.</>
              ) : (
                <>
                  Take {subject} out of every search and leave everything on disk. The vectors, the
                  passages and the keyword index all stay exactly where they are, costing nothing —
                  what changes is that no search reads them. This is the one to use for a book whose
                  extraction went wrong, or a shelf that swamps every answer.
                </>
              )}
            </p>
            <Button
              size="sm"
              variant="outline"
              disabled={aside.isPending || books === 0}
              onClick={() =>
                aside.mutate(
                  { ...what, aside: !allAside },
                  { onSuccess: () => onOpenChange(false) },
                )
              }
            >
              {aside.isPending && <Loader2 className="animate-spin" />}
              {allAside ? "Put back" : `Set aside${someLive < books ? ` ${someLive} left` : ""}`}
            </Button>
            {aside.error && (
              <p className="text-xs text-destructive">{(aside.error as Error).message}</p>
            )}
          </section>

          <section className="space-y-2 rounded-md border border-destructive/40 p-3">
            <h3 className="flex items-center gap-2 text-sm font-medium">
              <Trash2 className="size-3.5" /> Remove for good
              <span className="ml-auto rounded bg-destructive/15 px-1.5 py-0.5 text-[10px] font-normal text-destructive">
                cannot be undone
              </span>
            </h3>
            <p className="text-xs leading-relaxed text-muted-foreground">
              The index forgets {subject}. The vectors go with it, so putting one back means adding
              the files again and embedding them again — which is the difference from setting them
              aside.{" "}
              <strong className="font-medium text-foreground">
                The files on disk are not touched.
              </strong>
            </p>

            {!preview ? (
              <Button size="sm" variant="outline" onClick={look}>
                Show me what that removes
              </Button>
            ) : (
              <div className="space-y-2">
                <div className="rounded-md bg-destructive/10 p-2 text-xs">
                  <p className="flex items-center gap-1.5 font-medium">
                    <AlertTriangle className="size-3" />
                    {count(preview.count, "book")} and {preview.chunks.toLocaleString()} chunks
                  </p>
                  <ul className="mt-1 space-y-0.5 text-muted-foreground">
                    {preview.books.slice(0, 6).map((book) => (
                      <li key={book.key} className="truncate">
                        {book.title}
                      </li>
                    ))}
                    {preview.books.length > 6 && <li>… and {preview.books.length - 6} more</li>}
                  </ul>
                  <p className="mt-1.5 text-[11px] text-muted-foreground">
                    Their chunk ids become dead space until{" "}
                    <code className="font-mono">dyp compact</code> reclaims them.
                  </p>
                </div>

                <label className="flex flex-col gap-1">
                  <span className="text-[11px] text-muted-foreground">
                    Type <code className="font-mono font-semibold">remove</code> to confirm.
                  </span>
                  <Input
                    value={typed}
                    onChange={(event) => setTyped(event.target.value)}
                    placeholder="remove"
                    className="h-8 text-sm"
                    autoComplete="off"
                    spellCheck={false}
                  />
                </label>
                <Button
                  size="sm"
                  variant="destructive"
                  disabled={typed.trim().toLowerCase() !== "remove" || remove.isPending}
                  onClick={() =>
                    remove.mutate(
                      { ...what, confirm: true },
                      {
                        onSuccess: () => {
                          onRemoved?.();
                          onOpenChange(false);
                        },
                      },
                    )
                  }
                >
                  {remove.isPending && <Loader2 className="animate-spin" />}
                  Remove {count(preview.count, "book")}
                </Button>
              </div>
            )}
            {(failed || remove.error) && (
              <p className="text-xs text-destructive">
                {failed ?? (remove.error as Error).message}
              </p>
            )}
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
}
