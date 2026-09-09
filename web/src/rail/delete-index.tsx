import * as React from "react";
import { AlertTriangle, FileText, HardDrive, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { api, type IndexDeletion } from "@/lib/api";
import { useDeleteIndex } from "@/lib/queries";
import { bytes, count } from "@/lib/utils";

/**
 * Erasing a library's index — the one irreversible thing in this app.
 *
 * This deliberately had no route at all for a while, on the grounds that a
 * browser is the wrong place to confirm days of embedding away. What changed is
 * not the risk but the shape: the server previews before it acts, from the same
 * function the terminal prints from, so what a person confirms is the actual
 * file count and the actual size rather than a sentence someone wrote once.
 *
 * The distinction it exists to make is the one people get wrong. An index holds
 * the **vectors**, which took hours and cannot be recovered. The **text** is not
 * touched — and the preview says where it is, so that is checkable rather than
 * a promise.
 *
 * "Somewhere else" is the part that was not always true. A library added in
 * place keeps its books in the index directory, and this panel said they were
 * elsewhere while printing that directory's own path above it. The server now
 * reports `shares_directory` and what stays behind, so both layouts read as
 * what they are.
 */
export function DeleteIndex({
  name,
  open,
  onOpenChange,
  onDeleted,
}: {
  name: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDeleted: () => void;
}) {
  const remove = useDeleteIndex();
  const [plan, setPlan] = React.useState<IndexDeletion | null>(null);
  const [typed, setTyped] = React.useState("");
  const [failed, setFailed] = React.useState<string | null>(null);

  // Fetched on open, not on mount: the numbers are only true now, and a dialog
  // that reopens holding a stale preview and a filled-in confirmation is one
  // click away from deleting something nobody looked at.
  React.useEffect(() => {
    if (!open) {
      setPlan(null);
      setTyped("");
      setFailed(null);
      return;
    }
    let live = true;
    api
      .deleteIndex(name, false)
      .then((result) => live && setPlan(result))
      .catch((thrown) => live && setFailed((thrown as Error).message));
    return () => {
      live = false;
    };
  }, [open, name]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogTitle className="pr-6 text-base">Delete the {name} index</DialogTitle>
        <DialogDescription>
          This cannot be undone. Forgetting the name instead leaves everything on disk.
        </DialogDescription>

        <div className="mt-4 space-y-3 text-sm">
          {failed && <p className="text-xs text-destructive">{failed}</p>}
          {!plan && !failed && (
            <p className="flex items-center gap-2 text-xs text-muted-foreground">
              <Loader2 className="size-3 animate-spin" /> Measuring…
            </p>
          )}

          {plan && (
            <>
              <div className="space-y-1 rounded-md border border-destructive/40 bg-destructive/10 p-3">
                <p className="flex items-center gap-1.5 text-xs font-medium">
                  <HardDrive className="size-3" /> Deleted
                </p>
                <p className="break-all font-mono text-[10px]">{plan.path}</p>
                <p className="text-xs text-muted-foreground">
                  {count(plan.files, "file")}, {bytes(plan.bytes)} — the database, its vectors and
                  its routing profile. Rebuilding them means embedding the library again.
                </p>
              </div>

              <div className="space-y-1 rounded-md border p-3">
                <p className="flex items-center gap-1.5 text-xs font-medium">
                  <FileText className="size-3" /> Kept
                </p>
                {plan.sources ? (
                  <>
                    <p className="break-all font-mono text-[10px]">{plan.sources}</p>
                    {/* Which of these is true depends on the layout, and the
                        one that was printed unconditionally was the wrong one
                        on a library added in place: it read "they are somewhere
                        else" directly under the path being deleted. */}
                    {plan.shares_directory ? (
                      <p className="text-xs text-muted-foreground">
                        The books themselves — and they are{" "}
                        <strong className="font-medium text-foreground">in that directory</strong>,
                        not somewhere else. Only the index files above are removed;{" "}
                        {count(plan.kept_files, "file")} ({bytes(plan.kept_bytes)}) stay where they
                        are, and the directory itself stays with them.
                      </p>
                    ) : (
                      <p className="text-xs text-muted-foreground">
                        The books themselves. They are somewhere else, and nothing here touches them
                        —<code className="font-mono"> dyp add</code> on that directory starts the
                        library again.
                      </p>
                    )}
                  </>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    This index names no source files, so there is nothing to spare — it may never
                    have been added to.
                  </p>
                )}
              </div>

              <label className="flex flex-col gap-1">
                <span className="text-[11px] text-muted-foreground">
                  Type <code className="font-mono font-semibold">{plan.name}</code> to confirm.
                </span>
                <Input
                  value={typed}
                  onChange={(event) => setTyped(event.target.value)}
                  placeholder={plan.name}
                  className="h-8 text-sm"
                  autoComplete="off"
                  spellCheck={false}
                />
              </label>

              {remove.error && (
                <p className="text-xs text-destructive">{(remove.error as Error).message}</p>
              )}

              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="destructive"
                  disabled={typed.trim() !== plan.name || remove.isPending}
                  onClick={() =>
                    remove.mutate(
                      { name: plan.name, confirm: true },
                      {
                        onSuccess: () => {
                          onDeleted();
                          onOpenChange(false);
                        },
                      },
                    )
                  }
                >
                  {remove.isPending ? <Loader2 className="animate-spin" /> : <AlertTriangle />}
                  Delete {bytes(plan.bytes)} of index
                </Button>
                <Button size="sm" variant="ghost" onClick={() => onOpenChange(false)}>
                  Keep it
                </Button>
              </div>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
