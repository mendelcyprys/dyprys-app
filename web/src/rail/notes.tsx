import * as React from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpen, FileWarning } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { DyprysError } from "@/lib/api";
import { useNotes } from "@/lib/queries";

/**
 * The library's own NOTES.md, rendered and never interpreted.
 *
 * This is the one piece of corpus knowledge the tool cannot derive: which shelf
 * `-c` cuts along cleanly, two subjects whose vocabulary collides, a title that
 * means two different books. Nobody discovers those from results — they notice,
 * several bad searches later, that they should have read this. Which is why it
 * is a button in the rail and a line above the first search, not a file path.
 */
export function NotesDialog({
  library,
  open,
  onOpenChange,
}: {
  library: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const notes = useNotes(library, open);
  const missing = notes.error instanceof DyprysError && notes.error.status === 404;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent side="right" className="flex flex-col gap-0 overflow-hidden">
        <DialogTitle className="flex items-center gap-2">
          <BookOpen className="size-4" /> {library} — notes
        </DialogTitle>
        <DialogDescription>
          What this collection knows about itself. Written by whoever built it; the tool only
          carries it.
        </DialogDescription>

        <div className="mt-5 flex-1 overflow-y-auto pr-2">
          {notes.isLoading && <p className="text-sm text-muted-foreground">Reading…</p>}
          {missing && (
            <p className="flex items-start gap-2 text-sm text-muted-foreground">
              <FileWarning className="mt-0.5 size-4 shrink-0" />
              This library has no NOTES.md. That is not a problem — but if a search here teaches you
              something the index cannot say, it belongs in one.
            </p>
          )}
          {notes.data && (
            <article className="prose-dyp text-sm leading-relaxed">
              <Markdown remarkPlugins={[remarkGfm]}>{notes.data}</Markdown>
            </article>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

/**
 * A single dismissible line, once per library per session.
 *
 * Notes only help if they are read *before* searching, and nobody opens a
 * second pane to check. Session storage rather than local: a new session is
 * exactly when re-reading is worth a nudge.
 */
export function NotesNudge({ library, onRead }: { library: string; onRead: () => void }) {
  const stored = `dyprys.notes-seen.${library}`;
  const [seen, setSeen] = React.useState(() => Boolean(window.sessionStorage.getItem(stored)));

  if (seen) return null;

  const dismiss = () => {
    window.sessionStorage.setItem(stored, "1");
    setSeen(true);
  };

  return (
    <div className="flex items-center gap-2 rounded-md border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-xs">
      <BookOpen className="size-3.5 shrink-0 text-sky-400" />
      <span className="flex-1">
        <strong className="font-medium">{library}</strong> has notes — vocabulary, shelves and
        collisions the index cannot tell you.
      </span>
      <Button
        size="sm"
        variant="ghost"
        className="h-6 px-2 text-xs"
        onClick={() => {
          onRead();
          dismiss();
        }}
      >
        Read
      </Button>
      <Button size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={dismiss}>
        Later
      </Button>
    </div>
  );
}
