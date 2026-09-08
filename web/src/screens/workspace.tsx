import * as React from "react";
import { Construction } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { NotesDialog, NotesNudge } from "@/rail/notes";
import type { LibraryRow } from "@/lib/api";
import { ModelCoverage } from "@/components/coverage";
import { Books } from "./books";

export const TABS = ["Ask", "Books", "Models", "Jobs"] as const;
export type Tab = (typeof TABS)[number];

/** The four tabs. Ask, Models and Jobs are phases 4, 3 and 5. */
export function Workspace({
  library,
  tab,
  onTab,
}: {
  library: LibraryRow;
  tab: Tab;
  onTab: (tab: Tab) => void;
}) {
  const [notesOpen, setNotesOpen] = React.useState(false);

  return (
    <main className="flex min-w-0 flex-1 flex-col">
      <nav className="flex items-center gap-1 border-b px-4">
        {TABS.map((each) => (
          <button
            key={each}
            onClick={() => onTab(each)}
            className={cn(
              "-mb-px border-b-2 px-3 py-2.5 text-sm transition-colors",
              each === tab
                ? "border-foreground font-medium"
                : "border-transparent text-muted-foreground hover:text-foreground",
            )}
          >
            {each}
          </button>
        ))}
      </nav>

      <div className="flex min-h-0 flex-1 flex-col gap-4 p-6">
        {library.notes && <NotesNudge library={library.name} onRead={() => setNotesOpen(true)} />}

        {tab === "Books" ? (
          // Keyed, so switching library resets the filter box with the
          // scope. A filter left behind reads as "this library holds nothing".
          <Books key={library.name} library={library.name} />
        ) : (
          <Unbuilt tab={tab} library={library} onNotes={() => setNotesOpen(true)} />
        )}
      </div>

      <NotesDialog library={library.name} open={notesOpen} onOpenChange={setNotesOpen} />
    </main>
  );
}

function Unbuilt({
  tab,
  library,
  onNotes,
}: {
  tab: Tab;
  library: LibraryRow;
  onNotes: () => void;
}) {
  return (
    <div className="space-y-4 overflow-y-auto">
      <div className="flex items-start gap-3 rounded-lg border border-dashed p-6">
        <Construction className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
        <div className="space-y-2">
          <p className="text-sm">
            <strong className="font-medium">{tab}</strong> is not built yet.
          </p>
          <p className="max-w-prose text-xs leading-relaxed text-muted-foreground">
            Libraries and books are done: which library, whether its index is really there, what is
            in it, and what the next question will be asked of. Models come next, then the search
            itself, then jobs.
          </p>
        </div>
      </div>

      <section className="space-y-3">
        <h2 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {library.name}
        </h2>
        <dl className="grid max-w-md grid-cols-[8rem_1fr] gap-x-4 gap-y-2 text-sm">
          <dt className="text-muted-foreground">books</dt>
          <dd className="tabular-nums">{library.books?.toLocaleString() ?? "—"}</dd>
          <dt className="text-muted-foreground">chunks</dt>
          <dd className="tabular-nums">{library.chunks?.toLocaleString() ?? "—"}</dd>
          <dt className="text-muted-foreground">embedded by</dt>
          <dd>
            <ModelCoverage models={library.models} />
          </dd>
          <dt className="text-muted-foreground">notes</dt>
          <dd>
            {library.notes ? (
              <button className="underline underline-offset-2" onClick={onNotes}>
                NOTES.md
              </button>
            ) : (
              <Badge variant="outline">none</Badge>
            )}
          </dd>
          <dt className="text-muted-foreground">index</dt>
          <dd className="break-all font-mono text-xs text-muted-foreground">{library.path}</dd>
        </dl>
      </section>
    </div>
  );
}
