import * as React from "react";
import { BookOpen, CircleDot, Database, Layers } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip } from "@/components/ui/tooltip";
import type { Libraries } from "@/lib/api";
import { useHealth } from "@/lib/queries";
import { useSelection } from "@/lib/selection";
import { LibraryPicker } from "./library-picker";
import { NotesDialog } from "./notes";

/**
 * The state of the question: which library, and (later) which model and which
 * books. It sits outside the tabs because those three choices outlive any one
 * search, and every tab means something different depending on them.
 */
export function Rail({ libraries }: { libraries: Libraries }) {
  const { library, select } = useSelection();
  const [notesOpen, setNotesOpen] = React.useState(false);
  const health = useHealth();
  const current = libraries.libraries.find((row) => row.name === library);
  const warm = (library && health.data?.loaded[library]) || [];

  return (
    <aside className="flex w-72 shrink-0 flex-col gap-4 border-r bg-card/40 p-4">
      <div className="flex items-center gap-2">
        <Layers className="size-4 text-muted-foreground" />
        <span className="text-sm font-semibold tracking-tight">dyprys</span>
      </div>

      <LibraryPicker libraries={libraries.libraries} selected={library} onSelect={(name) => select(name || null)} />

      {current?.exists && (
        <div className="space-y-3">
          <dl className="space-y-1.5 text-xs">
            <Row icon={<Database className="size-3" />} label="books" value={current.books ?? 0} />
            <Row
              icon={<Layers className="size-3" />}
              label="chunks"
              value={(current.chunks ?? 0).toLocaleString()}
            />
            <Row
              icon={<CircleDot className={warm.length ? "size-3 text-emerald-500" : "size-3"} />}
              label="warm"
              value={warm.length ? `${warm.length} model${warm.length > 1 ? "s" : ""}` : "none"}
            />
          </dl>

          <Button
            variant="outline"
            size="sm"
            className="w-full justify-start"
            onClick={() => setNotesOpen(true)}
          >
            <BookOpen /> Notes
            {current.notes && <Badge variant="outline" className="ml-auto">has some</Badge>}
          </Button>

          <p className="font-mono text-[10px] leading-relaxed text-muted-foreground">
            {current.path}
          </p>
        </div>
      )}

      <div className="mt-auto space-y-1">
        <Tooltip label={libraries.registry_path}>
          <p className="cursor-default truncate text-[10px] text-muted-foreground">
            registry: {libraries.registry_path.split("/").slice(-1)[0]}
          </p>
        </Tooltip>
      </div>

      {library && (
        <NotesDialog library={library} open={notesOpen} onOpenChange={setNotesOpen} />
      )}
    </aside>
  );
}

function Row({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-muted-foreground">{icon}</span>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="ml-auto tabular-nums">{value}</dd>
    </div>
  );
}
