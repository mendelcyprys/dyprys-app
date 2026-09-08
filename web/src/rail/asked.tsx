import { History } from "lucide-react";
import { Tooltip } from "@/components/ui/tooltip";
import { useAsked } from "@/lib/queries";
import { usePending } from "@/lib/pending";
import { useSelection } from "@/lib/selection";

/**
 * What this library has already been asked, from either frontend.
 *
 * `dyp asked` is cross-session and cross-interface memory: a question typed in
 * a terminal shows up here, and so does one asked by an agent through `--json`.
 * Clicking runs it again rather than opening the stored answer, because the
 * index moves and the stored hits are what it said last time.
 */
export function Asked({ onRun }: { onRun: () => void }) {
  const { library } = useSelection();
  const asked = useAsked(library);
  const { ask } = usePending();
  const rows = asked.data?.questions ?? [];

  if (rows.length === 0) return null;

  return (
    <div className="min-h-0 flex-1 space-y-1.5 overflow-hidden">
      <span className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
        <History className="size-3" /> Asked
      </span>
      <ul className="max-h-48 space-y-0.5 overflow-y-auto pr-1">
        {rows.map((row) => (
          <li key={row.id}>
            <Tooltip label={`${row.question} — ${when(row.at)}, ${row.mode}`}>
              <button
                onClick={() => {
                  ask(row.question);
                  onRun();
                }}
                className="w-full truncate rounded px-1.5 py-1 text-left text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                {row.question}
              </button>
            </Tooltip>
          </li>
        ))}
      </ul>
    </div>
  );
}

function when(at: string): string {
  const then = new Date(at);
  const days = Math.floor((Date.now() - then.getTime()) / 86_400_000);
  if (days === 0) return then.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (days === 1) return "yesterday";
  if (days < 30) return `${days} days ago`;
  return then.toLocaleDateString();
}
