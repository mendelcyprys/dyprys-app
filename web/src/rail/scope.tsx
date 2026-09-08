import { Layers3, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip } from "@/components/ui/tooltip";
import { useSelection } from "@/lib/selection";
import { basename } from "@/lib/utils";

/**
 * What the next question will be asked of.
 *
 * Sits in the rail rather than on the Books tab because it outlives the tab —
 * a scope chosen while browsing is still in force when the question is typed,
 * and a scope you cannot see from where you type is one you forget you set.
 */
export function Scope({ onEdit }: { onEdit: () => void }) {
  const { scope, setScope } = useSelection();

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          Scope
        </span>
        {scope.length > 0 && (
          // "whole library" read as a *label* sitting beside the heading --
          // stating the opposite of what was true, directly above a list of the
          // books actually in scope. It is a button; it has to say so.
          <Tooltip label="drop every book below and search the whole library again">
            <Button
              size="sm"
              variant="ghost"
              className="h-5 px-1.5 text-[11px]"
              onClick={() => setScope([])}
            >
              Clear
            </Button>
          </Tooltip>
        )}
      </div>

      {scope.length === 0 ? (
        <button
          onClick={onEdit}
          className="flex w-full items-center gap-2 rounded-md border border-dashed px-2.5 py-2 text-left text-xs text-muted-foreground transition-colors hover:border-ring hover:text-foreground"
        >
          <Layers3 className="size-3.5" />
          Whole library — choose books
        </button>
      ) : (
        <div className="flex flex-wrap gap-1">
          {scope.slice(0, 6).map((pattern) => (
            <Tooltip key={pattern} label={pattern}>
              <Badge className="max-w-full cursor-default gap-1 pr-1">
                <span className="truncate">{basename(pattern)}</span>
                <button
                  onClick={() => setScope(scope.filter((each) => each !== pattern))}
                  className="opacity-60 hover:opacity-100"
                >
                  <X className="size-2.5" />
                </button>
              </Badge>
            </Tooltip>
          ))}
          {scope.length > 6 && (
            <button onClick={onEdit}>
              <Badge variant="outline">+{scope.length - 6} more</Badge>
            </button>
          )}
        </div>
      )}

      {scope.length > 0 && (
        // Deliberate, measured behaviour: confining the exact-phrase leg to the
        // scope took lexical safety from 20/20 to 9/20, so it searches the whole
        // library on purpose. Said here rather than discovered later, because a
        // result from outside the scope with a correct-looking citation is read
        // as a bug the first time and as an untrustworthy scope the second.
        <p className="text-[10px] leading-snug text-muted-foreground">
          An exact phrase is still searched library-wide — those results are labelled where they
          appear.
        </p>
      )}
    </div>
  );
}
