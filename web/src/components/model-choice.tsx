import * as React from "react";
import { Check, Star } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

/**
 * Pick an ollama model, and optionally make it what this library remembers.
 *
 * Two different scopes, side by side and said as such. The choice is this
 * browser's, for the next search; the star writes into the *index*, which is
 * the same setting `dyp models --summariser` writes and the same one a bare
 * `--summarise` reads. Conflating them would mean a default set in a tab that
 * a terminal never sees, which is the opposite of what a default is for.
 */
export function ModelChoice({
  installed,
  chosen,
  remembered,
  onChoose,
  onRemember,
  busy,
  empty,
}: {
  installed: string[];
  chosen: string | null;
  remembered: string | null;
  onChoose: (model: string | null) => void;
  onRemember: (model: string | null) => void;
  busy: boolean;
  empty: React.ReactNode;
}) {
  const [typed, setTyped] = React.useState("");
  // Everything installed, plus anything already named that is not — a model
  // pulled on another machine, or ollama simply not running just now.
  const options = [...new Set([...installed, chosen, remembered].filter(Boolean) as string[])];

  if (options.length === 0) {
    return (
      <div className="space-y-2">
        <div className="rounded-md border border-dashed p-3 text-xs text-muted-foreground">
          {empty}
        </div>
        <Typed value={typed} onChange={setTyped} onSubmit={() => onChoose(typed.trim() || null)} />
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {options.map((model) => (
        <div
          key={model}
          className={cn(
            "flex items-center gap-2 rounded-md border px-2.5 py-2 transition-colors",
            model === chosen ? "border-primary bg-accent" : "hover:bg-accent/50",
          )}
        >
          <button
            onClick={() => onChoose(model === chosen ? null : model)}
            className="flex min-w-0 flex-1 items-center gap-2 text-left"
          >
            <Check className={cn("size-3.5 shrink-0", model === chosen ? "" : "opacity-0")} />
            <span className="truncate font-mono text-[11px]">{model}</span>
            {model === remembered && (
              <Badge variant="outline" className="shrink-0">
                library default
              </Badge>
            )}
            {!installed.includes(model) && (
              <Badge variant="warning" className="shrink-0">
                not installed
              </Badge>
            )}
          </button>

          <Tooltip
            label={
              model === remembered
                ? "forget it — this library will have no default"
                : "remember it for this library, in the index, so `dyp ask --summarise` in a terminal uses it too"
            }
          >
            <Button
              size="icon"
              variant="ghost"
              className="size-7 shrink-0"
              disabled={busy}
              onClick={() => onRemember(model === remembered ? null : model)}
            >
              <Star
                className={cn("size-3", model === remembered && "fill-amber-400 text-amber-400")}
              />
            </Button>
          </Tooltip>
        </div>
      ))}
    </div>
  );
}

function Typed({
  value,
  onChange,
  onSubmit,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
}) {
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
      className="flex gap-2"
    >
      <Input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="or type a model name"
        spellCheck={false}
        className="font-mono text-xs"
      />
      <Button type="submit" size="sm" variant="outline" disabled={!value.trim()}>
        Use
      </Button>
    </form>
  );
}
