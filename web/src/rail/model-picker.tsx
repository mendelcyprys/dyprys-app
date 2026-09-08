import * as React from "react";
import { AlertTriangle, Check, ChevronsUpDown, Circle, Flame } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Tooltip } from "@/components/ui/tooltip";
import { CoverageBar, shortModel } from "@/components/coverage";
import type { ModelRow } from "@/lib/api";
import { useHealth, useModels, useWarm } from "@/lib/queries";
import { bytes, cn } from "@/lib/utils";

/** Alias if there is one, else the readable middle of the weights handle. */
export function modelLabel(model: ModelRow): string {
  return model.alias ?? shortModel(model.name);
}

/**
 * Which vectors answer the question.
 *
 * Required, not preferred, once an index holds more than one model: search
 * refuses rather than guess, because two models are two different searches over
 * the same books. The refusal carries its `choices`, which is what makes this
 * picker the fix rather than an error message — see `ModelChoices` below, which
 * renders that error directly.
 */
export function ModelPicker({
  library,
  selected,
  onSelect,
}: {
  library: string;
  selected: string | null;
  onSelect: (name: string | null) => void;
}) {
  const [open, setOpen] = React.useState(false);
  const models = useModels(library);
  const health = useHealth();
  const warm = useWarm(library);
  const rows = models.data?.models ?? [];
  const current = rows.find((row) => row.name === selected);
  const resident = new Set(health.data?.loaded[library] ?? []);
  const only = rows.length === 1 ? rows[0] : undefined;

  // One model is not a choice, and asking someone to make it is noise. It is
  // still worth naming, because the same rail on the next library will hold a
  // decision that changes the answer.
  const effective = current ?? only;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          Model
        </span>
        {effective && resident.has(effective.name) && (
          <Tooltip label="loaded — the first question will not pay to read it from disk">
            <span className="flex items-center gap-1 text-[10px] text-emerald-500">
              <Flame className="size-2.5" /> warm
            </span>
          </Tooltip>
        )}
      </div>

      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            className={cn(
              "h-auto w-full justify-between px-2.5 py-1.5",
              !effective && rows.length > 1 && "border-amber-500/50",
            )}
          >
            <span className="flex min-w-0 flex-col items-start">
              <span className="truncate text-xs font-medium">
                {effective ? modelLabel(effective) : rows.length > 1 ? "Choose a model" : "—"}
              </span>
              {effective && (
                <span className="text-[10px] font-normal text-muted-foreground">
                  {effective.dim}d · {effective.store}
                </span>
              )}
            </span>
            <ChevronsUpDown className="ml-2 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>

        <PopoverContent className="w-[20rem] p-0">
          <Command>
            {rows.length > 4 && <CommandInput placeholder="Find a model…" />}
            <CommandList>
              <CommandEmpty className="px-3 py-6 text-center text-sm text-muted-foreground">
                Nothing has embedded this library yet.
              </CommandEmpty>
              {rows.map((row) => (
                <CommandItem
                  key={row.name}
                  value={row.name}
                  onSelect={() => {
                    onSelect(row.name);
                    setOpen(false);
                    if (row.file_present) warm.mutate(row.name);
                  }}
                  className="items-start"
                >
                  <Check
                    className={cn(
                      "mt-0.5 size-3.5 shrink-0",
                      row.name === effective?.name ? "opacity-100" : "opacity-0",
                    )}
                  />
                  <span className="flex min-w-0 flex-1 flex-col gap-1">
                    <span className="flex items-center gap-1.5">
                      <span className="truncate font-medium">{modelLabel(row)}</span>
                      {resident.has(row.name) && (
                        <Circle className="size-2 fill-emerald-500 text-emerald-500" />
                      )}
                      {!row.file_present && (
                        <Badge variant="warning">
                          <AlertTriangle className="size-2.5" /> weights gone
                        </Badge>
                      )}
                    </span>
                    <CoverageBar fraction={row.coverage} />
                    <span className="truncate text-[10px] text-muted-foreground">
                      {row.dim}d · {row.store} · {bytes(row.disk_bytes)} of vectors
                    </span>
                    {!row.file_present && (
                      // The index remembers which weights made its vectors and
                      // they are not here now: coverage is readable, searching
                      // is a 503 at query time. Better said before the question.
                      <span className="text-[10px] leading-snug text-amber-500">
                        the file this index was built with is missing — you can read its coverage,
                        not search with it
                      </span>
                    )}
                  </span>
                </CommandItem>
              ))}
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}

/**
 * The `model_ambiguous` refusal, rendered as the picker that resolves it.
 *
 * `choices` is on the error shape for exactly this, and this is the first-run
 * experience on any index with two models — so it is wired deliberately rather
 * than left to a generic error card.
 */
export function ModelChoices({
  choices,
  onSelect,
}: {
  choices: string[];
  onSelect: (name: string) => void;
}) {
  return (
    <div className="space-y-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3">
      <p className="text-xs">
        This library has been embedded by more than one model. Searching refuses rather than guess
        which vectors to answer from — they are two different searches over the same books.
      </p>
      <div className="flex flex-wrap gap-2">
        {choices.map((name) => (
          <Button key={name} size="sm" variant="outline" onClick={() => onSelect(name)}>
            {name}
          </Button>
        ))}
      </div>
    </div>
  );
}
