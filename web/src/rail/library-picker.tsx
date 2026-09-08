import * as React from "react";
import {
  Check,
  ChevronsUpDown,
  CircleSlash,
  FolderPlus,
  Star,
  Trash2,
} from "lucide-react";
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
import { ModelCoverage } from "@/components/coverage";
import type { LibraryRow } from "@/lib/api";
import { useRegistryWrite } from "@/lib/queries";
import { cn } from "@/lib/utils";
import { RegisterLibrary } from "./register-library";

/**
 * Which library the question is about.
 *
 * Shows more than a name because the listing carries more: whether the index is
 * actually there (a registered library on an unmounted drive is a real and
 * ordinary state), how many books, and how far each model has embedded it. An
 * absent index is greyed and unselectable rather than chooseable and then
 * failing at the first question.
 */
export function LibraryPicker({
  libraries,
  selected,
  onSelect,
}: {
  libraries: LibraryRow[];
  selected: string | null;
  onSelect: (name: string) => void;
}) {
  const [open, setOpen] = React.useState(false);
  // Controlled, so closing clears it. A filter left behind from last time is
  // read as "no library by that name" the next time the picker opens.
  const [filter, setFilter] = React.useState("");
  const current = libraries.find((row) => row.name === selected);
  const { forget, makeDefault } = useRegistryWrite();

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) setFilter("");
      }}
    >
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="h-auto w-full justify-between px-3 py-2"
        >
          <span className="flex min-w-0 flex-col items-start gap-0.5">
            <span className="truncate text-sm font-medium">
              {current?.name ?? "Choose a library"}
            </span>
            <span className="truncate text-[11px] font-normal text-muted-foreground">
              {current
                ? current.exists
                  ? `${current.books ?? 0} books`
                  : "index not found"
                : `${libraries.length} registered`}
            </span>
          </span>
          <ChevronsUpDown className="ml-2 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>

      <PopoverContent className="w-[22rem] p-0">
        <Command>
          <CommandInput
            placeholder="Find a library…"
            value={filter}
            onValueChange={setFilter}
          />
          <CommandList>
            <CommandEmpty className="px-3 py-6 text-center text-sm text-muted-foreground">
              No library by that name.
            </CommandEmpty>

            {libraries.map((row) => (
              <CommandItem
                key={row.name}
                value={`${row.name} ${row.path}`}
                disabled={!row.exists}
                onSelect={() => {
                  onSelect(row.name);
                  setOpen(false);
                }}
                className="items-start"
              >
                <Check
                  className={cn(
                    "mt-0.5 size-3.5 shrink-0",
                    row.name === selected ? "opacity-100" : "opacity-0",
                  )}
                />
                <span className="flex min-w-0 flex-1 flex-col gap-1">
                  <span className="flex items-center gap-1.5">
                    <span className="truncate font-medium">{row.name}</span>
                    {row.default && (
                      <Tooltip label="what a bare `dyp` command means">
                        <Star className="size-3 fill-amber-400 text-amber-400" />
                      </Tooltip>
                    )}
                    {!row.exists && (
                      <Badge variant="warning">
                        <CircleSlash className="size-2.5" /> no index
                      </Badge>
                    )}
                    {row.notes && <Badge variant="outline">notes</Badge>}
                  </span>

                  <span className="truncate font-mono text-[10px] text-muted-foreground">
                    {row.path}
                  </span>

                  {row.exists ? (
                    <ModelCoverage models={row.models} />
                  ) : (
                    // Not a toast: a library whose drive is not mounted is
                    // registered and unusable, and saying which path is missing
                    // is the whole of the fix.
                    <span className="text-[11px] text-amber-500">
                      nothing readable at that path — mount it, or forget the name
                    </span>
                  )}
                </span>

                <span className="flex shrink-0 flex-col gap-1">
                  {!row.default && row.exists && (
                    <Tooltip label="make this the default for `dyp` in a terminal too">
                      <Button
                        size="icon"
                        variant="ghost"
                        className="size-6"
                        onClick={(event) => {
                          event.stopPropagation();
                          makeDefault.mutate(row.name);
                        }}
                      >
                        <Star className="size-3" />
                      </Button>
                    </Tooltip>
                  )}
                  <Tooltip label="forget the name — the index and the text are not touched">
                    <Button
                      size="icon"
                      variant="ghost"
                      className="size-6 text-muted-foreground hover:text-destructive"
                      onClick={(event) => {
                        event.stopPropagation();
                        forget.mutate(row.name, {
                          onSuccess: () => row.name === selected && onSelect(""),
                        });
                      }}
                    >
                      <Trash2 className="size-3" />
                    </Button>
                  </Tooltip>
                </span>
              </CommandItem>
            ))}
          </CommandList>
        </Command>

        <div className="border-t p-1">
          <RegisterLibrary
            onRegistered={(name) => {
              onSelect(name);
              setOpen(false);
            }}
            trigger={
              <Button variant="ghost" size="sm" className="w-full justify-start">
                <FolderPlus /> Add a library…
              </Button>
            }
          />
        </div>
      </PopoverContent>
    </Popover>
  );
}
