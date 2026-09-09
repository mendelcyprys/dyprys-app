import * as React from "react";
import { Check, ChevronsUpDown, CircleSlash, EyeOff, FolderPlus, Star, Trash2 } from "lucide-react";
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
import { cn, count } from "@/lib/utils";
import { DeleteIndex } from "./delete-index";
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
  // Which library's index a confirmation is open for. Held by name rather than
  // by row, because the row is refetched while the dialog is up.
  const [deleting, setDeleting] = React.useState<string | null>(null);
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
                  ? count(current.books ?? 0, "book")
                  : "index not found"
                : `${libraries.length} registered`}
            </span>
          </span>
          <ChevronsUpDown className="ml-2 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>

      <PopoverContent className="w-[22rem] p-0">
        <Command>
          <CommandInput placeholder="Find a library…" value={filter} onValueChange={setFilter} />
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
                        aria-label={`make ${row.name} the default library`}
                        onClick={(event) => {
                          event.stopPropagation();
                          makeDefault.mutate(row.name);
                        }}
                      >
                        <Star className="size-3" />
                      </Button>
                    </Tooltip>
                  )}
                  {/* Two removals, and the difference between them is the
                      whole reason they are two buttons. Forgetting a name
                      touches nothing on disk and is undone by adding the same
                      path again; deleting the index destroys the vectors.

                      Each carries an `aria-label` as well as a tooltip: a
                      tooltip is a hover affordance and not an accessible name,
                      so without one these read as three unnamed buttons — next
                      to each other, and one of them irreversible. */}
                  <Tooltip label="forget the name — nothing on disk is touched, and adding the same path again brings it all back">
                    <Button
                      size="icon"
                      variant="ghost"
                      className="size-6 text-muted-foreground hover:text-foreground"
                      aria-label={`forget the name ${row.name}, keeping its files`}
                      onClick={(event) => {
                        event.stopPropagation();
                        forget.mutate(row.name, {
                          onSuccess: () => row.name === selected && onSelect(""),
                        });
                      }}
                    >
                      <EyeOff className="size-3" />
                    </Button>
                  </Tooltip>
                  {row.exists && (
                    <Tooltip label="delete the index itself — the vectors go, the text stays. Previewed and confirmed first.">
                      <Button
                        size="icon"
                        variant="ghost"
                        className="size-6 text-muted-foreground hover:text-destructive"
                        aria-label={`delete the ${row.name} index`}
                        onClick={(event) => {
                          event.stopPropagation();
                          setDeleting(row.name);
                        }}
                      >
                        <Trash2 className="size-3" />
                      </Button>
                    </Tooltip>
                  )}
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

      {deleting && (
        <DeleteIndex
          name={deleting}
          open
          onOpenChange={(next) => !next && setDeleting(null)}
          onDeleted={() => {
            if (deleting === selected) onSelect("");
            setOpen(false);
          }}
        />
      )}
    </Popover>
  );
}
