import * as React from "react";
import { FolderPlus, Terminal } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { DyprysError } from "@/lib/api";
import { useRegistryWrite } from "@/lib/queries";

/**
 * Name a directory on the machine the server is running on.
 *
 * A browser cannot pick a directory, so this is a text field for a server-side
 * path — honest about what it is rather than pretending to be a file dialog.
 * The server refuses a path that is not there, which is what makes typing one
 * survivable: a typo is a refusal, not a registry entry that fails forever.
 */
export function RegisterLibrary({
  trigger,
  onRegistered,
}: {
  trigger?: React.ReactNode;
  onRegistered?: (name: string) => void;
}) {
  const [open, setOpen] = React.useState(false);
  const [name, setName] = React.useState("");
  const [path, setPath] = React.useState("");
  const { register } = useRegistryWrite();

  const failure = register.error instanceof DyprysError ? register.error : null;

  function submit(event: React.FormEvent) {
    event.preventDefault();
    register.mutate(
      { name: name.trim(), path: path.trim() },
      {
        onSuccess: () => {
          onRegistered?.(name.trim());
          setOpen(false);
          setName("");
          setPath("");
        },
      },
    );
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        register.reset();
      }}
    >
      <DialogTrigger asChild>
        {trigger ?? (
          <Button size="sm" variant="outline">
            <FolderPlus /> Add a library
          </Button>
        )}
      </DialogTrigger>
      <DialogContent>
        <DialogTitle>Name a library</DialogTitle>
        <DialogDescription>
          A library is a directory holding an index. This records the name only — nothing is
          created, moved or embedded, so a wrong path costs one click to undo.
        </DialogDescription>

        <form onSubmit={submit} className="mt-4 space-y-4">
          <label className="block space-y-1.5">
            <span className="text-xs font-medium text-muted-foreground">Name</span>
            <Input
              autoFocus
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="a short name for it"
              spellCheck={false}
            />
            <span className="block text-[11px] text-muted-foreground">
              No spaces or slashes — the same name `dyp -L` takes.
            </span>
          </label>

          <label className="block space-y-1.5">
            <span className="text-xs font-medium text-muted-foreground">
              Directory, on the machine running the server
            </span>
            <Input
              value={path}
              onChange={(event) => setPath(event.target.value)}
              placeholder="/Users/you/dyprys/mine"
              spellCheck={false}
              className="font-mono text-xs"
            />
          </label>

          {failure && (
            <p className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {failure.detail}
              {failure.choices.length > 0 && (
                <span className="mt-1 block opacity-80">
                  already taken: {failure.choices.join(", ")}
                </span>
              )}
            </p>
          )}

          <div className="flex items-center justify-between gap-3 pt-1">
            <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
              <Terminal className="size-3" />
              same as <code className="font-mono">dyp library add</code>
            </span>
            <Button
              type="submit"
              size="sm"
              disabled={!name.trim() || !path.trim() || register.isPending}
            >
              {register.isPending ? "Registering…" : "Register"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
