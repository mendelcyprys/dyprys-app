import { AlertTriangle, PlugZap } from "lucide-react";
import { RegisterLibrary } from "@/rail/register-library";

/**
 * The three states where there is nothing to show, each said differently.
 *
 * They look alike in a UI and are not alike at all: one is a server that is not
 * running, one is a corrupt registry where *every* name has stopped working,
 * and one is a fresh installation. Telling someone "no libraries registered"
 * when they registered five is the message that turns a recoverable file into a
 * lost one.
 */

function Screen({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full items-center justify-center p-10">
      <div className="max-w-lg space-y-4">{children}</div>
    </div>
  );
}

export function ServerUnreachable() {
  return (
    <Screen>
      <h1 className="flex items-center gap-2 text-lg font-semibold">
        <PlugZap className="size-5 text-amber-500" /> The server is not answering
      </h1>
      <p className="text-sm text-muted-foreground">
        This page talks to a local <code className="font-mono">dyp serve</code>. Start one and it
        will connect on its own.
      </p>
      <pre className="rounded-md border bg-muted/40 px-3 py-2 font-mono text-xs">
        dyp serve --port 8765
      </pre>
    </Screen>
  );
}

export function RegistryUnreadable({ path }: { path: string }) {
  return (
    <Screen>
      <h1 className="flex items-center gap-2 text-lg font-semibold">
        <AlertTriangle className="size-5 text-destructive" /> The registry cannot be read
      </h1>
      <p className="text-sm text-muted-foreground">
        A registry file exists at the path below and could not be parsed, so the names of every
        library are unavailable. This is not the same as having none registered, and nothing here
        will overwrite it — a write sets it aside as{" "}
        <code className="font-mono">*.unreadable</code> first, so what is in it stays recoverable
        by hand.
      </p>
      <pre className="overflow-x-auto rounded-md border bg-muted/40 px-3 py-2 font-mono text-xs">
        {path}
      </pre>
      <p className="text-sm text-muted-foreground">
        Every library still opens by path in a terminal: <code className="font-mono">dyp --data DIR status</code>.
      </p>
    </Screen>
  );
}

export function NothingRegistered() {
  return (
    <Screen>
      <h1 className="text-lg font-semibold">No libraries yet</h1>
      <p className="text-sm text-muted-foreground">
        A library is a directory holding an index over your own texts. Name one you already have,
        or build a new one from a folder of <code className="font-mono">.txt</code> files in a
        terminal.
      </p>
      <div className="flex items-center gap-3">
        <RegisterLibrary />
      </div>
      <pre className="overflow-x-auto rounded-md border bg-muted/40 px-3 py-2 font-mono text-xs leading-relaxed">
        dyp library add neuro ~/dyprys/neuro{"\n"}
        dyp -L neuro add ~/books
      </pre>
      <p className="text-xs text-muted-foreground">
        Embedding is the slow step and is never started for you — it can run for days on a large
        library.
      </p>
    </Screen>
  );
}
