import * as React from "react";
import { AlertTriangle, Check, Pencil, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tooltip } from "@/components/ui/tooltip";
import { CoverageBar, shortModel } from "@/components/coverage";
import type { ModelRow } from "@/lib/api";
import { useAlias, useHealth, useModels, useWarm } from "@/lib/queries";
import { bytes, cn } from "@/lib/utils";

/**
 * What has embedded this library, how far, and whether it can still be used.
 *
 * The last of those is the reason this is a tab rather than a line in the rail:
 * an index remembers which weights made its vectors, and a model whose file has
 * since gone reads perfectly here and fails at query time with a 503.
 */
export function Models({ library }: { library: string }) {
  const models = useModels(library);
  const health = useHealth();
  const warm = useWarm(library);
  const rows = models.data?.models ?? [];
  const resident = new Set(health.data?.loaded[library] ?? []);

  if (models.isLoading) {
    return <p className="text-sm text-muted-foreground">Reading…</p>;
  }
  if (rows.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-6">
        <p className="text-sm">Nothing has embedded this library yet.</p>
        <p className="mt-2 max-w-prose text-xs leading-relaxed text-muted-foreground">
          Embedding is the one slow, expensive step — it can run for days on a large library — so it
          is never started for you. In a terminal:{" "}
          <code className="font-mono">dyp -L {library} embed --model PATH.gguf</code>.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3 overflow-y-auto">
      {rows.map((row) => (
        <Model
          key={row.name}
          library={library}
          model={row}
          warm={resident.has(row.name)}
          onWarm={() => warm.mutate(row.name)}
          warming={warm.isPending}
        />
      ))}
    </div>
  );
}

function Model({
  library,
  model,
  warm,
  onWarm,
  warming,
}: {
  library: string;
  model: ModelRow;
  warm: boolean;
  onWarm: () => void;
  warming: boolean;
}) {
  const [naming, setNaming] = React.useState(false);
  const [draft, setDraft] = React.useState(model.alias ?? "");
  const alias = useAlias(library);

  return (
    <section
      className={cn(
        "rounded-lg border p-4",
        !model.file_present && "border-amber-500/40 bg-amber-500/5",
      )}
    >
      <header className="flex flex-wrap items-start gap-x-3 gap-y-1">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h3 className="truncate text-sm font-medium">
              {model.alias ?? shortModel(model.name)}
            </h3>
            {warm && <Badge variant="ok">warm</Badge>}
            {!model.file_present && (
              <Badge variant="warning">
                <AlertTriangle className="size-2.5" /> weights missing
              </Badge>
            )}
          </div>
          {/* The full handle: it identifies the weights exactly, which is what
              `--model` resolves against and what a backup has to match. */}
          <p className="mt-0.5 break-all font-mono text-[10px] text-muted-foreground">
            {model.name}
          </p>
        </div>

        <div className="flex items-center gap-1">
          {naming ? (
            <form
              onSubmit={(event) => {
                event.preventDefault();
                alias.mutate(
                  { model: model.name, alias: draft.trim() },
                  { onSuccess: () => setNaming(false) },
                );
              }}
              className="flex items-center gap-1"
            >
              <Input
                autoFocus
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="short name"
                className="h-7 w-28 text-xs"
                spellCheck={false}
              />
              <Button type="submit" size="icon" className="size-7" disabled={!draft.trim()}>
                <Check className="size-3" />
              </Button>
              <Button
                type="button"
                size="icon"
                variant="ghost"
                className="size-7"
                onClick={() => setNaming(false)}
              >
                <X className="size-3" />
              </Button>
            </form>
          ) : (
            <Tooltip label="a short name to type instead of the handle — the terminal learns it too">
              <Button size="sm" variant="ghost" onClick={() => setNaming(true)}>
                <Pencil /> {model.alias ? "Rename" : "Name it"}
              </Button>
            </Tooltip>
          )}

          {model.file_present && !warm && (
            <Button size="sm" variant="outline" onClick={onWarm} disabled={warming}>
              {warming ? "Loading…" : "Warm"}
            </Button>
          )}
        </div>
      </header>

      {alias.error && (
        <p className="mt-2 text-xs text-destructive">{(alias.error as Error).message}</p>
      )}

      <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 text-xs sm:grid-cols-4">
        <Fact label="coverage">
          <CoverageBar fraction={model.coverage} />
          <span className="mt-0.5 block text-[10px] text-muted-foreground">
            {model.embedded.toLocaleString()} / {model.live_chunks.toLocaleString()} chunks
          </span>
        </Fact>
        <Fact label="vectors">
          {model.dim}d · {model.store}
          <span className="mt-0.5 block text-[10px] text-muted-foreground">
            {bytes(model.disk_bytes)} on disk
          </span>
        </Fact>
        <Fact label="routing">
          {model.routing.profiled_books.toLocaleString()} profiled
          {model.routing.stale_books > 0 && (
            // Routing cannot reach an unprofiled book at *any* rank, and the
            // search still returns a full k at 200 — so this number is the one
            // thing that says part of the library is invisible under --route.
            <span className="mt-0.5 block text-[10px] text-amber-500">
              {model.routing.stale_books.toLocaleString()} stale — run `dyp route`
            </span>
          )}
        </Fact>
        <Fact label="failures">
          {model.failures.toLocaleString()}
          {model.carries > 0 && (
            <span className="mt-0.5 block text-[10px] text-muted-foreground">
              {model.carries.toLocaleString()} carried
            </span>
          )}
        </Fact>
      </dl>

      {model.file_path && (
        <p className="mt-3 break-all font-mono text-[10px] text-muted-foreground">
          {model.file_present ? "weights: " : "expected at: "}
          {model.file_path}
        </p>
      )}
    </section>
  );
}

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className="mt-1">{children}</dd>
    </div>
  );
}
