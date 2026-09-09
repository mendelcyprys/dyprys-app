import * as React from "react";
import { Compass, FileQuestion, FileText, Settings2, Star, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip } from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { ModelChoice } from "@/components/model-choice";
import type { ModelRow, WeightsFile } from "@/lib/api";
import { useAvailableModels, useDefaults, useModels, useOllama, useRemember } from "@/lib/queries";
import { EFFORTS, useSearchSettings } from "@/lib/settings";
import { bytes, cn, count } from "@/lib/utils";

/**
 * The default a bare `--route` has: `routing.DEFAULT_BOOKS`.
 *
 * Spelled here rather than fetched, exactly as `cli.py` spells the rerank
 * default rather than importing it, and pinned by a test for the same reason —
 * the two drifting apart would mean the browser and a terminal quietly ran
 * different searches under the same name.
 */
const DEFAULT_ROUTE = 5;

/**
 * The choices the index cannot remember for you, in the order a search makes
 * them.
 *
 * Three sections, three questions, and each one is a separate axis: **where to
 * look** (routing narrows the books stage 2 reads), **how hard to look**
 * (expansion rewrites the query, reranking reorders what came back), **what
 * comes back** (passages, or drafted prose). Only the middle one is a choice of
 * one — expansion and reranking are measured substitutes. Routing composes with
 * either, and the model each option needs is nested under the option itself so
 * that picking a reranker and never reranking is not something this sheet lets
 * you do by accident.
 */
export function SettingsSheet({ library }: { library: string }) {
  const [open, setOpen] = React.useState(false);
  const { settings, update } = useSearchSettings();
  const weights = useAvailableModels(library, open);
  const ollama = useOllama(open);
  const defaults = useDefaults(library);
  const remember = useRemember(library);
  const models = useModels(library);

  // Which model's routing profile is the relevant one. Mirrors the picker: an
  // index with exactly one model needs no choice, so `settings.model` is null
  // there and the single row is still the one that will answer.
  const rows = models.data?.models ?? [];
  const chosen =
    rows.find((row) => row.name === settings.model) ?? (rows.length === 1 ? rows[0] : undefined);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" className="w-full justify-start">
          <Settings2 /> Search settings
          <span className="ml-auto flex gap-1">
            {settings.route > 0 && <Badge variant="outline">routed</Badge>}
            {settings.effort !== "fast" && <Badge variant="outline">{settings.effort}</Badge>}
          </span>
        </Button>
      </DialogTrigger>

      <DialogContent side="right" className="flex flex-col gap-0 overflow-hidden">
        <DialogTitle>Search settings — {library}</DialogTitle>
        <DialogDescription>
          Kept in this browser, for this library. Three independent choices: which books get read,
          how hard the search works, and what comes back.
        </DialogDescription>

        <div className="mt-6 flex-1 space-y-8 overflow-y-auto pr-2">
          <Routing
            route={settings.route}
            onRoute={(books) => update({ route: books })}
            model={chosen}
            loading={models.isLoading}
            unchosen={rows.length > 1 && !settings.model}
          />

          <section className="space-y-3">
            <div>
              <h3 className="flex items-center gap-2 text-sm font-medium">
                <Zap className="size-3.5" /> How hard to look
              </h3>
              {/* One control, not two checkboxes. Measured, expansion and
                  reranking are substitutes: together they recover the same
                  answers as the better one alone, at the sum of the costs. */}
              <p className="mt-1 text-xs text-muted-foreground">
                Expanding and reranking are substitutes, not complements — together they cost both
                and find what the better one finds alone. So this is a choice of one. Either
                composes with routing above.
              </p>
            </div>

            <div className="flex gap-1 rounded-lg border p-1">
              {EFFORTS.map((option) => (
                <button
                  key={option.value}
                  onClick={() => update({ effort: option.value })}
                  className={cn(
                    "flex-1 rounded-md px-3 py-2 text-xs transition-colors",
                    settings.effort === option.value
                      ? "bg-secondary font-medium text-secondary-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground">
              {EFFORTS.find((option) => option.value === settings.effort)?.detail}
            </p>

            {/* The model each option needs, under the option that needs it. A
                reranker chosen while Effort is Expand is a setting that does
                nothing, and the old sheet offered both at once. */}
            {settings.effort === "expand" && (
              <div className="space-y-2 rounded-md border p-3">
                <h4 className="text-xs font-medium">Expander</h4>
                <ModelChoice
                  installed={ollama.data?.models ?? []}
                  chosen={settings.expander}
                  remembered={defaults.data?.defaults.expander ?? null}
                  onChoose={(model) => update({ expander: model })}
                  onRemember={(model) => remember.mutate({ role: "expander", model })}
                  busy={remember.isPending}
                  empty={
                    <>
                      Nothing found at {ollama.data?.host ?? "the ollama server"}. Start{" "}
                      <code className="font-mono">ollama serve</code>, or name a model anyway.
                    </>
                  }
                />
                {!settings.expander && (
                  <p className="text-[11px] text-muted-foreground">
                    Nothing picked, so the library’s own default is used —{" "}
                    {defaults.data?.defaults.expander
                      ? `currently ${defaults.data.defaults.expander}.`
                      : "and there is none, so a search will refuse rather than guess."}
                  </p>
                )}
                <p className="text-[11px] leading-snug text-muted-foreground">
                  Do not expand when the answer is a rare literal — a name, a place, an odd
                  spelling. It bridges vocabulary, and a rare token has none to bridge.
                </p>
              </div>
            )}

            {settings.effort === "rerank" && (
              <div className="space-y-4 rounded-md border p-3">
                <div className="space-y-1.5">
                  <div className="flex items-baseline justify-between">
                    <span className="text-xs font-medium">Depth</span>
                    <span className="tabular-nums text-xs text-muted-foreground">
                      {count(settings.depth, "candidate")}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={3}
                    max={40}
                    step={1}
                    value={settings.depth}
                    onChange={(event) => update({ depth: Number(event.target.value) })}
                    className="w-full accent-[hsl(var(--primary))]"
                  />
                  {/* The one number that sets what reranking costs: it is a model
                      pass per candidate, so the time is linear in this. Measured
                      against a 0.6B cross-encoder, retrieval alone being 0.2s. */}
                  <p className="text-[11px] leading-snug text-muted-foreground">
                    One model pass each, so the wait is roughly linear in this — about 7.7s at 5 and
                    25.7s at 20 on a 0.6B cross-encoder. Reranking cannot find what the search
                    missed; a deeper shortlist is the only thing that can, and it is also the only
                    thing that costs.
                  </p>
                </div>

                <div className="space-y-2">
                  <h4 className="text-xs font-medium">Reranker</h4>
                  <p className="text-[11px] leading-snug text-muted-foreground">
                    A cross-encoder <code className="font-mono">.gguf</code>, not a chat model. The
                    index does not remember this one unless you star it, so it is sent with every
                    search — and asking to rerank without it is an error rather than a quiet
                    downgrade.
                  </p>
                  <Weights
                    chosen={settings.reranker}
                    remembered={defaults.data?.defaults.reranker ?? null}
                    onChoose={(path) => update({ reranker: path })}
                    onRemember={(path) => remember.mutate({ role: "reranker", model: path })}
                    busy={remember.isPending}
                    files={weights.data?.models}
                    searched={weights.data?.searched}
                    loading={weights.isLoading}
                  />
                </div>
              </div>
            )}
          </section>

          <section className="space-y-3">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="flex items-center gap-2 text-sm font-medium">
                  <FileText className="size-3.5" /> What comes back
                </h3>
                <p className="mt-1 text-xs text-muted-foreground">
                  Passages either way. Turn this on to also draft prose from them, with every
                  quotation checked against the text it cites — it costs seconds, so it is a switch
                  rather than something left on.
                </p>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={settings.summarise}
                aria-label="Draft an answer"
                onClick={() => update({ summarise: !settings.summarise })}
                className={cn(
                  "mt-1 flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors",
                  settings.summarise ? "border-primary bg-primary" : "border-input bg-muted",
                )}
              >
                <span
                  className={cn(
                    "block size-3.5 rounded-full bg-background transition-transform",
                    settings.summarise ? "translate-x-[1.15rem]" : "translate-x-[0.15rem]",
                  )}
                />
              </button>
            </div>

            {settings.summarise && (
              <div className="space-y-2 rounded-md border p-3">
                <h4 className="text-xs font-medium">Summariser</h4>
                <ModelChoice
                  installed={ollama.data?.models ?? []}
                  chosen={settings.summariser}
                  remembered={defaults.data?.defaults.summariser ?? null}
                  onChoose={(model) => update({ summariser: model })}
                  onRemember={(model) => remember.mutate({ role: "summariser", model })}
                  busy={remember.isPending}
                  empty={
                    <>
                      Nothing found at {ollama.data?.host ?? "the ollama server"}. Start{" "}
                      <code className="font-mono">ollama serve</code>, or name a model anyway.
                    </>
                  }
                />
                {!settings.summariser && (
                  <p className="text-[11px] text-muted-foreground">
                    Nothing picked, so the library’s own default is used — starred here, and{" "}
                    {defaults.data?.defaults.summariser
                      ? `currently ${defaults.data.defaults.summariser}.`
                      : "there is none, so a search will refuse rather than guess."}
                  </p>
                )}
              </div>
            )}
          </section>

          {remember.error && (
            <p className="text-xs text-destructive">{(remember.error as Error).message}</p>
          )}
          <p className="border-t pt-4 text-[11px] text-muted-foreground">
            The star writes into the index, so a default set here is the one a terminal reads too.
            Picking is just this browser, for the next search.
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/**
 * Which books stage 2 reads — the axis this sheet had no control for at all.
 *
 * `--route` is the largest speedup the tool has (~6x, reading about 1% of the
 * library) at a cost of roughly one answer in twenty-five, and it is orthogonal
 * to everything below it: the eval harness runs a routed rerank on purpose. The
 * browser never sent `route`, so every search from here read the whole library,
 * on a 3,000-book index as readily as a 20-book one.
 *
 * It is offered only when the chosen model has a profile, because a search
 * asked to route without one is a refusal and not a fallback — better said with
 * the switch than in a red box after the question.
 */
function Routing({
  route,
  onRoute,
  model,
  loading,
  unchosen,
}: {
  route: number;
  onRoute: (books: number) => void;
  model: ModelRow | undefined;
  loading: boolean;
  unchosen: boolean;
}) {
  const profiled = model?.routing.profiled_books ?? 0;
  // Only a profiled model can route. Unknown is not the same as none: with no
  // model chosen on a multi-model index there is nothing to read this off, and
  // the search would refuse on the model long before it refused on routing.
  const unavailable = Boolean(model) && profiled === 0;
  const on = route > 0;

  return (
    <section className="space-y-3">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="flex items-center gap-2 text-sm font-medium">
            <Compass className="size-3.5" /> Where to look
          </h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Routing scores every book first and searches only the best few — about six times faster,
            reading around 1% of the library, at a cost of roughly one answer in twenty-five. Its
            own choice: it composes with everything below.
          </p>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={on}
          aria-label="Route to the best books"
          disabled={unavailable || loading}
          onClick={() => onRoute(on ? 0 : DEFAULT_ROUTE)}
          className={cn(
            "mt-1 flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors",
            on ? "border-primary bg-primary" : "border-input bg-muted",
            (unavailable || loading) && "cursor-not-allowed opacity-40",
          )}
        >
          <span
            className={cn(
              "block size-3.5 rounded-full bg-background transition-transform",
              on ? "translate-x-[1.15rem]" : "translate-x-[0.15rem]",
            )}
          />
        </button>
      </div>

      {unavailable && (
        <p className="rounded-md border border-dashed p-3 text-[11px] leading-relaxed text-muted-foreground">
          This model has no routing profile, and a search asked to route without one is refused
          rather than quietly run flat. Build it under{" "}
          <strong className="font-medium">Jobs → Run route</strong>; it reads the vectors already on
          disk and embeds nothing.
        </p>
      )}

      {unchosen && !on && (
        <p className="text-[11px] text-muted-foreground">
          Choose a model in the rail to see whether it can route — the profile belongs to the model,
          not the library.
        </p>
      )}

      {on && (
        <div className="space-y-1.5 rounded-md border p-3">
          <div className="flex items-baseline justify-between">
            <span className="text-xs font-medium">Books read</span>
            <span className="tabular-nums text-xs text-muted-foreground">
              {/* "5 of 0" is what this said before a model was chosen, which
                  reads as broken rather than as unknown. The profile belongs to
                  a model, so with none picked there is no denominator to give. */}
              {model ? `${route} of ${profiled.toLocaleString()}` : count(route, "book")}
            </span>
          </div>
          <input
            type="range"
            min={1}
            max={50}
            step={1}
            value={route}
            onChange={(event) => onRoute(Number(event.target.value))}
            className="w-full accent-[hsl(var(--primary))]"
          />
          <p className="text-[11px] leading-snug text-muted-foreground">
            Stage 2 costs what this says, not what the library holds — which is the whole point on a
            large one. Narrower is faster and misses more: when a result is the wrong book but a
            plausible passage, widen this or turn routing off and ask again.
          </p>
          {(model?.routing.stale_books ?? 0) > 0 && (
            // The one failure invisible from the output: an unprofiled book
            // cannot be returned at *any* rank, and the search still comes back
            // with a full k at 200. It also arrives as a warning on the result,
            // but by then the search has already been run without it.
            <p className="text-[11px] leading-snug text-amber-500">
              {count(model!.routing.stale_books, "book")} embedded or changed since the profile was
              built. Routing cannot return {model!.routing.stale_books === 1 ? "it" : "them"} at any
              rank, and the search will not look short — run route again under Jobs.
            </p>
          )}
        </div>
      )}
    </section>
  );
}

function Weights({
  chosen,
  remembered,
  onChoose,
  onRemember,
  busy,
  files,
  searched,
  loading,
}: {
  chosen: string | null;
  remembered: string | null;
  onChoose: (path: string | null) => void;
  onRemember: (path: string | null) => void;
  busy: boolean;
  files?: WeightsFile[];
  searched?: string[];
  loading: boolean;
}) {
  if (loading) return <p className="text-xs text-muted-foreground">Looking…</p>;

  // Only the files that can actually do this job, plus the ones that did not
  // say. `rerank === false` is the file's own `pooling_type` declaring it an
  // embedding model: llama.cpp will load it as a reranker without complaint and
  // return numbers that are not relevance, so offering it would be offering a
  // silently wrong ranking.
  const usable = (files ?? []).filter((file) => file.rerank !== false);
  const hidden = (files ?? []).length - usable.length;

  if (!files?.length) {
    // An empty list is only readable next to where it looked — otherwise it
    // says "you have no models", which is almost never what happened.
    return (
      <div className="space-y-2 rounded-md border border-dashed p-3">
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <FileQuestion className="size-3.5" /> No <code className="font-mono">.gguf</code> found.
        </p>
        <p className="text-[11px] text-muted-foreground">
          Looked in {searched?.length ? searched.join(", ") : "nowhere"}. Set{" "}
          <code className="font-mono">DYPRYS_MODEL_DIR</code> before starting the server to look
          elsewhere.
        </p>
      </div>
    );
  }

  if (!usable.length) {
    return (
      <div className="space-y-2 rounded-md border border-dashed p-3">
        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <FileQuestion className="size-3.5" /> None of the {files.length}{" "}
          <code className="font-mono">.gguf</code> files found is a cross-encoder.
        </p>
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Each one declares itself an embedding model. Reranking needs a cross-encoder — one that
          scores a (query, passage) pair rather than embedding a single text. bge-reranker,
          jina-reranker and Qwen3-Reranker are the usual ones.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {hidden > 0 && (
        <p className="pb-1 text-[10px] text-muted-foreground">
          {hidden} other <code className="font-mono">.gguf</code>{" "}
          {hidden === 1 ? "file declares" : "files declare"} themselves embedding models, and are
          not shown — used as a reranker they score plausibly and rank wrongly.
        </p>
      )}
      {usable.map((file) => (
        <div
          key={file.path}
          className={cn(
            "flex items-center gap-2 rounded-md border px-2.5 py-2 transition-colors",
            file.path === chosen ? "border-primary bg-accent" : "hover:bg-accent/50",
          )}
        >
          <button
            onClick={() => onChoose(file.path === chosen ? null : file.path)}
            className="flex min-w-0 flex-1 items-center gap-2 text-left"
          >
            <span className="min-w-0 flex-1">
              <span className="block truncate font-mono text-[11px]">{file.name}</span>
              <span className="block truncate text-[10px] text-muted-foreground">{file.path}</span>
            </span>
            {file.path === remembered && (
              <Badge variant="outline" className="shrink-0">
                library default
              </Badge>
            )}
            {file.rerank === null && (
              // Shown, not hidden: the file said nothing about itself, and
              // "we could not tell" is a different answer from "yes".
              <Badge variant="outline" className="shrink-0">
                unverified
              </Badge>
            )}
            <span className="shrink-0 text-[10px] tabular-nums text-muted-foreground">
              {bytes(file.bytes)}
            </span>
          </button>
          <Tooltip
            label={
              file.path === remembered
                ? "forget it — this library will have no default reranker"
                : "remember it in the index, so `--rerank` needs no path here or in a terminal"
            }
          >
            <Button
              size="icon"
              variant="ghost"
              className="size-7 shrink-0"
              disabled={busy}
              onClick={() => onRemember(file.path === remembered ? null : file.path)}
            >
              <Star
                className={cn(
                  "size-3",
                  file.path === remembered && "fill-amber-400 text-amber-400",
                )}
              />
            </Button>
          </Tooltip>
        </div>
      ))}
    </div>
  );
}
