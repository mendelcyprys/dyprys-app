import * as React from "react";
import { FileQuestion, Settings2, Star, Zap } from "lucide-react";
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
import { useAvailableModels, useDefaults, useOllama, useRemember } from "@/lib/queries";
import { useSearchSettings, type Effort } from "@/lib/settings";
import { bytes, cn } from "@/lib/utils";

const EFFORT: { value: Effort; label: string; detail: string }[] = [
  {
    value: "fast",
    label: "Fast",
    detail: "the default: literal-safe, no extra model",
  },
  {
    value: "expand",
    label: "Expand",
    detail: "rewrite the query into the library’s words, then search",
  },
  {
    value: "rerank",
    label: "Rerank",
    // No figure: the cost is one pass of a cross-encoder per candidate, so it
    // is set by the depth, the model and the machine rather than by the
    // feature. Measured here at 25.7s for 20 passages against a 0.6B reranker;
    // quoting a number the page cannot know is worse than quoting none.
    detail: "a cross-encoder rescores every candidate — seconds per passage",
  },
];

/**
 * The choices the index cannot remember for you.
 *
 * A reranker is a path to a cross-encoder GGUF, must be passed on every search,
 * and bare `rerank` errors rather than silently returning unreranked results.
 * So it is chosen here, from what is actually on the machine, and kept per
 * library in this browser.
 */
export function SettingsSheet({ library }: { library: string }) {
  const [open, setOpen] = React.useState(false);
  const { settings, update } = useSearchSettings();
  const weights = useAvailableModels(library, open);
  const ollama = useOllama(open);
  const defaults = useDefaults(library);
  const remember = useRemember(library);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" className="w-full justify-start">
          <Settings2 /> Search settings
          {settings.effort !== "fast" && (
            <Badge variant="outline" className="ml-auto">
              {settings.effort}
            </Badge>
          )}
        </Button>
      </DialogTrigger>

      <DialogContent side="right" className="flex flex-col gap-0 overflow-hidden">
        <DialogTitle>Search settings — {library}</DialogTitle>
        <DialogDescription>
          Kept in this browser, for this library. Everything else a search needs, the index already
          remembers.
        </DialogDescription>

        <div className="mt-6 flex-1 space-y-8 overflow-y-auto pr-2">
          <section className="space-y-3">
            <div>
              <h3 className="flex items-center gap-2 text-sm font-medium">
                <Zap className="size-3.5" /> Effort
              </h3>
              {/* One control, not two checkboxes. Measured, expansion and
                  reranking are substitutes: together they recover the same
                  answers as the better one alone, at the sum of the costs. */}
              <p className="mt-1 text-xs text-muted-foreground">
                Expanding and reranking are substitutes, not complements — together they cost both
                and find what the better one finds alone. So this is a choice of one.
              </p>
            </div>

            <div className="flex gap-1 rounded-lg border p-1">
              {EFFORT.map((option) => (
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
              {EFFORT.find((option) => option.value === settings.effort)?.detail}
            </p>

            {settings.effort === "rerank" && (
              <div className="space-y-1.5 rounded-md border p-3">
                <div className="flex items-baseline justify-between">
                  <span className="text-xs font-medium">Depth</span>
                  <span className="tabular-nums text-xs text-muted-foreground">
                    {settings.depth} candidates
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
                  25.7s at 20 on a 0.6B cross-encoder. Reranking cannot find what the search missed;
                  a deeper shortlist is the only thing that can, and it is also the only thing that
                  costs.
                </p>
              </div>
            )}
          </section>

          <section className="space-y-3">
            <div>
              <h3 className="text-sm font-medium">Reranker</h3>
              <p className="mt-1 text-xs text-muted-foreground">
                A cross-encoder <code className="font-mono">.gguf</code>, not a chat model. The
                index does not remember this one, so it is sent with every search — and asking to
                rerank without it is an error rather than a quiet downgrade.
              </p>
            </div>

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
          </section>

          <section className="space-y-3">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-sm font-medium">Draft an answer</h3>
                <p className="mt-1 text-xs text-muted-foreground">
                  Prose instead of passages, with every quotation checked against the text it cites.
                  It costs seconds, and the passages are the result either way — so it is a switch
                  rather than something left on.
                </p>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={settings.summarise}
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
            )}
            {settings.summarise && !settings.summariser && (
              <p className="text-[11px] text-muted-foreground">
                Nothing picked, so the library’s own default is used — starred here, and{" "}
                {defaults.data?.defaults.summariser
                  ? `currently ${defaults.data.defaults.summariser}.`
                  : "there is none, so a search will refuse rather than guess."}
              </p>
            )}
          </section>

          <section className="space-y-3">
            <div>
              <h3 className="text-sm font-medium">Expander</h3>
              <p className="mt-1 text-xs text-muted-foreground">
                Used only when Effort is set to Expand. Pick nothing and the library’s own default
                is used.
              </p>
            </div>
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
            {remember.error && (
              <p className="text-xs text-destructive">{(remember.error as Error).message}</p>
            )}
            <p className="text-[11px] text-muted-foreground">
              The star writes into the index, so a default set here is the one a terminal reads too.
              Picking is just this browser, for the next search.
            </p>
          </section>
        </div>
      </DialogContent>
    </Dialog>
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
  files?: { path: string; name: string; bytes: number }[];
  searched?: string[];
  loading: boolean;
}) {
  if (loading) return <p className="text-xs text-muted-foreground">Looking…</p>;

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

  return (
    <div className="space-y-1">
      {files.map((file) => (
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
