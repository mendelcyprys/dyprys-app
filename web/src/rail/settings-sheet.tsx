import * as React from "react";
import { FileQuestion, Settings2, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useAvailableModels } from "@/lib/queries";
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
    detail: "rewrite the query into the library’s words (~4s)",
  },
  {
    value: "rerank",
    label: "Rerank",
    detail: "a cross-encoder rescores what was found (~9s)",
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
              onChoose={(path) => update({ reranker: path })}
              files={weights.data?.models}
              searched={weights.data?.searched}
              loading={weights.isLoading}
            />
          </section>

          <section className="space-y-3">
            <div>
              <h3 className="text-sm font-medium">Expander and summariser</h3>
              <p className="mt-1 text-xs text-muted-foreground">
                Ollama model names. Left empty, the library’s own remembered choice is used — which
                is usually the right answer, and is why these are text rather than a picker.
              </p>
            </div>
            <label className="block space-y-1">
              <span className="text-xs text-muted-foreground">expander</span>
              <Input
                value={settings.expander ?? ""}
                onChange={(event) => update({ expander: event.target.value || null })}
                placeholder="whatever this library remembers"
                spellCheck={false}
                className="font-mono text-xs"
              />
            </label>
            <label className="block space-y-1">
              <span className="text-xs text-muted-foreground">summariser</span>
              <Input
                value={settings.summariser ?? ""}
                onChange={(event) => update({ summariser: event.target.value || null })}
                placeholder="whatever this library remembers"
                spellCheck={false}
                className="font-mono text-xs"
              />
            </label>
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function Weights({
  chosen,
  onChoose,
  files,
  searched,
  loading,
}: {
  chosen: string | null;
  onChoose: (path: string | null) => void;
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
        <button
          key={file.path}
          onClick={() => onChoose(file.path === chosen ? null : file.path)}
          className={cn(
            "flex w-full items-center gap-2 rounded-md border px-2.5 py-2 text-left transition-colors",
            file.path === chosen ? "border-primary bg-accent" : "hover:bg-accent/50",
          )}
        >
          <span className="min-w-0 flex-1">
            <span className="block truncate font-mono text-[11px]">{file.name}</span>
            <span className="block truncate text-[10px] text-muted-foreground">{file.path}</span>
          </span>
          <span className="shrink-0 text-[10px] tabular-nums text-muted-foreground">
            {bytes(file.bytes)}
          </span>
        </button>
      ))}
    </div>
  );
}
