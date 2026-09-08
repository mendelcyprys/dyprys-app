import * as React from "react";
import { CornerDownLeft, Loader2, Search, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { askStream, DyprysError, type Answered, type SearchBody, type Stage } from "@/lib/api";
import { useSelection } from "@/lib/selection";
import { useSearchSettings } from "@/lib/settings";
import { ModelChoices } from "@/rail/model-picker";
import { Reader, type Reading } from "./reader";
import { Results } from "./results";

/** What `--rerank N` and `--route N` mean when the UI asks for them. */
const RERANK_DEPTH = 20;

/**
 * The search, streamed.
 *
 * `POST /ask/stream` from the start rather than retrofitted: a ten-second
 * routed search with no output is indistinguishable from a hang, and the stage
 * frames — the rephrasings an expander chose, the books routing picked,
 * "rescoring 20 passages", the summariser's retry — already exist to say what
 * is happening. They run as a live line, then collapse into a disclosure.
 */
export function Ask({ library }: { library: string }) {
  const [question, setQuestion] = React.useState("");
  const [stages, setStages] = React.useState<Stage[]>([]);
  const [answered, setAnswered] = React.useState<Answered | null>(null);
  const [failure, setFailure] = React.useState<DyprysError | null>(null);
  const [running, setRunning] = React.useState(false);
  const [reading, setReading] = React.useState<Reading | null>(null);
  const [showStages, setShowStages] = React.useState(false);
  const abort = React.useRef<AbortController | null>(null);

  const { scope } = useSelection();
  const { settings, update } = useSearchSettings();

  async function run(event: React.FormEvent) {
    event.preventDefault();
    if (!question.trim() || running) return;

    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;

    setRunning(true);
    setStages([]);
    setFailure(null);
    setShowStages(false);

    const body: SearchBody = {
      question,
      k: 5,
      model: settings.model,
      collection: scope.length ? scope : null,
      // Never both. Measured, they are substitutes: together they recover the
      // same answers as the better one alone, at the sum of the costs.
      rerank: settings.effort === "rerank" ? RERANK_DEPTH : 0,
      reranker: settings.effort === "rerank" ? settings.reranker : null,
      expand: settings.effort === "expand" ? (settings.expander ?? true) : null,
      summarise: settings.summariser ?? null,
    };

    try {
      setAnswered(
        await askStream(
          library,
          body,
          (stage) => setStages((all) => [...all, stage]),
          controller.signal,
        ),
      );
    } catch (thrown) {
      if ((thrown as Error).name === "AbortError") return;
      setFailure(
        thrown instanceof DyprysError ? thrown : new DyprysError("error", String(thrown), 0),
      );
      setAnswered(null);
    } finally {
      if (abort.current === controller) setRunning(false);
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      <form onSubmit={run} className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="What are you looking for? Remembered words go in verbatim."
            className="h-10 pl-8"
            autoFocus
          />
        </div>
        <Button type="submit" className="h-10" disabled={!question.trim() || running}>
          {running ? <Loader2 className="animate-spin" /> : <CornerDownLeft />}
          {running ? "Searching" : "Ask"}
        </Button>
        {running && (
          <Button
            type="button"
            variant="ghost"
            className="h-10"
            onClick={() => {
              abort.current?.abort();
              setRunning(false);
            }}
          >
            <X /> Stop
          </Button>
        )}
      </form>

      <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
        <Badge variant="outline">{settings.effort}</Badge>
        {scope.length > 0 && <Badge variant="outline">{scope.length} books</Badge>}
        {settings.summariser && <Badge variant="outline">summarised</Badge>}
        {settings.effort === "rerank" && !settings.reranker && (
          // Bare rerank is a 503, not a downgrade — better said before the
          // search than after it.
          <span className="text-amber-500">
            no reranker chosen — pick one in Search settings, or this will refuse
          </span>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
        {running && stages.length > 0 && (
          <p className="mb-4 flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="size-3 animate-spin" />
            {stages[stages.length - 1].label}
            {stages[stages.length - 1].detail && (
              <span className="opacity-70">{stages[stages.length - 1].detail}</span>
            )}
          </p>
        )}

        {failure && <Failed failure={failure} onModel={(name) => update({ model: name })} />}

        {answered && (
          <div className="space-y-4">
            {stages.length > 0 && (
              <details
                open={showStages}
                onToggle={(event) => setShowStages(event.currentTarget.open)}
                className="rounded-md border text-xs"
              >
                <summary className="cursor-pointer px-3 py-2 text-muted-foreground">
                  how this was found — {stages.length} steps
                </summary>
                <ol className="space-y-0.5 border-t px-3 py-2">
                  {stages.map((stage, index) => (
                    <li
                      key={index}
                      className="text-muted-foreground"
                      style={{ paddingLeft: stage.indent * 12 }}
                    >
                      {stage.label}
                      {stage.detail && <span className="ml-1 opacity-70">{stage.detail}</span>}
                    </li>
                  ))}
                </ol>
              </details>
            )}

            <Results answered={answered} scope={scope} onRead={setReading} />
          </div>
        )}

        {!answered && !failure && !running && (
          <div className="max-w-prose space-y-2 pt-8 text-sm text-muted-foreground">
            <p>Ask in ordinary words, or paste the words you remember.</p>
            <p className="text-xs leading-relaxed">
              Remembered phrases go in <strong className="font-medium">verbatim</strong> — the
              literal half of the search finds text in order, and rewording destroys the one thing
              it had. Rare names and odd spellings likewise: include them exactly.
            </p>
            <p className="text-xs leading-relaxed">
              The answer is the first result about six times in ten, and somewhere in the top five
              about eight. Read several.
            </p>
          </div>
        )}
      </div>

      <Reader library={library} reading={reading} onClose={() => setReading(null)} />
    </div>
  );
}

function Failed({ failure, onModel }: { failure: DyprysError; onModel: (name: string) => void }) {
  // The refusal that exists to be turned into a picker. `choices` is on the
  // error shape for exactly this, and on a two-model index it is the first
  // thing anyone meets.
  if (failure.kind === "model_ambiguous") {
    return <ModelChoices choices={failure.choices} onSelect={onModel} />;
  }
  return (
    <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm">
      <p>{failure.detail}</p>
      {failure.role && (
        <p className="mt-1 text-xs text-muted-foreground">
          missing: {failure.role}. Choose one in Search settings.
        </p>
      )}
    </div>
  );
}
