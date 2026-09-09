import * as React from "react";
import { CornerDownLeft, Loader2, Search, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { askStream, DyprysError, type Answered, type SearchBody, type Stage } from "@/lib/api";
import { usePending } from "@/lib/pending";
import { useSelection } from "@/lib/selection";
import { useDefaults } from "@/lib/queries";
import { effortLabel, useSearchSettings } from "@/lib/settings";
import { ModelChoices } from "@/rail/model-picker";
import { Reader, type Reading } from "./reader";
import { Results } from "./results";
import { count } from "@/lib/utils";

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
  // The scope the *answer* was found under, not the one the rail holds now.
  // Editing the scope after a search must not silently relabel which of its
  // results escaped it -- the same mistake as reading provenance off a
  // pipeline that has since been run again.
  const [asked, setAsked] = React.useState<string[]>([]);
  const [failure, setFailure] = React.useState<DyprysError | null>(null);
  const [running, setRunning] = React.useState(false);
  const [reading, setReading] = React.useState<Reading | null>(null);
  const [showStages, setShowStages] = React.useState(false);
  // Null until j/k is pressed. Starting at 0 meant the first result scrolled
  // itself to the top of the pane on every search -- which pushed the drafted
  // answer, the thing above it, off the screen entirely.
  const [cursor, setCursor] = React.useState<number | null>(null);
  const abort = React.useRef<AbortController | null>(null);

  const { scope } = useSelection();
  const { settings, update } = useSearchSettings();
  const { pending, taken } = usePending();
  // What the library remembers, so this does not warn about a missing reranker
  // the index can supply on its own.
  const remembered = useDefaults(library).data?.defaults;

  async function run(event?: React.FormEvent, asked?: string) {
    event?.preventDefault();
    const wanted = (asked ?? question).trim();
    if (!wanted || running) return;
    setQuestion(wanted);

    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;

    setRunning(true);
    setStages([]);
    setFailure(null);
    setShowStages(false);
    setCursor(null);

    const body: SearchBody = {
      question: wanted,
      k: 5,
      model: settings.model,
      collection: scope.length ? scope : null,
      // Its own axis, and sent whether or not anything below is on. Routing
      // narrows which books stage 2 reads; the effort below decides how the
      // query is written and how the results are ordered. Nothing couples them
      // -- `_pipeline` takes the router and the reranker as separate arguments
      // and `eval` runs a routed rerank on purpose -- so a UI that made this a
      // step of `effort` would put the biggest speedup out of reach exactly
      // when a large library wanted a better ranking too.
      route: settings.route,
      // Never both. Measured, they are substitutes: together they recover the
      // same answers as the better one alone, at the sum of the costs.
      // `rerank` is the depth: how many candidates the cross-encoder rescores,
      // which is the whole cost. One model pass each -- 7.7s for five and
      // 25.7s for twenty against a 0.6B reranker, where retrieval alone is
      // 0.2s -- so this is the number worth being able to change.
      rerank: settings.effort === "rerank" ? settings.depth : 0,
      reranker: settings.effort === "rerank" ? settings.reranker : null,
      expand: settings.effort === "expand" ? (settings.expander ?? true) : null,
      // `bool | str` on purpose: true means "whatever this library remembers",
      // a string names a model. Sending the name alone made an empty box mean
      // "do not summarise" while the settings sheet said it meant "use the
      // library's own choice" -- the two disagreed and the sheet was wrong.
      summarise: settings.summarise ? (settings.summariser ?? true) : null,
    };

    try {
      setAsked(scope);
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

  // A question handed over from the history list or the palette. Taken once and
  // cleared, because it is an event and not a setting: left in place it would
  // re-run on every render that touched it.
  React.useEffect(() => {
    if (!pending) return;
    taken();
    void run(undefined, pending);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending]);

  // j/k through the results, enter to open the reader -- but never while
  // someone is typing a question, which is the whole of the rest of the time.
  React.useEffect(() => {
    const rows = answered?.results ?? [];
    if (rows.length === 0) return;
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA)$/.test(target.tagName)) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "j" || event.key === "k") {
        event.preventDefault();
        setCursor((at) => {
          if (at === null) return 0;
          return Math.max(0, Math.min(rows.length - 1, event.key === "j" ? at + 1 : at - 1));
        });
      }
      if (event.key === "Enter") {
        const chosen = cursor === null ? undefined : rows[cursor];
        if (!chosen) return;
        event.preventDefault();
        setReading({
          path: chosen.path,
          offset: chosen.offset,
          book: chosen.book,
          text: chosen.text,
        });
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [answered, cursor]);

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
        {/* What this search will do, before it does it — one badge per axis
            that is not at its default, so "routed" and "rerank 10" can both be
            true and both be visible. */}
        <Badge variant="outline">
          {settings.route > 0 ? `routed ${settings.route}` : "whole library"}
        </Badge>
        <Badge variant="outline">
          {settings.effort === "rerank"
            ? `rerank ${settings.depth}`
            : effortLabel(settings.effort).toLowerCase()}
        </Badge>
        {scope.length > 0 && <Badge variant="outline">{count(scope.length, "book")}</Badge>}
        {settings.summarise && (
          <Badge variant="outline">{settings.summariser ?? "summarised"}</Badge>
        )}
        {settings.effort === "rerank" && !settings.reranker && !remembered?.reranker && (
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
                  how this was found — {count(stages.length, "step")}
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

            <Results answered={answered} scope={asked} cursor={cursor} onRead={setReading} />
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
            <p className="flex flex-wrap items-center gap-1 pt-2 text-xs">
              <Key>j</Key> <Key>k</Key> move through results, <Key>enter</Key> opens one in its
              book, <Key>Cmd K</Key> for everything else.
            </p>
          </div>
        )}
      </div>

      <Reader library={library} reading={reading} onClose={() => setReading(null)} />
    </div>
  );
}

function Key({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="rounded border px-1 py-0.5 font-mono text-[10px] text-foreground">
      {children}
    </kbd>
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
