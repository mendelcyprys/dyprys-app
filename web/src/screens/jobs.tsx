import * as React from "react";
import {
  Ban,
  BookPlus,
  Compass,
  Hammer,
  Loader2,
  Play,
  ScrollText,
  Shrink,
  Type,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tooltip } from "@/components/ui/tooltip";
import { DyprysError, type JobKind, type JobState } from "@/lib/api";
import { useCheck, useJobControls, useJobLog, useJobs, useModels } from "@/lib/queries";
import { useSearchSettings } from "@/lib/settings";
import { cn } from "@/lib/utils";

/**
 * The kinds that build vectors, and so must be told which model to build for.
 *
 * `dyp route` and `dyp embed` refuse to guess on a multi-model index, exactly
 * as search does — and a detached subprocess refuses by exiting a second after
 * it starts, which over HTTP looks like a job that never ran. Passing the
 * model, or refusing to start without one, is the fix; showing the log
 * afterwards is only the consolation.
 */
const NEEDS_MODEL: JobKind[] = ["embed", "route"];

const KINDS: {
  kind: JobKind;
  icon: React.ReactNode;
  title: string;
  what: string;
}[] = [
  {
    kind: "embed",
    icon: <Hammer />,
    title: "Embed",
    what: "turn text into vectors — the one slow, expensive step, and resumable",
  },
  {
    kind: "add",
    icon: <BookPlus />,
    title: "Add",
    what: "read a directory of text into the index",
  },
  {
    kind: "route",
    icon: <Compass />,
    title: "Route",
    what: "profile books so a search can skip most of the library",
  },
  { kind: "lexical", icon: <Type />, title: "Lexical", what: "build the BM25 index" },
  { kind: "compact", icon: <Shrink />, title: "Compact", what: "reclaim space from removed books" },
];

/**
 * What is being built, and what is worth building next.
 *
 * One run seen from two places: `dyp watch` in a terminal follows a run the
 * browser started, and this reports on one a terminal started — because every
 * mutation is a detached subprocess and progress is committed to the index per
 * batch, so this is a read of the index rather than a pipe.
 */
export function Jobs({ library }: { library: string }) {
  const check = useCheck(library);
  const [showLog, setShowLog] = React.useState<JobKind | null>(null);
  const states = useJobs(library).data;
  const controls = useJobControls(library);
  const models = useModels(library).data?.models ?? [];
  const { settings } = useSearchSettings();
  // One model is not a choice, so it is the answer; more than one and nobody
  // may guess, this UI included.
  const model = models.length === 1 ? models[0].name : settings.model;
  const mustChoose = models.length > 1 && !model;

  // When the UI started a run, so a job that exits a second later can say so
  // instead of quietly reading "not running" — which is what a missing
  // `--model` looked like before it was passed.
  const [startedAt, setStartedAt] = React.useState<Partial<Record<JobKind, number>>>({});

  // The index is busy or it is not; which kind, only sometimes. Named `held`
  // rather than `busy` because a card's own `busy` prop means "a click of mine
  // is in flight", which is a different thing entirely.
  const held = Object.values(states ?? {}).find((job) => job.busy);

  function start(kind: JobKind, options: Record<string, unknown> = {}) {
    controls.start.mutate(
      { kind, options: NEEDS_MODEL.includes(kind) && model ? { ...options, model } : options },
      {
        onSuccess: () => {
          setStartedAt((all) => ({ ...all, [kind]: Date.now() }));
          setShowLog(kind);
        },
      },
    );
  }

  const unprofiled = (check.data?.models ?? []).reduce(
    (most, model) => Math.max(most, model.routing.unprofiled_books),
    0,
  );
  const outstanding = (check.data?.models ?? []).reduce(
    (most, model) => Math.max(most, model.outstanding),
    0,
  );

  return (
    <div className="space-y-4 overflow-y-auto">
      {unprofiled > 0 && (
        // The single most valuable nudge here, because the failure it prevents
        // is silent: a book embedded since the last `dyp route` has no profile,
        // so a routed search cannot return it at any rank — and still returns a
        // full k at 200, which looks exactly like a complete search.
        <div className="flex flex-wrap items-center gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-xs">
          <Compass className="size-4 shrink-0 text-amber-500" />
          <span className="flex-1">
            <strong className="font-medium">
              {unprofiled.toLocaleString()} book{unprofiled === 1 ? "" : "s"} have no routing
              profile.
            </strong>{" "}
            A routed search cannot return them at any rank, and still comes back with a full set of
            results — nothing in the output says part of the library was skipped.
          </span>
          <Button
            size="sm"
            onClick={() => start("route")}
            disabled={controls.start.isPending || states?.route.running || mustChoose}
          >
            Run route
          </Button>
        </div>
      )}

      {held && (
        // One lock, five kinds. A held lock says the index is busy and — unless
        // this server started the run — says nothing about what is busy. Said
        // once, here, rather than repeated as five "running" cards.
        <div className="flex flex-wrap items-center gap-3 rounded-lg border border-sky-500/40 bg-sky-500/10 p-3 text-xs">
          <Loader2 className="size-4 shrink-0 animate-spin text-sky-400" />
          <span className="flex-1">
            {held.holder_kind
              ? `A ${held.holder_kind} is running against this index`
              : "Something is running against this index — possibly from a terminal"}
            {held.pid ? ` (pid ${held.pid}).` : "."} Nothing else can start until it finishes.
          </span>
        </div>
      )}

      {controls.start.error instanceof DyprysError && (
        <Held failure={controls.start.error} onLog={() => setShowLog("embed")} />
      )}

      {KINDS.map((each) => (
        <Job
          key={each.kind}
          meta={each}
          state={states?.[each.kind]}
          outstanding={each.kind === "embed" ? outstanding : 0}
          needsModel={NEEDS_MODEL.includes(each.kind) && mustChoose}
          blocked={Boolean(held) && !states?.[each.kind].running}
          startedAt={startedAt[each.kind]}
          onStart={(options) => start(each.kind, options)}
          onStop={(force) => controls.stop.mutate({ kind: each.kind, force })}
          busy={controls.start.isPending || controls.stop.isPending}
          log={showLog === each.kind}
          onLog={() => setShowLog(showLog === each.kind ? null : each.kind)}
        />
      ))}

      {showLog && <Log library={library} kind={showLog} />}
    </div>
  );
}

/**
 * 409, which is an ordinary state and not a failure toast.
 *
 * Something already holds the index's lock — quite possibly the user's own
 * terminal — and the useful thing to offer is the log, not an apology.
 */
function Held({ failure, onLog }: { failure: DyprysError; onLog: () => void }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border p-3 text-xs">
      <Ban className="size-4 shrink-0 text-muted-foreground" />
      <span className="flex-1">{failure.detail}</span>
      {failure.status === 409 && (
        <Button size="sm" variant="outline" onClick={onLog}>
          <ScrollText /> See the log
        </Button>
      )}
    </div>
  );
}

function Job({
  meta,
  state,
  outstanding,
  needsModel,
  blocked,
  startedAt,
  onStart,
  onStop,
  busy,
  log,
  onLog,
}: {
  meta: (typeof KINDS)[number];
  state?: JobState;
  outstanding: number;
  needsModel: boolean;
  blocked: boolean;
  startedAt?: number;
  onStart: (options?: Record<string, unknown>) => void;
  onStop: (force?: boolean) => void;
  busy: boolean;
  log: boolean;
  onLog: () => void;
}) {
  const [paths, setPaths] = React.useState("");
  const running = state?.running ?? false;
  // Started here and no longer running. Nothing records an exit code — the run
  // is detached and outlives this process on purpose — so a job that did its
  // work in three seconds and one that died on a missing flag look identical
  // from here. This says only what is certain, and points at the log, which
  // knows. The four-second floor is the gap between spawning and taking the
  // lock, during which a perfectly healthy run reads as not running.
  const since = startedAt ? Date.now() - startedAt : 0;
  const ended = Boolean(startedAt) && !running && since > 4_000 && since < 120_000;

  return (
    <section className={cn("rounded-lg border p-4", running && "border-sky-500/40 bg-sky-500/5")}>
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <span className="text-muted-foreground [&_svg]:size-4">{meta.icon}</span>
        <div className="min-w-0 flex-1">
          <h3 className="flex items-center gap-2 text-sm font-medium">
            {meta.title}
            {running && (
              <Badge variant="ok">
                <Loader2 className="size-2.5 animate-spin" /> running
              </Badge>
            )}
          </h3>
          <p className="mt-0.5 text-xs text-muted-foreground">{meta.what}</p>
        </div>

        {state?.log && (
          <Button size="sm" variant="ghost" onClick={onLog}>
            <ScrollText /> {log ? "Hide log" : "Log"}
          </Button>
        )}

        {running ? (
          // Safe to offer, and the reason is worth saying on the button: DELETE
          // sends SIGINT and `dyp embed` finishes the batch in flight and
          // commits it, so the next run picks up exactly where this stopped.
          <Tooltip label="stops after the current batch; nothing is lost and the next run resumes there">
            <Button size="sm" variant="outline" onClick={() => onStop(false)} disabled={busy}>
              Stop
            </Button>
          </Tooltip>
        ) : meta.kind === "add" ? null : (
          <Tooltip
            label={
              needsModel
                ? "this library has more than one model — choose one in the rail, because building for the wrong vectors is worse than not building"
                : meta.what
            }
          >
            <span>
              <Button
                size="sm"
                variant="outline"
                onClick={() => onStart()}
                disabled={busy || needsModel || blocked}
              >
                <Play /> Run
              </Button>
            </span>
          </Tooltip>
        )}
      </header>

      {meta.kind === "add" && !running && (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            onStart({
              paths: paths
                .split("\n")
                .map((line) => line.trim())
                .filter(Boolean),
            });
          }}
          className="mt-3 flex items-center gap-2"
        >
          <Input
            value={paths}
            onChange={(event) => setPaths(event.target.value)}
            placeholder="/a/directory/of/text — on the machine running the server"
            className="font-mono text-xs"
            spellCheck={false}
          />
          <Button type="submit" size="sm" variant="outline" disabled={!paths.trim() || busy}>
            <Play /> Run
          </Button>
        </form>
      )}

      {needsModel && !running && (
        <p className="mt-3 text-xs text-amber-500">
          This library has been embedded by more than one model, so `{meta.kind}` refuses to guess
          which vectors it is for. Choose one in the rail.
        </p>
      )}

      {ended && (
        <p className="mt-3 text-xs text-muted-foreground">
          This run has finished. What it did — or why it stopped — is in the log.
        </p>
      )}

      {meta.kind === "embed" && !running && outstanding > 0 && (
        // Never started implicitly, and the button says how much is left rather
        // than pretending it is a small thing: this can run for days.
        <p className="mt-3 text-xs text-amber-500">
          {outstanding.toLocaleString()} chunks are not embedded yet. This is the slow step — it can
          run for hours or days, and it is resumable, so stopping loses nothing.
        </p>
      )}

      {running && <Progress state={state!} />}
    </section>
  );
}

function Progress({ state }: { state: JobState }) {
  const share = state.share ?? 0;
  return (
    <div className="mt-4 space-y-2">
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-sky-500 transition-[width] duration-500"
          style={{ width: `${Math.max(1, Math.min(100, share * 100))}%` }}
        />
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        {state.done !== null && state.live !== null && (
          <span className="tabular-nums">
            {state.done.toLocaleString()} / {state.live.toLocaleString()} chunks
          </span>
        )}
        {state.rate !== null && (
          <Tooltip label="measured from the gap between this page's own polls — the first reading is the median this machine has managed before">
            <span className="cursor-default tabular-nums">{state.rate.toFixed(1)}/s</span>
          </Tooltip>
        )}
        {state.eta_seconds !== null && (
          <span className="tabular-nums">{eta(state.eta_seconds)}</span>
        )}
        {state.book_in_flight && <span className="truncate">{state.book_in_flight}</span>}
        {state.pid && <span className="tabular-nums opacity-60">pid {state.pid}</span>}
      </div>
    </div>
  );
}

function eta(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)}s left`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min left`;
  const hours = seconds / 3600;
  return hours < 48 ? `${hours.toFixed(1)} h left` : `${Math.round(hours / 24)} days left`;
}

function Log({ library, kind }: { library: string; kind: JobKind }) {
  const log = useJobLog(library, kind);
  return (
    <section className="rounded-lg border">
      <header className="border-b px-3 py-2 text-xs text-muted-foreground">
        {log.data?.log ?? `${kind} log`}
      </header>
      <pre className="max-h-72 overflow-auto p-3 font-mono text-[10px] leading-relaxed">
        {log.data?.log_tail?.trim() || "nothing logged yet"}
      </pre>
    </section>
  );
}
