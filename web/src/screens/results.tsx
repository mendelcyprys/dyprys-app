import * as React from "react";
import { AlertTriangle, FileWarning, Quote, ShieldAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Tooltip } from "@/components/ui/tooltip";
import type { Answer, Answered, Result } from "@/lib/api";
import { matchesScope } from "@/lib/scope-match";
import { basename, cn } from "@/lib/utils";
import type { Reading } from "./reader";

/**
 * What a search returned, under three rules that a UI breaks by default.
 *
 *  1. `text: null` is never rendered as a quotation. The passage could not be
 *     proved against its stored hash and `state` says why; coercing null to ""
 *     erases the only signal saying the tool cannot vouch for the words.
 *  2. `cos` is shown and never sorted on. It is comparable only inside one
 *     response and a higher number is not a better answer, so the server's rank
 *     order is the answer.
 *  3. `warnings` is rendered unconditionally — see `Warnings` below.
 */
export function Results({
  answered,
  scope,
  cursor,
  onRead,
}: {
  answered: Answered;
  scope: string[];
  /** Which row j/k is on, or null before either has been pressed. Presentation
   *  only — it never reorders anything. */
  cursor: number | null;
  onRead: (reading: Reading) => void;
}) {
  return (
    <div className="space-y-4">
      <Warnings answered={answered} />
      {answered.answer && <Drafted answer={answered.answer} onRead={onRead} />}

      {answered.results.map((result, index) => (
        <Passage
          key={result.chunk_id}
          result={result}
          scope={scope}
          focused={index === cursor}
          onRead={onRead}
        />
      ))}

      <p className="pt-2 text-[11px] text-muted-foreground">
        {answered.results.length} passages · {answered.mode}
        {answered.routed && " · routed"} · read{" "}
        {(answered.scanned_fraction * 100).toFixed(answered.scanned_fraction < 0.01 ? 2 : 0)}% of
        the library · {(answered.elapsed_ms / 1000).toFixed(1)}s
      </p>
    </div>
  );
}

/**
 * The one failure invisible from the output.
 *
 * A book embedded since the last `dyp route` has no profile, so a routed search
 * cannot return it at *any* rank — and still comes back with a full `k` at 200,
 * which looks exactly like a complete search. On a terminal this is a line on
 * stderr; here it is `warnings`, and nothing else will say it.
 */
function Warnings({ answered }: { answered: Answered }) {
  if (answered.warnings.length === 0) return null;
  return (
    <div className="space-y-2">
      {answered.warnings.map((warning, index) => (
        <div
          key={index}
          className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs"
        >
          <ShieldAlert className="mt-0.5 size-3.5 shrink-0 text-amber-500" />
          <span className="flex-1">{warning.message}</span>
        </div>
      ))}
    </div>
  );
}

function Passage({
  result,
  scope,
  focused,
  onRead,
}: {
  result: Result;
  scope: string[];
  focused: boolean;
  onRead: (reading: Reading) => void;
}) {
  const card = React.useRef<HTMLElement>(null);
  // Top, not "nearest": a passage here is routinely taller than the pane -- a
  // 6,500px chunk out of a pitchbook is ordinary -- and "nearest" then leaves
  // the chosen result wherever it happened to be. Instant, because a smooth
  // scroll across ten thousand pixels is a distraction, not a transition.
  React.useEffect(() => {
    if (focused) card.current?.scrollIntoView({ block: "start", behavior: "auto" });
  }, [focused]);

  const proved = result.text !== null;
  // Deliberate and measured: the exact-phrase leg searches the whole library
  // even under a scope, because confining it took lexical safety from 20/20 to
  // 9/20. So a scoped search can return a book that was not selected, with a
  // citation that looks entirely correct. Said here, or it is read as a bug.
  const escaped =
    scope.length > 0 && !scope.some((pattern) => matchesScope(pattern, result.book, result.path));

  return (
    <article
      ref={card}
      className={cn(
        "rounded-lg border p-4 transition-colors",
        proved ? "hover:border-ring" : "border-dashed opacity-80",
        focused && "border-ring ring-1 ring-ring",
      )}
    >
      <header className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="text-xs tabular-nums text-muted-foreground">{result.rank}.</span>
        <h3 className="min-w-0 flex-1 truncate text-sm font-medium">{result.book}</h3>

        {result.provenance
          ?.split("·")
          .map((part) => part.trim())
          .filter(Boolean)
          .map((part) => (
            <Tooltip key={part} label={explain(part)}>
              <Badge variant="outline" className="cursor-default font-mono">
                {part}
              </Badge>
            </Tooltip>
          ))}

        {result.cos !== null && (
          <Tooltip label="closeness in meaning — comparable only within this search, and a higher number is not a better answer">
            <span className="cursor-default text-[10px] tabular-nums text-muted-foreground">
              cos {result.cos.toFixed(2)}
            </span>
          </Tooltip>
        )}
      </header>

      {escaped && (
        <p className="mt-2 flex items-center gap-1.5 text-[11px] text-amber-500">
          <AlertTriangle className="size-3" />
          outside your scope — found by exact phrase, which is searched library-wide on purpose
        </p>
      )}

      {proved ? (
        <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed">{result.text}</p>
      ) : (
        <p className="mt-3 flex items-start gap-2 text-sm text-muted-foreground">
          <FileWarning className="mt-0.5 size-4 shrink-0" />
          <span>
            This passage could not be proved against the file it came from ({result.state}), so it
            is not shown. The location below is still where it was recorded.
          </span>
        </p>
      )}

      <footer className="mt-3 flex items-center gap-2">
        <button
          onClick={() =>
            onRead({
              path: result.path,
              offset: result.offset,
              book: result.book,
              text: result.text,
            })
          }
          className="truncate font-mono text-[10px] text-muted-foreground underline-offset-2 hover:underline"
        >
          {result.path}:{result.offset}
        </button>
      </footer>
    </article>
  );
}

function explain(part: string): string {
  if (part.startsWith("phrase")) return "the passage holds your words in order — a literal hit";
  if (part.startsWith("words")) return "it shares vocabulary with the query";
  if (part.startsWith("vec")) return "it matched on meaning";
  if (part.startsWith("rerank")) return "a cross-encoder put it here";
  return part;
}

/**
 * The drafted answer, and the two things about it that are easy to lose.
 *
 * A refusal is a first-class answer, not an error: "these passages do not
 * answer the question", surviving a rephrasing, is the surest evidence the
 * library lacks something. And after a successful retry the answer is about
 * `drawn_from` rather than the results beside it — so those passages are named,
 * because a citation has to point at something the reader can see.
 */
function Drafted({ answer, onRead }: { answer: Answer; onRead: (reading: Reading) => void }) {
  const refused = answer.prose.trim() === "NO ANSWER IN PASSAGES";

  if (refused) {
    return (
      <section className="rounded-lg border border-dashed p-4">
        <p className="text-sm">The model reports that these passages do not answer the question.</p>
        {answer.refused ? (
          <p className="mt-2 text-xs text-muted-foreground">
            It was asked again for “{answer.refused}”, and found nothing there either — which is
            good evidence this library does not hold it.
          </p>
        ) : (
          <p className="mt-2 text-xs text-muted-foreground">
            A single weak result is not proof of absence. Try different words before concluding.
          </p>
        )}
      </section>
    );
  }

  return (
    <section className="rounded-lg border bg-card p-4">
      {answer.retried && (
        <div className="mb-3 rounded-md bg-muted/50 p-2.5 text-xs text-muted-foreground">
          The passages below did not answer, so the library was searched again for{" "}
          <strong className="font-medium text-foreground">“{answer.retried}”</strong>. This answer
          is about those passages:
          <ul className="mt-1.5 space-y-0.5">
            {answer.drawn_from?.map((passage) => (
              <li key={passage.chunk_id}>
                <button
                  onClick={() =>
                    onRead({
                      path: passage.path,
                      offset: passage.offset,
                      book: passage.book,
                      text: null,
                    })
                  }
                  className="font-mono text-[10px] underline-offset-2 hover:underline"
                >
                  {basename(passage.path)}:{passage.offset}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="whitespace-pre-wrap text-sm leading-relaxed">{answer.prose}</p>

      {answer.verified.length > 0 && (
        <div className="mt-4 space-y-1.5 border-t pt-3">
          <h4 className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
            checked against the source
          </h4>
          {answer.verified.map((claim, index) => (
            <p key={index} className="flex items-start gap-1.5 text-xs text-muted-foreground">
              <Quote className="mt-0.5 size-3 shrink-0" />
              <span>
                “{claim.quote}” — passage {claim.cited}
              </span>
            </p>
          ))}
        </div>
      )}

      {answer.rejected.length > 0 && (
        <div className="mt-3 space-y-1.5 border-t pt-3">
          <h4 className="text-[11px] font-medium uppercase tracking-wide text-destructive">
            not found in the passages
          </h4>
          {answer.rejected.map((claim, index) => (
            <p key={index} className="text-xs text-muted-foreground">
              “{claim.quote}” — attributed to passage {claim.cited}, and not in it
            </p>
          ))}
          <p className="pt-1 text-[11px] text-muted-foreground">
            Every quote is checked against the text it cites. These did not match, so treat that
            part of the answer as the model’s own.
          </p>
        </div>
      )}
    </section>
  );
}
