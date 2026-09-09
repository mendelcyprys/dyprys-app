import * as React from "react";
import { AlertTriangle, Archive, FileWarning, Loader2, Type } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Check, JobKind } from "@/lib/api";
import { useRemoval } from "@/lib/queries";
import { count } from "@/lib/utils";

/**
 * What is wrong with this index, said before it costs a search.
 *
 * `dyp check` has always computed all of this and the browser read two fields
 * of it — outstanding chunks and unprofiled books. The rest was typed
 * `unknown[]` and never rendered, which is how `neuro` came to hold three books
 * whose extraction failed without anything but a terminal ever saying so.
 *
 * Every entry here shares one shape of failure: **the search still returns k
 * results and still exits 0.** A garbled book is ranked and never matches; a
 * missing file is ranked and returns no text; an incomplete BM25 index takes
 * the literal half of every hybrid search away silently. None of them is an
 * error anywhere, which is exactly why they need saying here.
 *
 * Nothing is shown when nothing is wrong. A panel that always has something in
 * it is a panel nobody reads.
 */
export function Health({
  library,
  check,
  onRun,
  canRun,
}: {
  library: string;
  check?: Check;
  onRun: (kind: JobKind) => void;
  canRun: boolean;
}) {
  const { aside } = useRemoval(library);
  if (!check) return null;

  const faults: React.ReactNode[] = [];

  // Incomplete and over-full are different problems with different fixes, and
  // `lexical_complete` is false for both. Removing books leaves their FTS rows
  // behind until compaction, exactly as it leaves their chunk ids — and `Run
  // lexical` backfills what is missing, so it cannot help with a surplus.
  if (!check.lexical_complete && check.lexical_chunks > check.live_chunks) {
    faults.push(
      <Fault
        key="lexical-stale"
        icon={<Type className="size-4 shrink-0 text-muted-foreground" />}
        title={`The BM25 index holds ${(
          check.lexical_chunks - check.live_chunks
        ).toLocaleString()} rows for books that were removed.`}
        what="Harmless to search — those chunks are gone, so nothing can rank them. They are the same dead space the removed books' chunk ids are, and compaction clears both at once."
        action={
          <Button size="sm" variant="outline" onClick={() => onRun("compact")} disabled={!canRun}>
            Run compact
          </Button>
        }
      />,
    );
  } else if (!check.lexical_complete) {
    faults.push(
      <Fault
        key="lexical"
        icon={<Type className="size-4 shrink-0 text-amber-500" />}
        title={`The BM25 index covers ${check.lexical_chunks.toLocaleString()} of ${check.live_chunks.toLocaleString()} chunks.`}
        what="The literal half of every search is looking at part of the library. A remembered phrase in the missing part cannot be found by its words — and nothing in a result says so."
        action={
          <Button size="sm" onClick={() => onRun("lexical")} disabled={!canRun}>
            Run lexical
          </Button>
        }
      />,
    );
  }

  if (check.drift.missing.length > 0) {
    faults.push(
      <Fault
        key="missing"
        icon={<FileWarning className="size-4 shrink-0 text-destructive" />}
        title={`${count(check.drift.missing.length, "file")} gone from disk.`}
        what="Their chunks are still in the index, so a search ranks them and hands back no text. The Books tab marks which."
        detail={check.drift.missing.slice(0, 4)}
      />,
    );
  }

  if (check.drift.changed.length > 0) {
    faults.push(
      <Fault
        key="changed"
        icon={<FileWarning className="size-4 shrink-0 text-amber-500" />}
        title={`${count(check.drift.changed.length, "file")} edited since indexing.`}
        what="A passage is proved against the bytes recorded for it. Where an edit moved them further than one shift accounts for, the passage comes back unproved and is not shown."
        detail={check.drift.changed.slice(0, 4)}
      />,
    );
  }

  if (check.garbled.length > 0) {
    // Older servers sent no key with a garbled book; without one there is
    // nothing safe to act on, so the button is simply not offered.
    const keys = check.garbled.map((book) => book.key).filter((key): key is string => Boolean(key));
    faults.push(
      <Fault
        key="garbled"
        icon={<AlertTriangle className="size-4 shrink-0 text-amber-500" />}
        title={`${count(check.garbled.length, "book")} look like failed extraction.`}
        what="Words are run together, so nothing tokenises and no query can match them — at full embedding cost. Setting them aside takes them out of every search and keeps the embedding, which is what to do until the text can be extracted again and re-added."
        detail={check.garbled.map(
          (book) =>
            `${book.title} — ${count(book.chunks, "chunk")}${
              book.p90_token ? `, tokens up to ${book.p90_token} characters` : ""
            }`,
        )}
        // This panel described the problem that set-aside was built for and
        // offered no way to do it — the one place in the app where the fix and
        // the diagnosis were a tab apart. By key, never by title: two books can
        // answer to one name, and this is a mutation.
        action={
          keys.length > 0 && (
            <Button
              size="sm"
              variant="outline"
              disabled={aside.isPending}
              onClick={() => aside.mutate({ keys, aside: true })}
            >
              {aside.isPending ? <Loader2 className="animate-spin" /> : <Archive />}
              Set {keys.length === 1 ? "it" : "them"} aside
            </Button>
          )
        }
      />,
    );
  }

  if (check.empty.length > 0) {
    faults.push(
      <Fault
        key="empty"
        icon={<AlertTriangle className="size-4 shrink-0 text-muted-foreground" />}
        title={`${count(check.empty.length, "book")} with no text.`}
        what="Indexed, and holding nothing to search. Usually an extraction that produced an empty file."
        detail={check.empty.map((book) => book.title)}
      />,
    );
  }

  const failed = check.models.filter((model) => model.failed > 0);
  if (failed.length > 0) {
    faults.push(
      <Fault
        key="failed"
        icon={<AlertTriangle className="size-4 shrink-0 text-destructive" />}
        title="Some chunks could not be embedded."
        what="They are skipped by every search under that model, and the coverage figure counts them as done with."
        detail={failed.map((model) => `${model.name} — ${count(model.failed, "chunk")}`)}
      />,
    );
  }

  if (faults.length === 0) return null;

  return (
    <section className="space-y-2 rounded-lg border border-amber-500/40 bg-amber-500/5 p-3">
      <h2 className="text-xs font-medium">
        Things a search will not tell you
        <span className="ml-2 font-normal text-muted-foreground">
          each of these still returns results and still succeeds
        </span>
      </h2>
      {faults}
    </section>
  );
}

function Fault({
  icon,
  title,
  what,
  detail,
  action,
}: {
  icon: React.ReactNode;
  title: string;
  what: string;
  detail?: string[];
  action?: React.ReactNode;
}) {
  return (
    <div className="flex items-start gap-3 rounded-md border bg-background p-3 text-xs">
      {icon}
      <div className="min-w-0 flex-1 space-y-1">
        <p className="font-medium">{title}</p>
        <p className="leading-relaxed text-muted-foreground">{what}</p>
        {detail && detail.length > 0 && (
          <ul className="space-y-0.5 pt-0.5">
            {detail.map((line) => (
              <li key={line} className="truncate font-mono text-[10px] text-muted-foreground">
                {line}
              </li>
            ))}
          </ul>
        )}
      </div>
      {action}
    </div>
  );
}
