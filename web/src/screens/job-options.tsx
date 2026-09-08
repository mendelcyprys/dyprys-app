import * as React from "react";
import { Play, Scissors } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Tooltip } from "@/components/ui/tooltip";
import { shortModel } from "@/components/coverage";
import type { Chunking, ModelRow, Status, WeightsFile } from "@/lib/api";
import { bytes, cn } from "@/lib/utils";

/**
 * The flags `dyp add` and `dyp embed` take, as forms.
 *
 * `jobs.FLAGS` has allowed all of these since the job runner landed — the
 * browser simply never sent any of them, so every run it started was unbounded,
 * full duty, whole library. That is the wrong default for the one command in
 * this tool that can run for days.
 *
 * Two rules hold throughout. A control that is **already decided** is reported
 * rather than offered: a model bound to a chunking cannot be moved to another,
 * and a registered model's quantisation cannot be changed by a flag. And a
 * value equal to the CLI's own default is **not sent** — an argv that says
 * `--duty 100` and one that says nothing are the same run, and the shorter one
 * is the one worth reading in the log.
 */

/** `chunker.TARGET_BYTES` / `OVERLAP_BYTES`. Kept in step by hand; see below. */
const DEFAULT_TARGET = 3600;
const DEFAULT_OVERLAP = 0;
const DEFAULT_BATCH = 8;

/**
 * Chunk sizes worth starting from. The field stays free — this is a byte count
 * and any of them is legal — but a blank numeric box is a bad way to ask for a
 * decision nobody can make without knowing what the number does.
 *
 * The scale is what matters, not the exact figure: English prose runs about
 * four bytes to the token, so these are roughly 300, 900 and 2,000 tokens. A
 * smaller chunk makes a passage that is precisely the answer and often too
 * short to stand on its own; a larger one carries its context but dilutes the
 * embedding, since one vector has to stand for everything in it.
 */
const SIZES = [
  { target: 1200, label: "Tight", what: "~300 tokens · precise, often needs its neighbours" },
  {
    target: DEFAULT_TARGET,
    label: "Default",
    what: "~900 tokens · a long paragraph or a short section",
  },
  { target: 8000, label: "Wide", what: "~2,000 tokens · carries context, dilutes the vector" },
];

function Field({
  label,
  hint,
  children,
  className,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <label className={cn("flex flex-col gap-1", className)}>
      <span className="text-[11px] font-medium text-muted-foreground">{label}</span>
      {children}
      {hint && <span className="text-[10px] leading-snug text-muted-foreground">{hint}</span>}
    </label>
  );
}

function Toggle({
  checked,
  onChange,
  label,
  hint,
  disabled,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
  hint: string;
  disabled?: boolean;
}) {
  return (
    <div className={cn("flex items-start gap-2", disabled && "opacity-50")}>
      <Checkbox
        checked={checked}
        onCheckedChange={onChange}
        disabled={disabled}
        className="mt-0.5"
      />
      <span className="min-w-0">
        <span className="block text-xs">{label}</span>
        <span className="block text-[10px] leading-snug text-muted-foreground">{hint}</span>
      </span>
    </div>
  );
}

/** A chunking, as a person picks one: the size, and how much text uses it. */
function chunkingLabel(chunking: Chunking): string {
  const overlap = chunking.overlap ? ` / ${chunking.overlap} B overlap` : "";
  return `${chunking.target.toLocaleString()} B${overlap} — ${chunking.chunks.toLocaleString()} chunks`;
}

const select =
  "h-9 w-full rounded-md border border-input bg-transparent px-2 text-xs shadow-sm " +
  "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring " +
  "disabled:cursor-not-allowed disabled:opacity-50";

// --- add --------------------------------------------------------------------

/**
 * Read text in — and decide how it is cut, which is decided here and nowhere
 * else.
 *
 * Re-running this over paths already in the index with a **different** target
 * is not a mistake and not a re-ingest: `ingest_book` recognises the same bytes
 * under a chunking it has not been split by before, records that second
 * chunking alongside the first, and reports the book as "rechunked". Nothing
 * existing is disturbed and no vector is lost. That is the whole mechanism for
 * having two chunk sizes in one library, so it is said on the form.
 */
export function AddOptions({
  status,
  busy,
  onStart,
}: {
  status?: Status;
  busy: boolean;
  onStart: (options: Record<string, unknown>) => void;
}) {
  const [paths, setPaths] = React.useState("");
  const [ext, setExt] = React.useState(".txt");
  const [chapters, setChapters] = React.useState(false);
  const [deep, setDeep] = React.useState(false);
  const [target, setTarget] = React.useState(String(DEFAULT_TARGET));
  const [overlap, setOverlap] = React.useState(String(DEFAULT_OVERLAP));

  const chunkings = status?.chunkings ?? [];
  const size = Number(target);
  const over = Number(overlap);
  const valid = Number.isFinite(size) && size >= 200 && Number.isFinite(over) && over >= 0;
  const existing = chunkings.find((c) => c.target === size && c.overlap === over);
  const lines = paths
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!lines.length || !valid) return;
    onStart({
      paths: lines,
      ext: ext.trim() && ext.trim() !== ".txt" ? ext.trim() : undefined,
      chapters: chapters || undefined,
      deep: deep || undefined,
      target: size !== DEFAULT_TARGET ? size : undefined,
      overlap: over !== DEFAULT_OVERLAP ? over : undefined,
    });
  }

  return (
    <form onSubmit={submit} className="mt-4 space-y-4 border-t pt-4">
      <Field
        label="Paths"
        hint="one per line, on the machine running the server — files or directories"
      >
        {/* A textarea, not an input: these are split on newlines, and an
            <input> cannot hold one, so the split was unreachable code. */}
        <textarea
          value={paths}
          onChange={(event) => setPaths(event.target.value)}
          rows={2}
          spellCheck={false}
          placeholder="/a/directory/of/text"
          className={cn(
            "w-full resize-y rounded-md border border-input bg-transparent px-3 py-2",
            "font-mono text-xs shadow-sm placeholder:text-muted-foreground",
            "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
          )}
        />
      </Field>

      <div className="grid grid-cols-2 gap-3">
        <Field label="Extensions" hint="comma-separated; extraction from PDF or EPUB is upstream">
          <Input
            value={ext}
            onChange={(event) => setExt(event.target.value)}
            placeholder=".txt"
            className="h-9 font-mono text-xs"
            spellCheck={false}
          />
        </Field>
        <Field
          label="Chunk size (bytes)"
          className="col-span-2"
          hint={
            SIZES.find((s) => s.target === size)?.what ??
            "any byte count is legal — the presets are starting points, not limits"
          }
        >
          <div className="flex gap-2">
            {SIZES.map((preset) => (
              <Button
                key={preset.target}
                type="button"
                size="sm"
                variant={preset.target === size ? "secondary" : "outline"}
                className="h-9 flex-1 flex-col gap-0 py-1"
                onClick={() => setTarget(String(preset.target))}
              >
                <span className="text-[11px] font-medium">{preset.label}</span>
                <span className="text-[9px] font-normal opacity-70">
                  {preset.target.toLocaleString()} B
                </span>
              </Button>
            ))}
            <Input
              value={target}
              onChange={(event) => setTarget(event.target.value)}
              inputMode="numeric"
              aria-label="Chunk size in bytes"
              className={cn("h-9 w-24 text-xs tabular-nums", !valid && "border-destructive")}
            />
          </div>
        </Field>
        <Field
          label="Overlap (bytes)"
          hint="buys recall across a boundary, at a proportional share of the embedding bill"
        >
          <Input
            value={overlap}
            onChange={(event) => setOverlap(event.target.value)}
            inputMode="numeric"
            className="h-9 text-xs tabular-nums"
          />
        </Field>
        <div className="space-y-2 pt-5">
          <Toggle
            checked={chapters}
            onChange={setChapters}
            label="Each directory is one book"
            hint="its files are chapters in filename order — the shape an EPUB extraction leaves"
          />
          <Toggle
            checked={deep}
            onChange={setDeep}
            label="Re-hash every file"
            hint="instead of trusting size and mtime; slow, and only needed when you suspect a silent edit"
          />
        </div>
      </div>

      {chunkings.length > 0 && (
        <div className="rounded-md border bg-muted/40 p-2.5 text-[11px] leading-relaxed">
          <p className="flex flex-wrap items-center gap-1.5">
            <Scissors className="size-3 shrink-0 text-muted-foreground" />
            <span className="text-muted-foreground">already split by</span>
            {chunkings.map((chunking) => (
              <Badge
                key={chunking.id}
                variant={chunking.target === size && chunking.overlap === over ? "ok" : "outline"}
              >
                {chunking.target.toLocaleString()} B
              </Badge>
            ))}
          </p>
          <p className="mt-1.5 text-muted-foreground">
            {existing
              ? "Text already cut this way will read as unchanged; anything new is added to this chunking."
              : "A size this library has not used yet. Re-adding paths that are already here " +
                "cuts them a second way alongside the first — nothing existing is disturbed, and " +
                "no vector is lost. A second chunking needs a second model to embed it."}
          </p>
        </div>
      )}

      <Button type="submit" size="sm" variant="outline" disabled={!lines.length || !valid || busy}>
        <Play /> Add {lines.length ? `${lines.length} path${lines.length > 1 ? "s" : ""}` : ""}
      </Button>
    </form>
  );
}

// --- embed ------------------------------------------------------------------

/**
 * The slow one, bounded.
 *
 * Every control here exists because the unbounded run is the wrong default:
 * `--for` and `--duty` are how this shares a laptop, `--limit` is how you test
 * a model on a few hundred chunks before committing days to it, and `-c` is how
 * you embed one shelf.
 *
 * The two that are *not* free choices are handled as refusals rather than
 * inputs, because getting them wrong costs a subprocess that dies a second
 * after it starts — which over HTTP looks like a job that never ran:
 *
 * - **chunking**: bound on a model's first embed and never movable. Offered
 *   only while the model is unbound; reported once it is.
 * - **int8**: settable only as a model is first registered. Afterwards the
 *   conversion is `dyp models --quantise`, and the flag exits 2.
 */
export function EmbedOptions({
  status,
  models,
  weights,
  selected,
  outstanding,
  busy,
  onStart,
}: {
  status?: Status;
  models: ModelRow[];
  /** The .gguf files on the machine — the only way to reach an unbound model. */
  weights: WeightsFile[];
  /** What the rail has chosen, which is the sensible starting point. */
  selected: string | null;
  outstanding: number;
  busy: boolean;
  onStart: (options: Record<string, unknown>) => void;
}) {
  // A registered model's name, or the path of a .gguf that is not registered
  // yet. Both are what `--model` accepts, so this is one field.
  const [model, setModel] = React.useState<string>(selected ?? models[0]?.name ?? "");
  // A path typed by hand. `available` looks in the directories this index's own
  // weights came from, plus $DYPRYS_MODEL_DIR — which finds nothing at all for
  // a library that has never been embedded, the one case that needs a picker
  // most. So there is always a way to say where the file is.
  const [typing, setTyping] = React.useState(false);
  const [typed, setTyped] = React.useState("");
  const [target, setTarget] = React.useState<string>("");
  const [duration, setDuration] = React.useState("");
  const [duty, setDuty] = React.useState("100");
  const [limit, setLimit] = React.useState("");
  const [batch, setBatch] = React.useState(String(DEFAULT_BATCH));
  const [collection, setCollection] = React.useState("");
  const [int8, setInt8] = React.useState(false);

  // The rail is the source of truth while nobody has touched this field.
  const [touched, setTouched] = React.useState(false);
  React.useEffect(() => {
    if (!touched && selected) setModel(selected);
  }, [selected, touched]);

  const chunkings = status?.chunkings ?? [];
  // A cross-encoder is not an embedding model. It declares `pooling_type` RANK,
  // which means it scores a (query, passage) pair rather than producing a
  // vector -- so embedding a library with one builds a store of numbers that no
  // search can use, over hours or days. Hidden rather than badged: unlike the
  // reranker picker, where the wrong choice is one search, the wrong choice
  // here is the whole index.
  const embedders = weights.filter(
    (file) => file.rerank !== true && !models.some((row) => row.file_path === file.path),
  );
  const registered = typing ? undefined : models.find((row) => row.name === model);
  const bound = registered?.chunking_id ?? null;
  const boundTo = chunkings.find((c) => c.id === bound);
  // One chunking is not a choice; more than one and the CLI refuses to guess.
  const mustPickChunking = !bound && chunkings.length > 1;
  const chosen = bound
    ? boundTo
    : (chunkings.find((c) => String(c.id) === target) ??
      (chunkings.length === 1 ? chunkings[0] : undefined));

  const wanted = typing ? typed.trim() : model;
  const dutyNumber = Number(duty);
  const validDuty = Number.isFinite(dutyNumber) && dutyNumber >= 1 && dutyNumber <= 100;
  const ready = Boolean(wanted) && (!mustPickChunking || Boolean(chosen)) && validDuty;

  function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!ready) return;
    onStart({
      model: wanted,
      // The size, not the id: `--target` names bytes. Sent only when the
      // library is split more than one way, which is when it means anything.
      target: chunkings.length > 1 && chosen ? chosen.target : undefined,
      for: duration.trim() || undefined,
      duty: validDuty && dutyNumber !== 100 ? dutyNumber : undefined,
      limit: Number(limit) > 0 ? Number(limit) : undefined,
      batch: Number(batch) > 0 && Number(batch) !== DEFAULT_BATCH ? Number(batch) : undefined,
      collection: collection.trim() || undefined,
      int8: !registered && int8 ? true : undefined,
    });
  }

  return (
    <form onSubmit={submit} className="mt-4 space-y-4 border-t pt-4">
      <div className="grid grid-cols-2 gap-3">
        <Field
          label="Model"
          className="col-span-2"
          hint={
            registered
              ? `registered · ${registered.dim}d · stores ${registered.store}`
              : model
                ? "not registered yet — this run creates it, and its quantisation and chunking are fixed as it does"
                : "which weights build the vectors"
          }
        >
          <select
            value={typing ? "\u0000typed" : model}
            onChange={(event) => {
              setTouched(true);
              if (event.target.value === "\u0000typed") {
                setTyping(true);
              } else {
                setTyping(false);
                setModel(event.target.value);
              }
            }}
            className={select}
          >
            {!model && <option value="">Choose a model…</option>}
            {models.length > 0 && (
              <optgroup label="In this index">
                {models.map((row) => (
                  <option key={row.name} value={row.name}>
                    {row.alias ?? shortModel(row.name)} — {Math.round(row.coverage * 100)}% embedded
                  </option>
                ))}
              </optgroup>
            )}
            {embedders.length > 0 && (
              <optgroup label="On this machine, not in this index">
                {embedders.map((file) => (
                  <option key={file.path} value={file.path}>
                    {file.name} — {bytes(file.bytes)}
                  </option>
                ))}
              </optgroup>
            )}
            <option value={"\u0000typed"}>Somewhere else — type a path…</option>
          </select>
        </Field>

        {typing && (
          <Field
            label="Path to the weights"
            className="col-span-2"
            hint="an absolute path to a .gguf on the machine running the server. Set $DYPRYS_MODEL_DIR to have them listed here instead."
          >
            <Input
              value={typed}
              onChange={(event) => setTyped(event.target.value)}
              placeholder="/path/to/model.gguf"
              className="h-9 font-mono text-xs"
              spellCheck={false}
            />
          </Field>
        )}

        {chunkings.length > 1 && (
          <Field
            label="Chunking"
            className="col-span-2"
            hint={
              bound
                ? "already fixed — a model embeds one chunking and the index refuses to move it. Use a second model for the other size."
                : "a model embeds one chunking, chosen here and never again"
            }
          >
            {bound ? (
              <p className="flex h-9 items-center gap-2 rounded-md border border-dashed px-2 text-xs">
                <Scissors className="size-3 text-muted-foreground" />
                {boundTo ? chunkingLabel(boundTo) : `chunking ${bound}`}
              </p>
            ) : (
              <select
                value={target}
                onChange={(event) => setTarget(event.target.value)}
                className={cn(select, !chosen && "border-amber-500/50")}
              >
                <option value="">Choose a chunk size…</option>
                {chunkings.map((chunking) => (
                  <option key={chunking.id} value={String(chunking.id)}>
                    {chunkingLabel(chunking)}
                  </option>
                ))}
              </select>
            )}
          </Field>
        )}

        <Field label="Stop after" hint="e.g. 45m or 2h — finishes the batch in flight">
          <Input
            value={duration}
            onChange={(event) => setDuration(event.target.value)}
            placeholder="unbounded"
            className="h-9 text-xs"
          />
        </Field>
        <Field label="Chunk limit" hint="stop after this many — for a bounded first look">
          <Input
            value={limit}
            onChange={(event) => setLimit(event.target.value)}
            inputMode="numeric"
            placeholder={outstanding ? `all ${outstanding.toLocaleString()}` : "all"}
            className="h-9 text-xs tabular-nums"
          />
        </Field>
        <Field
          label="Duty (%)"
          hint="share of the time spent working — costs what it says, there is no free headroom"
        >
          <Input
            value={duty}
            onChange={(event) => setDuty(event.target.value)}
            inputMode="numeric"
            className={cn("h-9 text-xs tabular-nums", !validDuty && "border-destructive")}
          />
        </Field>
        <Field label="Batch" hint={`chunks per model call (default ${DEFAULT_BATCH})`}>
          <Input
            value={batch}
            onChange={(event) => setBatch(event.target.value)}
            inputMode="numeric"
            className="h-9 text-xs tabular-nums"
          />
        </Field>

        <Field
          label="Only these books"
          className="col-span-2"
          hint="one pattern — a shelf (neuroscience/), one work (Kandel), or a glob (*Imaging*). Resolved before the weights are opened, so a pattern matching nothing costs milliseconds."
        >
          <Input
            value={collection}
            onChange={(event) => setCollection(event.target.value)}
            placeholder="the whole library"
            className="h-9 font-mono text-xs"
            spellCheck={false}
          />
        </Field>
      </div>

      {registered ? (
        <p className="text-[11px] text-muted-foreground">
          Vectors are stored as <strong className="font-medium">{registered.store}</strong>. That is
          fixed when a model is first registered; converting is{" "}
          <code className="font-mono">dyp models --quantise</code>.
        </p>
      ) : (
        wanted && (
          <Toggle
            checked={int8}
            onChange={setInt8}
            label="Store vectors as int8"
            hint="a quarter of the disk, no measurable loss in ranking — and only settable now, as this model is registered"
          />
        )
      )}

      <div className="flex flex-wrap items-center gap-3">
        <Tooltip
          label={
            !wanted
              ? "choose a model, or type the path to one"
              : mustPickChunking && !chosen
                ? "this library is split more than one way — say which to embed, because a model embeds one and cannot be moved"
                : "resumable: stopping loses at most the batch in flight"
          }
        >
          <span>
            <Button type="submit" size="sm" disabled={!ready || busy}>
              <Play /> Start embedding
            </Button>
          </span>
        </Tooltip>
        {!duration.trim() && !Number(limit) && (
          <span className="text-[11px] text-muted-foreground">
            Unbounded — this runs until it is done or you stop it, which on a large library is days.
          </span>
        )}
      </div>
    </form>
  );
}
