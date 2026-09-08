/**
 * The one place that knows the server exists.
 *
 * Two rules hold everywhere below.
 *
 * The library name is a **path parameter on every call**. There is no
 * server-side "current library", which is what lets two tabs run two libraries
 * at once — so nothing here is allowed to remember one.
 *
 * A failure arrives as `{error, detail, choices, role}` with a status that
 * means something (404 unknown, 409 locked, 422 empty query, 503 a model this
 * machine does not have). `DyprysError` keeps all four, because the UI's job in
 * two of those cases is to render `choices` as a picker.
 */

export class DyprysError extends Error {
  constructor(
    readonly kind: string,
    readonly detail: string,
    readonly status: number,
    readonly choices: string[] = [],
    readonly role: string | null = null,
  ) {
    super(detail);
    this.name = "DyprysError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...init,
      headers: init?.body ? { "content-type": "application/json" } : undefined,
    });
  } catch {
    // Distinguished from a refusal: the server is not answering at all, and
    // "check that `dyp serve` is running" is the only useful thing to say.
    throw new DyprysError("unreachable", "the dyprys server is not answering", 0);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new DyprysError(
      body.error ?? "error",
      body.detail ?? response.statusText,
      response.status,
      body.choices ?? [],
      body.role ?? null,
    );
  }
  const type = response.headers.get("content-type") ?? "";
  return (type.includes("json") ? response.json() : response.text()) as Promise<T>;
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body ?? {}) });

// --- what the server says --------------------------------------------------

export interface LibraryModel {
  name: string;
  embedded: number;
  total: number;
  coverage: number;
}

export interface LibraryRow {
  name: string;
  path: string;
  default: boolean;
  /** Registered, but is the index actually there? An unmounted drive is common. */
  exists: boolean;
  /** The library's NOTES.md, if it left any. Read it before searching. */
  notes: string | null;
  books?: number;
  chunks?: number;
  models?: LibraryModel[];
}

export interface Libraries {
  libraries: LibraryRow[];
  registry_path: string;
  /** False means a registry file exists and could not be parsed. Nothing works. */
  registry_readable: boolean;
}

export interface BookSource {
  ordinal: number;
  path: string;
  size_bytes: number;
  /** False: the file is gone. The chunks remain, and search returns text: null. */
  present: boolean;
}

export interface BookRow {
  /** What ingest derived from the filename. Never edited; it moves with the file. */
  title: string;
  /**
   * What someone chose to call it here, or null. Show `label ?? title`.
   *
   * Not identity and not a rename of the file — but `-c` matches it, so a book
   * you have named is a book you can scope to by that name.
   */
  label: string | null;
  /** What is worth remembering about this book. Shown wherever it appears. */
  note: string | null;
  /** The path. The only thing that says *which* book — titles collide. */
  key: string;
  chunks: number;
  lexical_indexed: number;
  sources: BookSource[];
  chunkings: { id: number; target: number; overlap: number; chunks: number }[];
  /** Chunks this model has embedded, by model name. */
  embedded: Record<string, number>;
  /** Chunks this model *could* embed — its own chunking, not the library's. */
  live_chunks: Record<string, number>;
}

export interface ModelRow {
  name: string;
  /** A short thing to type. Stored in the index, so the terminal knows it too. */
  alias: string | null;
  dim: number;
  store: string;
  embedded: number;
  /** Chunks this model could embed — its own chunking, not the library's. */
  live_chunks: number;
  coverage: number;
  disk_bytes: number;
  failures: number;
  carries: number;
  routing: { profiled_books: number; stale_books: number };
  file_path: string | null;
  /** The index remembers these weights; they are no longer on this machine. */
  file_present: boolean;
  /**
   * Which chunking this model embeds, or null until it has embedded anything.
   *
   * A model embeds one chunking and the index refuses to move it, so this is
   * the difference between offering a chunk size and reporting one.
   */
  chunking_id: number | null;
}

export interface Chunking {
  id: number;
  /** Target chunk size in bytes — what `add --target` and `embed --target` name. */
  target: number;
  overlap: number;
  chunks: number;
}

export interface Status {
  books: number;
  sources: number;
  chunks: number;
  text_bytes: number;
  /** Every way this library has been split. More than one is a real choice. */
  chunkings: Chunking[];
  models: {
    name: string;
    dim: number;
    embedded: number;
    /** Against this model's own chunking, never the library's total. */
    live_chunks: number;
    coverage: number;
  }[];
  pending_carries: number;
  failed_chunks: number;
  compaction_interrupted: boolean;
  notes: string | null;
}

export interface WeightsFile {
  path: string;
  name: string;
  bytes: number;
  directory: string;
  /** What the GGUF declares itself to be, e.g. "qwen3", "jina-bert-v2". */
  architecture: string | null;
  /**
   * Whether this file declares itself a cross-encoder — read from its own
   * `pooling_type`, not inferred from its name.
   *
   * **Null means the file did not say**, which is not the same as false: a
   * .gguf converted before the key existed can still rerank. Treating unknown
   * as no would lock someone out of a working model.
   */
  rerank: boolean | null;
}

export interface Result {
  rank: number;
  chunk_id: number;
  book: string;
  chapter: number | null;
  /** The path. The only thing that says *which* book. Cite from this. */
  path: string;
  offset: number;
  /** Comparable only inside this response. Show it; never sort on it. */
  cos: number | null;
  /** How it was found: "vec 1 · phrase 1". More than one score could carry. */
  provenance: string | null;
  /** What the owner wrote about this book, if anything. Usually null. */
  book_note: string | null;
  state: string;
  /** null when the passage could not be proved. Never render it as a quote. */
  text: string | null;
}

export interface Claim {
  cited: number;
  quote: string;
  where?: string;
}

export interface Answer {
  prose: string;
  verified: Claim[];
  rejected: Claim[];
  /** The rephrasing that worked — the answer is about `drawn_from`, not results. */
  retried: string | null;
  /** The rephrasing that also found nothing. A refusal that survived one. */
  refused: string | null;
  drawn_from?: { chunk_id: number; book: string; path: string; offset: number }[];
}

export interface Warning {
  kind: string;
  message: string;
}

export interface Answered {
  query: string;
  mode: string;
  routed: boolean;
  scanned_fraction: number;
  elapsed_ms: number;
  results: Result[];
  /** Always present, empty when there is nothing to say. Always render it. */
  warnings: Warning[];
  answer: Answer | null;
  expansion: unknown;
  models: Record<string, string>;
}

export interface Stage {
  kind: string;
  label: string;
  detail: string;
  indent: number;
}

export interface SearchBody {
  question: string;
  k?: number;
  model?: string | null;
  collection?: string[] | string | null;
  route?: number;
  rerank?: number;
  reranker?: string | null;
  expand?: boolean | string | null;
  expander?: string | null;
  summarise?: boolean | string | null;
  summariser?: string | null;
  full?: boolean;
}

export interface SourceWindow {
  path: string;
  /** Where this text begins in the file. Snapped to a sentence, so not `offset`. */
  offset: number;
  text: string;
  /**
   * Where it stops. The next stretch begins here — exactly, which is the point:
   * asking again at `offset + span` skips whatever the sentence-snap trimmed,
   * and a reader built on that drops a sentence at every join.
   */
  end: number;
  /** The file's length, or null if it could not be measured. `end === bytes` is the end. */
  bytes: number | null;
}

export type JobKind = "embed" | "add" | "route" | "lexical" | "compact";

export interface JobState {
  kind: JobKind;
  /** Something holds this index. The same for every kind — they share a lock. */
  busy: boolean;
  /** This kind is what holds it. Per kind, unlike `busy`. */
  running: boolean;
  /** Which kind holds it, when that can be known rather than guessed. */
  holder_kind: JobKind | null;
  pid: number | null;
  model: string | null;
  done: number | null;
  live: number | null;
  share: number | null;
  /** Chunks per second, measured from the gap between *your own* polls. */
  rate: number | null;
  eta_seconds: number | null;
  book_in_flight: string | null;
  log: string | null;
  log_tail?: string;
}

export interface CheckModel {
  name: string;
  dim: number;
  embedded: number;
  to_copy: number;
  to_embed: number;
  failed: number;
  outstanding: number;
  coverage: number;
  routing: {
    built: boolean;
    stale_books: number;
    drifted_books: number;
    /** Books `--route` cannot return at any rank. Non-zero: run `route`. */
    unprofiled_books: number;
  };
}

/** A book the extractor mangled, or one that holds no text at all. */
export interface BookFault {
  title: string;
  chunks: number;
  /** 90th-percentile token length. High means word boundaries were lost. */
  p90_token?: number;
}

export interface Check {
  deep: boolean;
  sources: number;
  drift: { missing: string[]; changed: string[]; intact: number; clean: boolean };
  live_chunks: number;
  dead_chunks: number;
  lexical_chunks: number;
  /** False: the literal half of every search is incomplete, and says nothing. */
  lexical_complete: boolean;
  /** Books whose text came out as runs of glued-together characters. */
  garbled: BookFault[];
  /** Books with nothing in them. */
  empty: BookFault[];
  models: CheckModel[];
}

export interface AskedHit {
  chunk: number;
  title: string;
  path: string;
  offset: number;
  cos: number;
  why: string | null;
}

export interface AskedRow {
  id: number;
  at: string;
  question: string;
  mode: string;
  ms: number;
  detail: {
    routed: boolean;
    books: number | null;
    models: Record<string, string>;
    expansion: unknown;
    hits: AskedHit[];
    answer: Answer | null;
  };
}

export type Role = "expander" | "summariser" | "reranker";

/** What a library remembers, and so what a bare `true` resolves to. */
export type Defaults = Record<Role, string | null>;

export interface Health {
  ok: boolean;
  loaded: Record<string, string[]>;
  libraries: string[];
}

export const api = {
  health: () => request<Health>("/health"),

  libraries: () => request<Libraries>("/libraries"),

  register: (name: string, path: string, makeDefault?: boolean) =>
    post<Libraries>("/libraries", { name, path, default: makeDefault }),

  forget: (name: string) =>
    request<Libraries>(`/libraries/${encodeURIComponent(name)}`, { method: "DELETE" }),

  makeDefault: (name: string) => post<Libraries>(`/libraries/${encodeURIComponent(name)}/default`),

  /** The library's NOTES.md, raw. Rendered, never interpreted. */
  notes: (name: string) => request<string>(`/libraries/${encodeURIComponent(name)}/notes`),

  books: (name: string, pattern?: string) =>
    request<{ books: BookRow[] }>(
      `/libraries/${encodeURIComponent(name)}/books${
        pattern ? `?pattern=${encodeURIComponent(pattern)}` : ""
      }`,
    ),

  /** Name a book, or say what is worth remembering about it. Display only. */
  describeBook: (
    name: string,
    key: string,
    given: { label?: string | null; note?: string | null },
  ) =>
    post<{ books: BookRow[] }>(`/libraries/${encodeURIComponent(name)}/books/describe`, {
      key,
      ...given,
    }),

  models: (name: string) =>
    request<{ models: ModelRow[] }>(`/libraries/${encodeURIComponent(name)}/models`),

  /** The .gguf files on the machine, so a reranker can be picked not typed. */
  availableModels: (name: string) =>
    request<{ models: WeightsFile[]; searched: string[] }>(
      `/libraries/${encodeURIComponent(name)}/models/available`,
    ),

  alias: (name: string, model: string, alias: string) =>
    post<{ name: string; alias: string }>(
      `/libraries/${encodeURIComponent(name)}/models/${encodeURIComponent(model)}/alias`,
      { alias },
    ),

  /** Load a model before the first question, where the wait is expected. */
  warm: (name: string, model?: string) =>
    post<unknown>(`/libraries/${encodeURIComponent(name)}/warm`, model ? { model } : {}),

  /** Bytes around an offset, snapped to sentences, for a path this index owns. */
  source: (name: string, path: string, offset: number, span: number) =>
    request<SourceWindow>(
      `/libraries/${encodeURIComponent(name)}/source?path=${encodeURIComponent(path)}` +
        `&offset=${offset}&span=${span}`,
    ),

  jobs: (name: string) =>
    request<Record<JobKind, JobState>>(`/libraries/${encodeURIComponent(name)}/jobs`),

  job: (name: string, kind: JobKind, tail = 40) =>
    request<JobState>(`/libraries/${encodeURIComponent(name)}/jobs/${kind}?tail=${tail}`),

  startJob: (name: string, kind: JobKind, options: Record<string, unknown> = {}) =>
    post<{ kind: string; pid: number; log: string; started_at: string }>(
      `/libraries/${encodeURIComponent(name)}/jobs/${kind}`,
      options,
    ),

  stopJob: (name: string, kind: JobKind, force = false) =>
    request<{ stopped: boolean; pid: number | null; why?: string; note?: string }>(
      `/libraries/${encodeURIComponent(name)}/jobs/${kind}?force=${force}`,
      { method: "DELETE" },
    ),

  /** What this library has already been asked — from either frontend. */
  asked: (name: string, limit = 12, find?: string) =>
    request<{ questions: AskedRow[] }>(
      `/libraries/${encodeURIComponent(name)}/asked?limit=${limit}` +
        (find ? `&find=${encodeURIComponent(find)}` : ""),
    ),

  /** What the local ollama server has. A machine fact, not a library one. */
  ollama: () => request<{ models: string[]; host: string }>("/ollama"),

  defaults: (name: string) =>
    request<{ defaults: Defaults }>(`/libraries/${encodeURIComponent(name)}/defaults`),

  /** Stored in the index, so `dyp ask` in a terminal reads the same answer. */
  remember: (name: string, role: Role, model: string | null) =>
    post<{ defaults: Defaults }>(`/libraries/${encodeURIComponent(name)}/defaults`, {
      role,
      model,
    }),

  status: (name: string) => request<Status>(`/libraries/${encodeURIComponent(name)}/status`),

  check: (name: string, deep = false) =>
    request<Check>(`/libraries/${encodeURIComponent(name)}/check?deep=${deep}`),
};

/**
 * A search, as it happens.
 *
 * NDJSON: any number of `{stage}` frames, then exactly one `{result}` or
 * `{error}` frame. A ten-second routed search with no output is
 * indistinguishable from a hang, which is what the stages are for.
 *
 * The last frame is always one or the other — a stream that simply stops is a
 * bug, not an empty result — so ending without one is raised rather than
 * quietly returning nothing.
 */
export async function askStream(
  library: string,
  body: SearchBody,
  onStage: (stage: Stage) => void,
  signal?: AbortSignal,
): Promise<Answered> {
  let response: Response;
  try {
    response = await fetch(`/api/libraries/${encodeURIComponent(library)}/ask/stream`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (failure) {
    if ((failure as Error).name === "AbortError") throw failure;
    throw new DyprysError("unreachable", "the dyprys server is not answering", 0);
  }

  // A refusal before the headers is still a status code — an empty query, an
  // unmatched scope, a model this machine does not have.
  if (!response.ok || !response.body) {
    const failed = await response.json().catch(() => ({}));
    throw new DyprysError(
      failed.error ?? "error",
      failed.detail ?? response.statusText,
      response.status,
      failed.choices ?? [],
      failed.role ?? null,
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const handle = (line: string): Answered | undefined => {
    const frame = JSON.parse(line);
    if (frame.stage) {
      onStage(frame.stage as Stage);
      return undefined;
    }
    if (frame.error) {
      // After the headers a failure cannot be a status code, so it arrives here.
      throw new DyprysError(
        frame.error,
        frame.detail ?? "",
        500,
        frame.choices ?? [],
        frame.role ?? null,
      );
    }
    return frame.result as Answered;
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const result = handle(line);
      if (result) return result;
    }
  }
  if (buffer.trim()) {
    const result = handle(buffer);
    if (result) return result;
  }
  throw new DyprysError("truncated", "the search stopped without an answer", 0);
}
