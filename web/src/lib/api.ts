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
  title: string;
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
}

export interface WeightsFile {
  path: string;
  name: string;
  bytes: number;
  directory: string;
}

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

  status: (name: string) => request<unknown>(`/libraries/${encodeURIComponent(name)}/status`),

  check: (name: string, deep = false) =>
    request<unknown>(`/libraries/${encodeURIComponent(name)}/check?deep=${deep}`),
};
