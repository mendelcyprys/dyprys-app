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

  makeDefault: (name: string) =>
    post<Libraries>(`/libraries/${encodeURIComponent(name)}/default`),

  /** The library's NOTES.md, raw. Rendered, never interpreted. */
  notes: (name: string) => request<string>(`/libraries/${encodeURIComponent(name)}/notes`),

  status: (name: string) => request<unknown>(`/libraries/${encodeURIComponent(name)}/status`),

  check: (name: string, deep = false) =>
    request<unknown>(`/libraries/${encodeURIComponent(name)}/check?deep=${deep}`),
};
