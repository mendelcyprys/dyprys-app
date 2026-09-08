import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type JobKind, type Libraries, type Role } from "@/lib/api";

export const keys = {
  libraries: ["libraries"] as const,
  health: ["health"] as const,
  notes: (name: string) => ["notes", name] as const,
};

export function useLibraries() {
  return useQuery({ queryKey: keys.libraries, queryFn: api.libraries });
}

export function useHealth() {
  // Cheap, and the only thing that says which models are resident. Polled
  // slowly: a warm dot going stale for ten seconds costs nothing.
  return useQuery({ queryKey: keys.health, queryFn: api.health, refetchInterval: 10_000 });
}

export function useNotes(name: string | null, enabled: boolean) {
  return useQuery({
    queryKey: keys.notes(name ?? ""),
    queryFn: () => api.notes(name!),
    enabled: Boolean(name) && enabled,
    staleTime: Infinity,
  });
}

/** The three registry writes. Each returns the whole listing, so seed it. */
export function useRegistryWrite() {
  const cache = useQueryClient();
  const seed = (payload: Libraries) => cache.setQueryData(keys.libraries, payload);

  return {
    register: useMutation({
      mutationFn: ({
        name,
        path,
        makeDefault,
      }: {
        name: string;
        path: string;
        makeDefault?: boolean;
      }) => api.register(name, path, makeDefault),
      onSuccess: seed,
    }),
    forget: useMutation({ mutationFn: api.forget, onSuccess: seed }),
    makeDefault: useMutation({ mutationFn: api.makeDefault, onSuccess: seed }),
  };
}

/**
 * What is in a library.
 *
 * `pattern` goes to the server rather than being filtered here, so the filter
 * box and `-c` are one matcher: what you type to find a book is what selecting
 * it will mean. Debouncing is the caller's job — this only caches.
 */
export function useBooks(library: string | null, pattern: string, enabled = true) {
  return useQuery({
    queryKey: ["books", library, pattern] as const,
    queryFn: () => api.books(library!, pattern || undefined),
    enabled: Boolean(library) && enabled,
    placeholderData: (previous) => previous,
  });
}

export function useModels(library: string | null) {
  return useQuery({
    queryKey: ["models", library] as const,
    queryFn: () => api.models(library!),
    enabled: Boolean(library),
  });
}

export function useAvailableModels(library: string | null, enabled: boolean) {
  return useQuery({
    queryKey: ["weights", library] as const,
    queryFn: () => api.availableModels(library!),
    enabled: Boolean(library) && enabled,
    staleTime: 60_000,
  });
}

export function useAlias(library: string) {
  const cache = useQueryClient();
  return useMutation({
    mutationFn: ({ model, alias }: { model: string; alias: string }) =>
      api.alias(library, model, alias),
    onSuccess: () => {
      cache.invalidateQueries({ queryKey: ["models", library] });
      cache.invalidateQueries({ queryKey: keys.libraries });
    },
  });
}

/**
 * Load a model before the first question.
 *
 * Loading a 300 MB–1 GB GGUF is the dominant cost of the first search, and
 * doing it when someone picks rather than when they ask moves the wait to where
 * they expect one.
 */
export function useWarm(library: string | null) {
  const cache = useQueryClient();
  return useMutation({
    mutationFn: (model?: string) => api.warm(library!, model),
    onSuccess: () => cache.invalidateQueries({ queryKey: keys.health }),
  });
}

/**
 * Totals, and the chunkings the library has been split by.
 *
 * The chunkings are why this is a hook and not a one-off: a library split two
 * ways cannot be embedded without saying which way, so the browser has to know
 * the sizes before it can offer them.
 */
export function useStatus(library: string | null) {
  return useQuery({
    queryKey: ["status", library] as const,
    queryFn: () => api.status(library!),
    enabled: Boolean(library),
  });
}

export function useCheck(library: string | null) {
  return useQuery({
    queryKey: ["check", library] as const,
    queryFn: () => api.check(library!),
    enabled: Boolean(library),
  });
}

/**
 * Every job kind, polled.
 *
 * On an interval on purpose, and not once: `rate` is computed from the gap
 * between *your own* polls, falling back until a second one arrives to the
 * median this machine has managed. A UI that polls once shows the fallback
 * guess forever and calls it a measurement.
 *
 * Faster while something is running, slow otherwise — and paused when the tab
 * is hidden, since nobody is reading it and a days-long embed does not need
 * watching from a background tab.
 */
export function useJobs(library: string | null) {
  return useQuery({
    queryKey: ["jobs", library] as const,
    queryFn: () => api.jobs(library!),
    enabled: Boolean(library),
    // Derived from the data rather than passed in, so there is one observer and
    // one interval: two hooks on the same key with different intervals poll
    // twice and disagree about the rate each of them measures.
    refetchInterval: (query) =>
      Object.values(query.state.data ?? {}).some((job) => job.running) ? 2_000 : 10_000,
    refetchIntervalInBackground: false,
  });
}

export function useJobLog(library: string | null, kind: JobKind | null) {
  return useQuery({
    queryKey: ["job-log", library, kind] as const,
    queryFn: () => api.job(library!, kind!, 60),
    enabled: Boolean(library) && Boolean(kind),
    refetchInterval: 4_000,
  });
}

export function useJobControls(library: string) {
  const cache = useQueryClient();
  const refresh = () => {
    cache.invalidateQueries({ queryKey: ["jobs", library] });
    cache.invalidateQueries({ queryKey: ["check", library] });
    // An `add` at a new target creates a chunking, and an `embed` binds a model
    // to one. Both are read straight back by the form that started the run.
    cache.invalidateQueries({ queryKey: ["status", library] });
    cache.invalidateQueries({ queryKey: ["models", library] });
    cache.invalidateQueries({ queryKey: keys.libraries });
  };
  return {
    start: useMutation({
      mutationFn: ({ kind, options }: { kind: JobKind; options?: Record<string, unknown> }) =>
        api.startJob(library, kind, options),
      onSuccess: refresh,
    }),
    stop: useMutation({
      mutationFn: ({ kind, force }: { kind: JobKind; force?: boolean }) =>
        api.stopJob(library, kind, force),
      onSuccess: refresh,
    }),
  };
}

/**
 * Cross-session and cross-*interface* memory.
 *
 * Searches run from a terminal appear here, and since the service layer landed
 * so do `--json` ones — which is the point: a question worth asking twice
 * should not have to be reconstructed from memory.
 */
export function useAsked(library: string | null, find?: string) {
  return useQuery({
    queryKey: ["asked", library, find ?? ""] as const,
    queryFn: () => api.asked(library!, 12, find),
    enabled: Boolean(library),
  });
}

export function useOllama(enabled: boolean) {
  return useQuery({
    queryKey: ["ollama"] as const,
    queryFn: api.ollama,
    enabled,
    staleTime: 60_000,
  });
}

export function useDefaults(library: string | null) {
  return useQuery({
    queryKey: ["defaults", library] as const,
    queryFn: () => api.defaults(library!),
    enabled: Boolean(library),
  });
}

/**
 * Remember a model for this library.
 *
 * The write lands in the index, not in this browser, so it is the same setting
 * `dyp models --summariser` writes and the same one `--summarise` with no name
 * reads. That is the point: one library, one answer.
 */
export function useRemember(library: string) {
  const cache = useQueryClient();
  return useMutation({
    mutationFn: ({ role, model }: { role: Role; model: string | null }) =>
      api.remember(library, role, model),
    onSuccess: (payload) => cache.setQueryData(["defaults", library], payload),
  });
}
