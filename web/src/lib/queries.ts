import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type Libraries } from "@/lib/api";

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
export function useBooks(library: string | null, pattern: string) {
  return useQuery({
    queryKey: ["books", library, pattern] as const,
    queryFn: () => api.books(library!, pattern || undefined),
    enabled: Boolean(library),
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
