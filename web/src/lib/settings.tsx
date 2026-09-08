import * as React from "react";

/**
 * The choices a search needs that the index does not remember.
 *
 * The reranker is the whole reason this exists. `--expand` and `--summarise`
 * default to whatever the library last used and are stored in the index; the
 * cross-encoder is not, must be named on every search, and bare `rerank` is an
 * error rather than a quiet downgrade. So it lives here, per library, in the
 * browser — which is the only place in this system that can remember it.
 *
 * `effort` is one control rather than two checkboxes on purpose. Measured,
 * `--expand` and `--rerank` are substitutes: together they recover the same
 * answers as the better one alone, at the sum of the costs. A UI with two
 * checkboxes invites the combination that is strictly worse.
 */

export type Effort = "fast" | "expand" | "rerank";

export interface Settings {
  model: string | null;
  reranker: string | null;
  expander: string | null;
  /** How many candidates the cross-encoder rescores. The price of reranking. */
  depth: number;
  /** Draft prose from the passages. Off by default: it costs seconds. */
  summarise: boolean;
  /** Which model drafts it. Null means whatever the library remembers. */
  summariser: string | null;
  effort: Effort;
}

const EMPTY: Settings = {
  model: null,
  reranker: null,
  expander: null,
  // The same default a bare `--rerank` has. It was 20 here, which is twice the
  // cost of the documented default and nothing said so.
  depth: 10,
  summarise: false,
  summariser: null,
  effort: "fast",
};

const key = (library: string) => `dyprys.settings.${library}`;

function stored(library: string | null): Settings {
  if (!library) return EMPTY;
  try {
    const raw = window.localStorage.getItem(key(library));
    return raw ? { ...EMPTY, ...JSON.parse(raw) } : EMPTY;
  } catch {
    return EMPTY;
  }
}

export function useSettings(library: string | null) {
  const [settings, setSettings] = React.useState<Settings>(() => stored(library));

  // Settings belong to one library — a model handle means nothing in another,
  // and a reranker chosen for one is a reasonable default only for that one.
  React.useEffect(() => setSettings(stored(library)), [library]);

  const update = React.useCallback(
    (patch: Partial<Settings>) =>
      setSettings((current) => {
        const next = { ...current, ...patch };
        if (library) {
          try {
            window.localStorage.setItem(key(library), JSON.stringify(next));
          } catch {
            // A browser refusing to store this costs a re-pick, not a failure.
          }
        }
        return next;
      }),
    [library],
  );

  return { settings, update };
}

const Context = React.createContext<{
  settings: Settings;
  update: (patch: Partial<Settings>) => void;
}>({ settings: EMPTY, update: () => {} });

export function SettingsProvider({
  library,
  children,
}: {
  library: string | null;
  children: React.ReactNode;
}) {
  const value = useSettings(library);
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export const useSearchSettings = () => React.useContext(Context);
