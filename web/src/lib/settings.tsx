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
 *
 * `route` is deliberately *not* part of `effort`, and the two must not be
 * folded together. Routing decides which books stage 2 reads; effort decides
 * how the query is written and how the results are ordered. They compose --
 * `eval` runs a routed rerank on purpose -- and the only measured exclusion in
 * this system is expand against rerank. Folding routing into the same control
 * would make the library's biggest speedup unreachable whenever someone wanted
 * a better ranking, which is precisely when a large library needs both.
 */

export type Effort = "fast" | "expand" | "rerank";

/**
 * The three efforts, with the words the UI uses for them — defined once so the
 * sheet that sets one and the badge that reports it cannot say different
 * things. `"fast"` is the stored value and stays that way (it is in everyone's
 * localStorage); *"Plain"* is what it is called now, because routing is the
 * speed control and two things claiming that name is what made routing hard to
 * find.
 */
export const EFFORTS: { value: Effort; label: string; detail: string }[] = [
  {
    value: "fast",
    label: "Plain",
    detail: "the default: the query as you typed it, literal-safe, no extra model",
  },
  {
    value: "expand",
    label: "Expand",
    detail: "rewrite the query into the library\u2019s words, then search \u2014 a few seconds",
  },
  {
    value: "rerank",
    label: "Rerank",
    // No figure: the cost is one pass of a cross-encoder per candidate, so it
    // is set by the depth, the model and the machine rather than by the
    // feature. Measured here at 25.7s for 20 passages against a 0.6B reranker;
    // quoting a number the page cannot know is worse than quoting none.
    detail: "a cross-encoder rescores every candidate \u2014 seconds per passage",
  },
];

/** What to call an effort in a badge. */
export const effortLabel = (effort: Effort) =>
  EFFORTS.find((option) => option.value === effort)?.label ?? effort;

export interface Settings {
  model: string | null;
  reranker: string | null;
  expander: string | null;
  /** How many candidates the cross-encoder rescores. The price of reranking. */
  depth: number;
  /**
   * Books stage 1 narrows to, or 0 for the whole library.
   *
   * Its own axis, not a step of `effort`: see the note above. Needs a routing
   * profile, and a search asked to route without one is a refusal, not a
   * fallback -- so the sheet reads the model's `routing.profiled_books` before
   * offering it.
   */
  route: number;
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
  // Off, like `dyp ask` with no `--route`. It costs about one answer in
  // twenty-five, which is not a price to charge anybody by default.
  route: 0,
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
