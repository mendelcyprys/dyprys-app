import * as React from "react";

/**
 * What the question is about: which library, and which books inside it.
 *
 * Client state on purpose. The server has no "current library" — the name is a
 * path parameter on every call — so two tabs can sit on two libraries, and the
 * only thing to remember is what this tab was last looking at.
 *
 * The scope is a **list of patterns**, which is what `collection` now takes.
 * A checkbox list of books cannot be honestly written as one glob, so selected
 * books contribute their own path (unique, unlike a title) and a promoted
 * filter contributes itself. Empty means the whole library, which is the
 * difference between "everything" and "nothing" and is worth not confusing.
 */

const LIBRARY = "dyprys.library";
const scopeKey = (library: string) => `dyprys.scope.${library}`;

interface Selection {
  library: string | null;
  select: (name: string | null) => void;
  /** Empty is the whole library. */
  scope: string[];
  setScope: (patterns: string[]) => void;
}

const Context = React.createContext<Selection>({
  library: null,
  select: () => {},
  scope: [],
  setScope: () => {},
});

function stored(key: string): string[] {
  try {
    const raw = window.localStorage.getItem(key);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((v) => typeof v === "string") : [];
  } catch {
    return [];
  }
}

export function SelectionProvider({ children }: { children: React.ReactNode }) {
  const [library, setLibrary] = React.useState<string | null>(
    () => window.localStorage.getItem(LIBRARY),
  );
  const [scope, setScopeState] = React.useState<string[]>(() =>
    library ? stored(scopeKey(library)) : [],
  );

  const select = React.useCallback((name: string | null) => {
    setLibrary(name);
    // A scope is a set of paths inside one library and means nothing in
    // another, so it is reloaded rather than carried across.
    setScopeState(name ? stored(scopeKey(name)) : []);
    if (name) window.localStorage.setItem(LIBRARY, name);
    else window.localStorage.removeItem(LIBRARY);
  }, []);

  const setScope = React.useCallback(
    (patterns: string[]) => {
      setScopeState(patterns);
      if (library) window.localStorage.setItem(scopeKey(library), JSON.stringify(patterns));
    },
    [library],
  );

  return (
    <Context.Provider value={{ library, select, scope, setScope }}>{children}</Context.Provider>
  );
}

export const useSelection = () => React.useContext(Context);
