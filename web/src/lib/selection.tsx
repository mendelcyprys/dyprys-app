import * as React from "react";

/**
 * Which library the question is about.
 *
 * Client state on purpose. The server has no "current library" — the name is a
 * path parameter on every call — so two tabs can sit on two libraries, and the
 * only thing to remember is what this tab was last looking at.
 */

const STORED = "dyprys.library";

interface Selection {
  library: string | null;
  select: (name: string | null) => void;
}

const Context = React.createContext<Selection>({ library: null, select: () => {} });

export function SelectionProvider({ children }: { children: React.ReactNode }) {
  const [library, setLibrary] = React.useState<string | null>(
    () => window.localStorage.getItem(STORED),
  );

  const select = React.useCallback((name: string | null) => {
    setLibrary(name);
    if (name) window.localStorage.setItem(STORED, name);
    else window.localStorage.removeItem(STORED);
  }, []);

  return <Context.Provider value={{ library, select }}>{children}</Context.Provider>;
}

export const useSelection = () => React.useContext(Context);
