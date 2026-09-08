import * as React from "react";

/**
 * A question handed to the Ask tab from somewhere else — history, the palette.
 *
 * Deliberately tiny and deliberately not part of the selection: the library,
 * the model and the scope are *state*, and this is an *event*. Modelling it as
 * state would leave a question sitting there to be re-run on every render that
 * touched it.
 */
const Context = React.createContext<{
  pending: string | null;
  ask: (question: string) => void;
  taken: () => void;
}>({ pending: null, ask: () => {}, taken: () => {} });

export function PendingProvider({ children }: { children: React.ReactNode }) {
  const [pending, setPending] = React.useState<string | null>(null);
  const value = React.useMemo(
    () => ({ pending, ask: setPending, taken: () => setPending(null) }),
    [pending],
  );
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export const usePending = () => React.useContext(Context);
