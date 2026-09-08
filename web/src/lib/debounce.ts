import * as React from "react";

/** A value that stops changing for `delay` ms before it is passed on. */
export function useDebounced<T>(value: T, delay = 250): T {
  const [settled, setSettled] = React.useState(value);
  React.useEffect(() => {
    const timer = window.setTimeout(() => setSettled(value), delay);
    return () => window.clearTimeout(timer);
  }, [value, delay]);
  return settled;
}
