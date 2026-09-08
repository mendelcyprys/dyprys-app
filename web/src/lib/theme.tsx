import * as React from "react";

/**
 * Light, dark, or whatever the machine says.
 *
 * Three states rather than a boolean, because "follow the system" is a real
 * answer and not the absence of one: a laptop that turns dark at sunset should
 * take this page with it, and a two-way toggle can only freeze whichever half
 * you last touched.
 *
 * The class is applied here *and* by an inline script in `index.html` that runs
 * before the first paint. Both are needed: the script stops the flash, this
 * keeps the class right when the choice changes or the system flips underneath
 * a page that is already open.
 */

export type Theme = "light" | "dark" | "system";

const KEY = "dyprys.theme";

function stored(): Theme {
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw === "light" || raw === "dark" ? raw : "system";
  } catch {
    return "system";
  }
}

const Context = React.createContext<{
  theme: Theme;
  /** What is actually on screen, which "system" alone does not tell you. */
  resolved: "light" | "dark";
  setTheme: (theme: Theme) => void;
}>({ theme: "system", resolved: "dark", setTheme: () => {} });

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, set] = React.useState<Theme>(stored);
  const [systemDark, setSystemDark] = React.useState(
    () => window.matchMedia("(prefers-color-scheme: dark)").matches,
  );

  // Kept subscribed even while the choice is explicit, so switching back to
  // "system" does not need a reload to learn what the system currently says.
  React.useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (event: MediaQueryListEvent) => setSystemDark(event.matches);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  const resolved: "light" | "dark" = theme === "system" ? (systemDark ? "dark" : "light") : theme;

  React.useEffect(() => {
    document.documentElement.classList.toggle("dark", resolved === "dark");
  }, [resolved]);

  const setTheme = React.useCallback((next: Theme) => {
    set(next);
    try {
      window.localStorage.setItem(KEY, next);
    } catch {
      // A browser refusing to store this costs a re-pick on the next load.
    }
  }, []);

  const value = React.useMemo(() => ({ theme, resolved, setTheme }), [theme, resolved, setTheme]);
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export const useTheme = () => React.useContext(Context);
