import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { App } from "./app";
import { PendingProvider } from "./lib/pending";
import { SelectionProvider } from "./lib/selection";
import { ThemeProvider } from "./lib/theme";
import "./index.css";

const queries = new QueryClient({
  defaultOptions: {
    queries: {
      // A refusal is an answer, not a flake: retrying a 404 or a
      // model_ambiguous only delays the picker the user is meant to see.
      retry: (attempt, error) => attempt < 2 && (error as { status?: number }).status === 0,
      staleTime: 5_000,
      refetchOnWindowFocus: false,
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <QueryClientProvider client={queries}>
        <SelectionProvider>
          <PendingProvider>
            <App />
          </PendingProvider>
        </SelectionProvider>
      </QueryClientProvider>
    </ThemeProvider>
  </React.StrictMode>,
);
