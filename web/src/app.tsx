import * as React from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { DyprysError } from "@/lib/api";
import { useLibraries } from "@/lib/queries";
import { useSelection } from "@/lib/selection";
import { SettingsProvider } from "@/lib/settings";
import { Rail } from "@/rail/rail";
import { NothingRegistered, RegistryUnreadable, ServerUnreachable } from "@/screens/blocked";
import { Workspace, type Tab } from "@/screens/workspace";

export function App() {
  const libraries = useLibraries();
  const { library, select } = useSelection();
  const [tab, setTab] = React.useState<Tab>("Ask");

  // The remembered library may have been forgotten, renamed, or its drive
  // unmounted since this tab last looked. Fall back to the default rather than
  // holding a selection nothing answers for.
  React.useEffect(() => {
    const rows = libraries.data?.libraries;
    if (!rows) return;
    const chosen = rows.find((row) => row.name === library);
    if (chosen?.exists) return;
    const fallback =
      rows.find((row) => row.default && row.exists) ?? rows.find((row) => row.exists);
    if (fallback && fallback.name !== library) select(fallback.name);
  }, [libraries.data, library, select]);

  if (libraries.isLoading) {
    return (
      <Shell>
        <div className="p-10 text-sm text-muted-foreground">Loading…</div>
      </Shell>
    );
  }
  if (libraries.error instanceof DyprysError && libraries.error.kind === "unreachable") {
    return (
      <Shell>
        <ServerUnreachable />
      </Shell>
    );
  }
  if (!libraries.data) {
    return (
      <Shell>
        <ServerUnreachable />
      </Shell>
    );
  }
  if (!libraries.data.registry_readable) {
    return (
      <Shell>
        <RegistryUnreadable path={libraries.data.registry_path} />
      </Shell>
    );
  }
  if (libraries.data.libraries.length === 0) {
    return (
      <Shell>
        <NothingRegistered />
      </Shell>
    );
  }

  const current = libraries.data.libraries.find((row) => row.name === library);

  return (
    <Shell>
      <SettingsProvider library={library}>
        <div className="flex h-full">
          <Rail libraries={libraries.data} onEditScope={() => setTab("Books")} />
          {current?.exists ? (
            <Workspace library={current} tab={tab} onTab={setTab} />
          ) : (
            <main className="flex flex-1 items-center justify-center p-10 text-sm text-muted-foreground">
              Choose a library whose index is present.
            </main>
          )}
        </div>
      </SettingsProvider>
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <TooltipProvider delayDuration={300}>
      <div className="h-dvh overflow-hidden">{children}</div>
    </TooltipProvider>
  );
}
