import { Monitor, Moon, Sun } from "lucide-react";
import { Tooltip } from "@/components/ui/tooltip";
import { useTheme, type Theme } from "@/lib/theme";
import { cn } from "@/lib/utils";

const CHOICES: { theme: Theme; icon: React.ReactNode; label: string }[] = [
  { theme: "light", icon: <Sun className="size-3" />, label: "Light" },
  { theme: "dark", icon: <Moon className="size-3" />, label: "Dark" },
  { theme: "system", icon: <Monitor className="size-3" />, label: "Follow the system" },
];

/** Three buttons rather than a switch: "follow the system" is a real answer. */
export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  return (
    <div className="inline-flex rounded-md border p-0.5" role="group" aria-label="Colour theme">
      {CHOICES.map((choice) => (
        <Tooltip key={choice.theme} label={choice.label}>
          <button
            type="button"
            aria-label={choice.label}
            aria-pressed={theme === choice.theme}
            onClick={() => setTheme(choice.theme)}
            className={cn(
              "rounded-[3px] px-2 py-1 text-muted-foreground transition-colors",
              "hover:text-foreground",
              theme === choice.theme && "bg-secondary text-foreground",
            )}
          >
            {choice.icon}
          </button>
        </Tooltip>
      ))}
    </div>
  );
}
