import { cn, percent } from "@/lib/utils";
import type { LibraryModel } from "@/lib/api";

/**
 * How far a model has embedded its own chunking.
 *
 * Against the model's chunking, not the whole library — a library split two
 * ways would otherwise show every model permanently short of complete. That is
 * what the server already divides by, so this only has to not undo it.
 */
export function CoverageBar({ fraction, className }: { fraction: number; className?: string }) {
  const done = fraction >= 0.999;
  return (
    <span className={cn("inline-flex items-center gap-1.5", className)}>
      <span className="h-1 w-10 overflow-hidden rounded-full bg-muted">
        <span
          className={cn("block h-full rounded-full", done ? "bg-emerald-500" : "bg-sky-500")}
          style={{ width: `${Math.max(2, Math.min(100, fraction * 100))}%` }}
        />
      </span>
      <span className="tabular-nums text-[11px] text-muted-foreground">{percent(fraction)}</span>
    </span>
  );
}

/** Every model that touched a library, because coverage is per model. */
export function ModelCoverage({ models }: { models?: LibraryModel[] }) {
  if (!models?.length) {
    return <span className="text-[11px] text-muted-foreground">not embedded</span>;
  }
  return (
    <span className="flex flex-col gap-0.5">
      {models.map((model) => (
        <span key={model.name} className="flex items-center gap-1.5">
          <span className="max-w-[9rem] truncate font-mono text-[10px] text-muted-foreground">
            {shortModel(model.name)}
          </span>
          <CoverageBar fraction={model.coverage} />
        </span>
      ))}
    </span>
  );
}

/**
 * The readable part of a model handle.
 *
 * `hf_ggml-org_embeddinggemma-300M-Q8_0@b5ce9d77a3fc` is a name that identifies
 * weights exactly and tells a person nothing. The full handle stays available
 * on the Models tab; a picker row needs the middle of it.
 */
export function shortModel(name: string): string {
  const withoutDigest = name.split("@")[0];
  const parts = withoutDigest.split("_");
  return parts.length > 2 ? parts.slice(2).join("_") : withoutDigest;
}
