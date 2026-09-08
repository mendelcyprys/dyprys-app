import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** A byte count as a person reads it. */
export function bytes(n: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = n;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size < 10 && unit > 0 ? size.toFixed(1) : Math.round(size)} ${units[unit]}`;
}

export function percent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

/** The last path segment, which is what distinguishes two books with one title. */
export function basename(path: string): string {
  return path.split("/").filter(Boolean).pop() ?? path;
}

/**
 * "1 book", "3 books". Enough English for the counts this app shows.
 *
 * Here rather than at each call site because "1 books" appeared in four places
 * independently, and it will appear in a fifth the next time someone writes a
 * count by hand.
 */
export function count(n: number, one: string, many = `${one}s`): string {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`;
}
