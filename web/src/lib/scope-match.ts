/**
 * The server's `-c` matcher, mirrored — glob or substring, on title or path,
 * case-insensitively (`search._matches`).
 *
 * Only ever used to decide whether a returned passage came from *outside* the
 * scope, which happens legitimately: the exact-phrase leg searches the whole
 * library even under a scope, deliberately, because confining it took lexical
 * safety from 20/20 to 9/20.
 *
 * A naive `path.includes(pattern)` would be right for a book picked by hand and
 * wrong for every glob — it would badge every result of a `*Imaging*` scope as
 * having escaped it, which teaches the reader to ignore the badge. Mirroring
 * the real matcher is the only version of this worth shipping.
 */
function globToRegExp(pattern: string): RegExp {
  const source = pattern.replace(/[.+^${}()|\\]/g, "\\$&").replace(/[*?]|\[[^\]]*\]/g, (token) => {
    if (token === "*") return ".*";
    if (token === "?") return ".";
    // A character class passes through, with fnmatch's leading ! for negation.
    const body = token.slice(1, -1);
    return `[${body.startsWith("!") ? `^${body.slice(1)}` : body}]`;
  });
  return new RegExp(`^${source}$`);
}

export function matchesScope(pattern: string, title: string, path: string): boolean {
  const needle = pattern.toLowerCase();
  const [book, key] = [title.toLowerCase(), path.toLowerCase()];
  if (/[*?[]/.test(pattern)) {
    const matcher = globToRegExp(needle);
    return matcher.test(book) || matcher.test(key);
  }
  return book.includes(needle) || key.includes(needle);
}
