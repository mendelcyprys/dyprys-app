"""Split a book into chunks.

Everything here is a pure function of the input bytes: no file handles, no
database, no clock.  Same bytes in, same boundaries out, forever.  That matters
because chunk boundaries are written once at ingest time and never recomputed --
re-deriving them per query was 57% of query time in the tool this replaces.

Boundaries are *byte* offsets, and we search for them in the raw bytes rather
than in decoded text.  UTF-8 is self-synchronising: a multi-byte character can
never contain a byte that looks like ASCII whitespace, so splitting on b"\\n\\n"
or b" " can never land in the middle of a character.  Working in bytes also
means the offsets we store are exactly what `file.seek()` wants.
"""

from __future__ import annotations

# Bump when the algorithm changes.  Recorded in the database so we can tell
# whether stored offsets were produced by this code or by an older version.
CHUNKER_VERSION = 1

# ~900 tokens at the ~4 bytes/token that English prose runs at.  The embedding
# model's real tokeniser arrives in step 02; until then this is the budget.
TARGET_BYTES = 3600

# No overlap by default.  Overlap buys recall for answers that straddle a
# boundary and costs a proportional share of the embedding budget (180 h -> 207 h
# at 15%).  Step 04's harness decides whether it earns that; the knob exists now
# so the decision is a parameter change, not a rewrite.
OVERLAP_BYTES = 0

# Exactly the ASCII bytes for which Python's str.isspace() is true.  The last
# four are the file/group/record/unit separators: invisible, meaningless here,
# and common in text extracted from PDFs.  Leaving them in means a passage can
# begin with control junk, and means a byte-trimmed span still looks untrimmed
# to str.strip() -- a mismatch that would quietly confuse everything downstream.
_WHITESPACE = b" \t\n\r\f\v\x1c\x1d\x1e\x1f"

# Break candidates, best first.  Paragraph break beats line break beats sentence
# end beats any space.  The first kind found in the search window wins outright.
_BREAKS = (b"\n\n", b"\n", b". ", b" ")


def chunk_bytes(
    data: bytes,
    target: int = TARGET_BYTES,
    overlap: int = OVERLAP_BYTES,
) -> list[tuple[int, int]]:
    """Return `(offset, length)` byte spans covering the text in `data`.

    Spans are in order and, with `overlap=0`, never overlap.  They do not
    necessarily tile the input: whitespace between chunks is dropped, so a span
    always begins and ends on a non-blank byte.
    """
    if target <= 0:
        raise ValueError(f"target must be positive, got {target}")
    if not 0 <= overlap < target:
        raise ValueError(f"overlap must be in [0, {target}), got {overlap}")

    n = len(data)
    # How far back from the ideal cut we will hunt for a natural boundary.
    slack = max(1, target // 4)

    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        ideal = start + target
        if ideal >= n:
            end = n
        else:
            end = _find_break(data, ideal - slack, ideal + 1)
            if end is None:
                # No natural boundary anywhere in the window -- a table, a long
                # URL, an index page.  Cut hard, but never mid-character.
                end = _utf8_floor(data, ideal)

        span = _tighten(data, start, end)
        if span is not None:
            spans.append(span)

        # `end` reaches `n` only on the last chunk, and there is nothing after it
        # to overlap *into*.  Stepping back by the overlap here would re-enter the
        # loop below `n`, cut the same tail again one byte shorter, and repeat --
        # emitting exactly `overlap` extra spans per file, the shortest of them a
        # single byte.  Those are not merely wasted embeddings: BM25 normalises by
        # document length, so a two-word span outranks the paragraph it was cut
        # from for any term they share.
        if end >= n:
            break

        # Guard against a degenerate `end` making no forward progress.
        start = max(start + 1, end - overlap)

    return spans


def _find_break(data: bytes, lo: int, hi: int) -> int | None:
    """Index just past the best break pattern occurring in `data[lo:hi]`."""
    for pattern in _BREAKS:
        found = data.rfind(pattern, lo, hi)
        if found != -1:
            return found + len(pattern)
    return None


def _utf8_floor(data: bytes, index: int) -> int:
    """Move `index` back to the nearest UTF-8 character boundary.

    Continuation bytes are 0b10xxxxxx; a boundary is any byte that is not one.
    """
    while index > 0 and (data[index] & 0xC0) == 0x80:
        index -= 1
    return index


def _tighten(data: bytes, start: int, end: int) -> tuple[int, int] | None:
    """Trim whitespace off both ends; `None` if nothing but whitespace remains."""
    while start < end and data[start] in _WHITESPACE:
        start += 1
    while end > start and data[end - 1] in _WHITESPACE:
        end -= 1
    if end <= start:
        return None
    return (start, end - start)
