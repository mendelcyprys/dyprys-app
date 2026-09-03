"""Read chunk text back out of the source files.

The one place that turns `(path, offset, length)` into characters.  Kept
separate and small so it stays obvious that nothing else in the system needs
to touch a book's bytes.

Everything here fails soft.  A library of thousands of books will always have
some file that moved, or sits on a drive that is not mounted today -- and the
vectors for those books are still perfectly good, so a query must degrade to
one unreadable passage rather than dying.  `dyp check` is where the reason gets
explained; this is not the place to raise about it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

# The signature stored per chunk at ingest: first 8 bytes of its SHA-256.
_HASH_BYTES = 8


def read_span(
    path: str | Path,
    offset: int,
    length: int,
    expect_hash: int | None = None,
) -> str | None:
    """The passage at `(offset, length)`, or `None` if it cannot be trusted.

    `None` means the source is unavailable or no longer matches what was
    indexed.  Passing `expect_hash` -- the chunk's stored `content_hash` --
    makes that guarantee exact: the text returned is byte-for-byte the text that
    was embedded, or nothing is returned at all.  It costs a few microseconds on
    3.6 kB, which is nothing beside a 4.6 ms query, and it is the difference
    between showing a stale passage and knowing not to.
    """
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read(length)
    except OSError:
        return None  # deleted, moved, or on a drive that is not mounted

    if len(data) != length:
        return None  # the file shrank; these offsets no longer describe it

    if expect_hash is not None and span_hash(data) != expect_hash:
        return None  # same length, different bytes -- edited without re-indexing

    # Spans always start and end on character boundaries, so a clean decode is
    # expected; "replace" means a corrupt source file degrades one passage
    # rather than crashing a query.
    return data.decode("utf-8", errors="replace")


def span_hash(data: bytes) -> int:
    """The chunk signature ingest stores, computed over `data`."""
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:_HASH_BYTES], "big", signed=True)


# --- displaying a passage whose source has moved on ---------------------------

EXACT = "exact"          # the bytes at the recorded offset are what was embedded
SHIFTED = "shifted"      # found intact elsewhere: the file was edited above it
CHANGED = "changed"      # the file is readable, this passage is not in it
MISSING = "missing"      # the file is gone, unmounted, or unreadable


@dataclass(frozen=True)
class Span:
    """A passage located for *display*, with how much to trust it.

    `read_span` answers "is this exactly what was embedded", which is the only
    question embedding and scoring may ask. Display has a second question it is
    allowed to ask -- "then where did it go?" -- and this carries the answer
    along with the text so a caller can never show a relocated passage without
    knowing that is what it is doing.
    """

    text: str | None
    state: str
    offset: int


def locate_span(
    path: str | Path,
    offset: int,
    length: int,
    expect_hash: int,
    shift: int = 0,
) -> Span:
    """The passage, found at its recorded offset or `shift` bytes from it.

    An edit to a file moves every passage after it by the same amount, and that
    amount is the change in the file's size -- which is recorded per source at
    ingest. So one extra candidate offset recovers the whole tail of an edited
    book, and the stored hash makes the recovery *provable* rather than likely:
    a passage is only ever reported as found when its bytes are byte-for-byte
    the bytes that were embedded.

    Measured on a 4.78 MB book of 1,352 chunks, against the offsets recorded
    before the edit:

        edit                            recovered
        insert 46 bytes at the START      100.0%
        insert 4,000 bytes at the START   100.0%
        delete 2,000 bytes near the START  99.9%
        insert in the MIDDLE               99.9%
        one word changed, same length      99.9%
        two edits at once                  49.9%

    The last row is the point: two edits give two different shifts, one guess
    covers one of them, and the rest come back `CHANGED` rather than coming back
    wrong. There is no version of this that shows a passage it cannot prove.

    `expect_hash` is required, unlike in `read_span`. Without it there is nothing
    to tell the two candidate offsets apart, so the function would return the
    stale offset's bytes labelled `EXACT` -- a claim of proof where no proof was
    possible. A caller with no hash wants `read_span`.
    """
    if expect_hash is None:
        raise ValueError("locate_span needs the chunk's content_hash to prove a match")
    try:
        with open(path, "rb") as handle:
            for candidate in ((offset,) if not shift else (offset, offset + shift)):
                if candidate < 0:
                    continue
                handle.seek(candidate)
                data = handle.read(length)
                if len(data) != length:
                    continue
                if span_hash(data) == expect_hash:
                    state = EXACT if candidate == offset else SHIFTED
                    return Span(data.decode("utf-8", errors="replace"), state, candidate)
    except OSError:
        return Span(None, MISSING, offset)
    return Span(None, CHANGED, offset)


# --- reading a passage with its surroundings ---------------------------------

# How far either side of a chunk to look for a clean edge. A chunk boundary is
# chosen by byte budget, so it lands mid-sentence often enough to matter: the
# first summary this project produced began "neuropeptides, and neurosteroids,
# as well as..." because that is where the chunk started, and the model copied
# the fragment it was given.
CONTEXT_BYTES = 500

# Where a passage may begin or end without reading as though it were cut off.
_OPENS = re.compile(r"(?:[.!?][\"'”)]?\s+|\n\s*)")


def read_window(
    path: str | Path,
    offset: int,
    length: int,
    before: int = CONTEXT_BYTES,
    after: int = CONTEXT_BYTES,
) -> tuple[str, int] | None:
    """The passage plus enough either side to start and end on a sentence.

    Returns `(text, start_offset)` -- the offset is where the returned text
    actually begins in the file, so a citation still names a real byte.

    This does **not** replace `read_span`. What was embedded is the chunk, and
    anything scoring or verifying against the vector must use the chunk. This is
    for reading: a passage shown to a person, or handed to a model that is being
    asked to understand it rather than to match it.
    """
    start = max(0, offset - max(before, 0))
    want = (offset - start) + length + max(after, 0)
    try:
        with open(path, "rb") as handle:
            handle.seek(start)
            raw = handle.read(want)
    except OSError:
        return None
    if not raw:
        return None

    # Snap to character boundaries *before* decoding, rather than decoding with
    # replacement and trimming afterwards. Every offset here is a byte offset,
    # and U+FFFD stands for a variable number of bytes -- counting characters
    # and adding them to a byte position is how the first version returned an
    # offset two bytes wrong on accented text, which is most of the Gutenberg
    # half of a library.
    head = 0
    while head < len(raw) and (raw[head] & 0xC0) == 0x80:
        head += 1
    raw, start = raw[head:], start + head
    for _ in range(4):
        try:
            raw.decode("utf-8")
            break
        except UnicodeDecodeError:
            raw = raw[:-1]
    text = raw.decode("utf-8", errors="replace")

    # Trim the leading context forward to a sentence start, and the trailing
    # context back to a sentence end -- but never into the chunk itself, which
    # is what the reader actually asked for. Each trim is converted back to a
    # byte delta by encoding the part removed.
    head_chars = len(raw[: offset - start].decode("utf-8", errors="replace"))
    if head_chars > 0:
        # The *first* clean edge in the leading context, not the last: the point
        # is to keep as much readable context as possible while still beginning
        # on a sentence. Taking the last one trims the context away entirely.
        opens = [m.end() for m in _OPENS.finditer(text[:head_chars])]
        if opens:
            start += len(text[: opens[0]].encode("utf-8"))
            text = text[opens[0]:]
            head_chars -= opens[0]

    # Where the chunk ends, in characters: decode exactly its own bytes.
    body = text[head_chars:]
    chunk_text = body.encode("utf-8")[:length].decode("utf-8", errors="ignore")
    tail_from = head_chars + len(chunk_text)
    if tail_from < len(text):
        closes = [m.start() + 1 for m in _OPENS.finditer(text[tail_from:])]
        if closes:
            text = text[: tail_from + closes[-1]]

    # Leading whitespace would move the text without moving the offset.
    lstripped = text.lstrip()
    start += len(text[: len(text) - len(lstripped)].encode("utf-8"))
    return lstripped.rstrip(), start
