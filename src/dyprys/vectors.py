"""The vector file: one sparse memmap per model, row `chunk id - 1`.

Preallocated to the whole library rather than appended to, so a chunk's row is
derivable from its id and there is no index to keep in agreement with anything.
Untouched rows cost no disk: a 42.8 GB file with ten books written occupies
0.09 GB on APFS.

Rows are stored either as fp32, or as int8 with a per-row scale. Quantisation
costs nothing measurable -- on 11,512 real vectors it left answer recall,
recall@5 and routing recall all unchanged -- and it is what makes 5,000 books
fit: 10.7 GB instead of 42.8 GB. Which format a file uses is recorded against
its model, not guessed from the file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FP32 = "fp32"
INT8 = "int8"

# int8 rows are stored as a per-row scale followed by the values, so one row is
# still one contiguous read and a routed book is still one contiguous slice.
_LEVELS = 127.0

# How much decoded float32 `score()` may hold at once. Small enough to stay in
# cache, large enough that the per-block overhead does not show.
SCORE_BLOCK_BYTES = 8 << 20


def _score_block(dim: int) -> int:
    """Rows per scoring block, so the decoded copy is ~SCORE_BLOCK_BYTES."""
    return max(256, SCORE_BLOCK_BYTES // (dim * 4))


def row_dtype(quantisation: str, dim: int) -> np.dtype:
    if quantisation == INT8:
        return np.dtype([("scale", "<f4"), ("data", "i1", (dim,))])
    return np.dtype(("<f4", (dim,)))


def bytes_per_row(quantisation: str, dim: int) -> int:
    return row_dtype(quantisation, dim).itemsize


class VectorStore:
    """Random-access rows, and contiguous slices for a routed unit."""

    def __init__(self, path: Path, dim: int, rows: int, quantisation: str = FP32):
        self.path = Path(path)
        self.dim = dim
        self.quantisation = quantisation
        self._dtype = row_dtype(quantisation, dim)
        self._open(max(rows, 1))

    # --- file handling -------------------------------------------------------

    def _open(self, rows: int) -> None:
        needed = rows * self._dtype.itemsize
        if not self.path.exists() or self.path.stat().st_size < needed:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # truncate() leaves the new region sparse -- unwritten rows are not
            # allocated until something touches them.
            with open(self.path, "a+b") as handle:
                handle.truncate(needed)
        self.rows = rows
        self._array = np.memmap(self.path, dtype=self._dtype, mode="r+", shape=(rows,))

    def grow(self, rows: int) -> None:
        """Extend to hold at least `rows`, after new books are ingested."""
        if rows > self.rows:
            self.flush()
            del self._array
            self._open(rows)

    def flush(self) -> None:
        self._array.flush()

    def close(self) -> None:
        self.flush()
        del self._array

    # --- writing -------------------------------------------------------------

    def write(self, chunk_id: int, vector: np.ndarray) -> None:
        self.write_many(chunk_id, vector.reshape(1, -1))

    def write_many(self, first_chunk_id: int, block: np.ndarray) -> None:
        start = first_chunk_id - 1
        block = np.asarray(block, dtype=np.float32)
        if self.quantisation == FP32:
            self._array[start : start + len(block)] = block
            return
        # A per-row scale, so a vector with an unusually small largest component
        # still uses the whole int8 range rather than a fraction of it.
        scale = np.abs(block).max(axis=1)
        safe = np.where(scale > 0, scale, 1.0)
        quantised = np.clip(
            np.rint(block / safe[:, None] * _LEVELS), -_LEVELS, _LEVELS
        ).astype(np.int8)
        rows = self._array[start : start + len(block)]
        rows["scale"] = scale
        rows["data"] = quantised

    def copy(self, source_chunk_id: int, target_chunk_id: int) -> None:
        """Carry a vector across an edit.  This is the whole point of salvage.

        Raw rows are copied, so a quantised vector is never decoded and
        re-encoded -- which would lose a little more each time it moved.
        """
        self._array[target_chunk_id - 1] = self._array[source_chunk_id - 1]

    # --- reading -------------------------------------------------------------

    def move_rows(self, source_chunk_id: int, target_chunk_id: int, count: int) -> None:
        """Slide a run of rows down, copying the stored bytes as they are.

        Compaction must not decode and re-encode: a quantised vector loses a
        little each round trip, so a library compacted often would drift away
        from what was embedded.
        """
        src, dst = source_chunk_id - 1, target_chunk_id - 1
        self._array[dst : dst + count] = self._array[src : src + count]

    def read_raw(self, first_chunk_id: int, count: int) -> np.ndarray:
        """A view of the stored rows, undecoded.

        For carrying rows between files -- compaction stages an overlapping move
        through a scratch file -- without the decode/re-encode round trip a
        quantised vector does not survive unchanged.
        """
        start = first_chunk_id - 1
        return self._array[start : start + count]

    def write_raw(self, first_chunk_id: int, rows: np.ndarray) -> None:
        """Store rows exactly as `read_raw` handed them over."""
        start = first_chunk_id - 1
        self._array[start : start + len(rows)] = rows

    def truncate(self, rows: int) -> None:
        """Drop everything past `rows`, after compaction has moved it down."""
        self.flush()
        del self._array
        with open(self.path, "r+b") as handle:
            handle.truncate(rows * self._dtype.itemsize)
        self._open(max(rows, 1))

    def _decode(self, rows: np.ndarray) -> np.ndarray:
        if self.quantisation == FP32:
            return np.asarray(rows, dtype=np.float32)
        return rows["data"].astype(np.float32) * (rows["scale"][:, None] / _LEVELS)

    def read(self, chunk_id: int) -> np.ndarray:
        return self._decode(self._array[chunk_id - 1 : chunk_id])[0]

    def slice(self, first_chunk_id: int, count: int) -> np.ndarray:
        """The rows of a segment, source or book, as float32.

        Materialises the block, so it is for whole-book work such as building
        routing centroids. Scoring a query against the corpus should use
        `score()`, which never holds more than one block at a time.
        """
        start = first_chunk_id - 1
        return self._decode(self._array[start : start + count])

    def score(
        self, first_chunk_id: int, count: int, query: np.ndarray, block: int | None = None
    ) -> np.ndarray:
        """Dot every row in the range against `query`, in bounded memory.

        Decoding a whole range at once would allocate a float32 copy of it --
        42 GB for a flat scan at 5,000 books. Working in blocks keeps peak
        memory at one block regardless of how much is being scanned, which is
        what lets flat search stay available as a correctness fallback.

        The block is sized in bytes rather than rows, because what has to stay
        bounded is the decoded copy and its width is `dim`. A fixed row count
        was quietly dimension-dependent: 65,536 rows is 201 MB at dim 768, and
        it was larger than the whole 53,567-chunk corpus, so the one thing
        blocking exists to prevent is exactly what happened. Measured over that
        corpus, dropping to this budget cut peak allocation from 330 MB to
        13 MB and was 16% *faster*, the decoded block now fitting in cache.
        """
        block = block or _score_block(self.dim)
        start = first_chunk_id - 1
        out = np.empty(count, dtype=np.float32)
        for offset in range(0, count, block):
            span = min(block, count - offset)
            rows = self._array[start + offset : start + offset + span]
            out[offset : offset + span] = self._decode(rows) @ query
        return out
