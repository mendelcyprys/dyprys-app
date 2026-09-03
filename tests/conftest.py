"""Fixtures shared across the suite.

The stub embedder is the important one: it makes "was this text embedded twice?"
answerable, and it keeps the suite from needing a 300 MB model or a minute of
compute to check that resumption works.
"""

import hashlib

import numpy as np
import pytest

from dyprys import db
from dyprys.embedder import normalise
from dyprys.ingest import ingest_paths

DIM = 8


class StubEmbedder:
    """Deterministic vectors, and a record of exactly what it was asked to do."""

    dim = DIM
    name = "stub-model@000000000000"

    def __init__(self):
        self.seen: list[str] = []
        self.batches: list[int] = []

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode()).digest()[:DIM]
        return np.frombuffer(digest, dtype=np.uint8).astype(np.float32)

    def embed_documents(self, texts):
        self.seen.extend(texts)
        self.batches.append(len(texts))
        return normalise(np.array([self._vector(t) for t in texts], dtype=np.float32))

    def embed_query(self, text):
        return normalise(self._vector(text).reshape(1, -1))[0]


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "index")
    yield connection
    connection.close()


@pytest.fixture
def embedder():
    return StubEmbedder()


@pytest.fixture
def library(conn, tmp_path):
    """Two small books, ingested."""
    lib = tmp_path / "lib"
    lib.mkdir()
    for n in range(2):
        (lib / f"book{n}.txt").write_text(
            "\n\n".join(f"Paragraph {i} of book {n}, about neurons. " * 12 for i in range(40)),
            encoding="utf-8",
        )
    ingest_paths(conn, [lib])
    return lib
