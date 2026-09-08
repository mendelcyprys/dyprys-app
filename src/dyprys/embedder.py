"""The embedding model, behind one small interface.

Everything llama.cpp-specific lives here, so the rest of the system deals in
`numpy` arrays and never in model handles.  That also means the embed loop can
be tested against a stub -- the suite must not need a 300 MB GGUF or a minute of
compute to check that resumption works.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from dyprys import errors

# EmbeddingGemma is trained with task prefixes, and they matter: on a sample
# query the documented pair widened the gap between the correct passage and a
# topical distractor from 0.477 to 0.569.  Taken from the model card, not from
# any other implementation.
DOCUMENT_PROMPT = "title: none | text: {}"
QUERY_PROMPT = "task: search result | query: {}"

# Each family was trained with its own prefixes, and they are not
# interchangeable. Using EmbeddingGemma's on nomic does not fail -- it returns
# vectors, and they are worse, which is the shape of every prompt-format bug in
# this project so far: a wrong reranker template made reranking worse than no
# reranking, and a raw prompt made an expansion model hand its instructions
# back. Keyed by `general.architecture` from the GGUF, which is a property of
# the weights rather than of the filename.
PROMPTS = {
    "gemma-embedding": (DOCUMENT_PROMPT, QUERY_PROMPT),
    "nomic-bert": ("search_document: {}", "search_query: {}"),
    "jina-bert-v2": ("{}", "{}"),
    "bert": ("{}", "{}"),
}

# No prefix. Wrong for a model that wants one, but the failure is a smaller
# one than applying another family's prefix, and it is the only safe guess.
NO_PROMPT = ("{}", "{}")


def prompts_for(architecture: str) -> tuple[str, str]:
    """The (document, query) prefixes a model family was trained with."""
    return PROMPTS.get(architecture, NO_PROMPT)


class Embedder:
    """A loaded GGUF embedding model."""

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 2048,
        n_batch: int = 2048,
        n_ubatch: int = 2048,
        n_gpu_layers: int = -1,
    ):
        from llama_cpp import Llama  # imported lazily: tests use a stub instead
        from dyprys.rerank import POOLING_RANK, declares

        self.path = Path(model_path)
        # A cross-encoder is not an embedding model, and llama.cpp will not say
        # so: it loads, and `embed()` returns a number per input instead of a
        # vector. What follows is hours or days of building a store nothing can
        # search. The file declares which it is -- RANK means it scores pairs --
        # so this is read before the weights are opened, not discovered after.
        if declares(self.path).get("pooling_type") == POOLING_RANK:
            raise errors.NotAnEmbedder(
                f"{self.path.name} is a cross-encoder, not an embedding model: "
                f"it scores (query, passage) pairs and cannot produce the "
                f"vectors an index is built from. It is a --reranker.",
                role="model")
        self._llm = Llama(
            model_path=str(self.path),
            embedding=True,
            n_ctx=n_ctx,
            n_batch=n_batch,
            # An encoder must see a whole sequence in one micro-batch --
            # llama.cpp asserts n_ubatch >= n_tokens and aborts the process
            # otherwise. The default of 512 is far below our ~850-token chunks,
            # so this must match the context window, not be left alone.
            n_ubatch=n_ubatch,
            # Offload every layer. The default is 0, i.e. CPU only, which
            # measured 2.37 chunks/s here against 6.54 on Metal -- the
            # difference between 2.3 hours and 6 for the current corpus.
            # Ignored harmlessly where no GPU backend is compiled in.
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.dim = len(self._llm.embed("x"))
        # The full digest identifies the weights; the first twelve characters
        # name them. Both are kept: the short form is what a person reads, the
        # long form is what someone verifies a download against.
        self.digest = _file_digest(self.path)
        self.name = f"{self.path.stem}@{self.digest[:12]}"
        self.file_name = self.path.name
        self.file_bytes = self.path.stat().st_size

    def provenance(self) -> dict:
        """What this weights file is, for anyone who has to obtain it again."""
        return {
            "file_name": self.file_name,
            "file_bytes": self.file_bytes,
            "file_sha256": self.digest,
        }

    @property
    def architecture(self) -> str:
        return str(self._llm.metadata.get("general.architecture", ""))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Unit-length float32 vectors, one row per text."""
        prefix = prompts_for(self.architecture)[0]
        return self._embed([prefix.format(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([prompts_for(self.architecture)[1].format(text)])[0]

    def _embed(self, prompts: list[str]) -> np.ndarray:
        raw = self._llm.embed(prompts)
        # llama-cpp returns a flat vector for a single input and a list for many;
        # normalise the shape before anything downstream has to care.
        if raw and not isinstance(raw[0], list):
            raw = [raw]
        return normalise(np.asarray(raw, dtype=np.float32))


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, so cosine similarity is a dot product.

    Normalising once at write time means every later comparison is a matrix
    multiply with no per-query division.
    """
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    # A zero vector stays zero rather than becoming NaN: it is what a failed
    # chunk is written as, and it must score 0 against everything, not poison a
    # whole query.
    np.divide(vectors, lengths, out=vectors, where=lengths > 0)
    return vectors


def _file_digest(path: Path, chunk: int = 1 << 22) -> str:
    """The SHA-256 of the weights themselves.

    The filename is not identity: swap a different model in under the same name
    and vectors from two different spaces would silently mix. Hashing 300 MB
    costs about 0.15 s once per run, which is nothing beside what follows.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class Truncated:
    """An embedder whose output is cut to a Matryoshka prefix and renormalised.

    `dyp models --truncate` shortens the *stored* vectors and nothing was
    shortening the query, so the model row said 512 while the embedder still
    produced 768 and every command refused to open the index. A truncated index
    was unqueryable, which made the saving worthless.

    Only sound for a model trained with Matryoshka representation learning,
    which is the same condition truncating the store carries.
    """

    def __init__(self, inner, dim: int):
        self.inner, self.dim = inner, dim
        self.name = inner.name

    def _cut(self, block: np.ndarray) -> np.ndarray:
        kept = np.asarray(block)[..., : self.dim]
        # A prefix of a unit vector is not a unit vector, and every score in
        # this system is a dot product that assumes one.
        norms = np.linalg.norm(kept, axis=-1, keepdims=True)
        return np.divide(kept, norms, out=np.zeros_like(kept), where=norms > 0)

    @property
    def architecture(self) -> str:
        return str(self._llm.metadata.get("general.architecture", ""))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._cut(self.inner.embed_documents(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return self._cut(self.inner.embed_query(text))

    def provenance(self):
        return self.inner.provenance()
