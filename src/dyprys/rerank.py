"""Step 08: rescore the shortlist with a cross-encoder.

Aimed by the harness before it was written, and then re-aimed by it. The first
measurement, at 31 books, reported "+5 recall@1 for 324x latency, and recall@5
never moves", and it was wrong twice over.

Re-run frozen at 3,453 books / 284,627 chunks, paired against the *same*
shortlist it reorders: recall@5 gains 28 questions and loses 2 across four
arms, while recall@1 is churn except where the retrieval underneath ranks badly.
This stage fixes *membership* of the top five, not the ordering within it.
"recall@5 never moves" was an artefact of testing only depths 5 and 10, where
the shortlist barely exceeds k and there is nothing to promote from.

It is a compensator rather than an improvement: significant on jina (+21/-8,
p = 0.024) whose weakness is ranking, and on routed search (+18/-6, p = 0.023)
whose penalty is also ranking, and not significant on flat gemma, which already
ranks well. Reranked, jina is indistinguishable from gemma flat (p = 0.851) at a
sixth of the build cost -- which is the case for using it.

Retrieval's recall@20 is the ceiling, and this extracts 99% of it (105 of 106
flat, 103 of 104 on jina). A better cross-encoder has nothing left to win.

It stays off by default on cost alone: ~870 ms per passage scored, a figure that
does not move with corpus size, so 8.9 s a query at depth 10 against 91 ms
routed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from dyprys import db, errors
from dyprys.search import Hit
from dyprys.text import read_span

# `LLAMA_POOLING_TYPE_RANK`, as a plain number so reading a file's metadata does
# not require llama_cpp to be importable.
POOLING_RANK = 4


def declares(model_path: str | Path) -> dict:
    """What a `.gguf` says it is: its architecture and its pooling type.

    A metadata-only load -- 40 ms for a 300 MB file, 100 ms for a 600 MB one --
    so this is cheap enough to run over every candidate in a picker.

    This is the one thing about a model file that can be *known* rather than
    guessed. `pooling_type` is written into the GGUF by whoever converted it,
    and a cross-encoder declares `4` (RANK) where an embedding model declares
    `1` (MEAN) or `2` (CLS). The filename says nothing: `hf_ggml-org_
    embeddinggemma-300M-Q8_0.gguf` and `hf_ggml-org_qwen3-reranker-0.6b-q8_0
    .gguf` are the same shape and one of them is not a reranker.

    Empty when the file cannot be read, or llama_cpp is not installed. Absence
    is not evidence: a caller must not treat "we could not tell" as "no".
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        return {}
    try:
        probe = Llama(model_path=str(model_path), vocab_only=True, verbose=False)
        meta = dict(probe.metadata)
    except Exception:
        # A file that will not open is a question for whoever tries to use it,
        # not for a listing. Saying nothing is the honest answer here.
        return {}
    found: dict = {}
    for key, value in meta.items():
        # Keys are `<arch>.<name>`, and the architecture is the prefix -- which
        # is how this learns the arch without a second lookup.
        if key.endswith(".pooling_type"):
            found["architecture"] = key.rsplit(".", 1)[0]
            try:
                found["pooling_type"] = int(value)
            except (TypeError, ValueError):
                pass
    if "architecture" not in found:
        found["architecture"] = meta.get("general.architecture")
    return found


def can_rerank(model_path: str | Path) -> bool | None:
    """Whether this file declares itself a cross-encoder. None: it did not say.

    Three states rather than two, deliberately. A `.gguf` that cannot be read,
    or one converted before the key existed, is unknown -- and refusing an
    unknown file would lock someone out of a working reranker on the strength of
    missing metadata.
    """
    pooling = declares(model_path).get("pooling_type")
    return None if pooling is None else pooling == POOLING_RANK

# A cross-encoder scores one string containing both query and passage, and the
# framing is not cosmetic: it is what the model was trained on. Measured on 12
# questions whose answer was already in the top 10, putting a correct passage
# first went retrieval 9/12, BGE-style separator 8/12, Qwen3's own template
# 10/12. The wrong template made reranking *worse than not reranking*, and a
# quick test on an obviously-relevant pair could not see it — both formats
# separate easy cases; only the right one ranks plausible rivals.
TEMPLATES = {
    "qwen3": (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the "
        'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
        "<Instruct>: Given a question, retrieve the passage that answers it\n"
        "<Query>: {query}\n"
        "<Document>: {passage}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),
    # BGE and jina rerankers take the bare pair.
    "bge": "{query}</s><s>{passage}",
}


def template_for(model_path: Path, architecture: str | None = None) -> str:
    """Pick a template, which must match what the model was trained on.

    The filename first, because that is what the measurement above was taken
    against. The declared architecture second, for the case the filename says
    nothing -- a BERT-family cross-encoder handed the Qwen3 chat template scores
    *worse than not reranking at all*, and falling straight to `qwen3` made that
    the outcome for every file not named bge or jina.
    """
    name = model_path.stem.lower()
    if "bge" in name or "jina" in name:
        return "bge"
    arch = (architecture or "").lower()
    if arch and "bert" in arch:
        return "bge"
    return "qwen3"


# Measured twice, five months and 111x of corpus apart, and both times a deeper
# shortlist was not better. At 31 books: depth 5 scored 70/110 and depth 10
# 73/110. At 3,453 books, routed: depth 10 scored 80/110 recall@1 and 102/110
# recall@5 against depth 20's 76 and 100 -- +5/-1 and +3/-1 paired, which at
# n=6 and n=4 establishes nothing except that 20 is not buying anything. It
# costs exactly double, because cost is linear in the number of pairs scored.
#
# The reason deeper does not help is that the extra candidates are the ones
# retrieval ranked 11th to 20th: they rarely hold the answer and occasionally
# hold something the cross-encoder likes more than the answer.
DEFAULT_DEPTH = 10


class Reranker:
    """A cross-encoder that scores (query, passage) pairs."""

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        template: str | None = None,
    ):
        from llama_cpp import LLAMA_POOLING_TYPE_RANK, Llama

        self.path = Path(model_path)
        # Before the expensive load, and before any scoring: llama.cpp honours
        # `pooling_type=RANK` on *any* model, so an embedding model loads here
        # without complaint and returns numbers that are not relevance. Measured
        # on one obvious pair, embeddinggemma-300M ranked an irrelevant passage
        # above the answer (-27.4 against -35.8) and said nothing about it. That
        # is the failure this refusal exists to convert into an error.
        said = declares(self.path)
        pooling = said.get("pooling_type")
        if pooling is not None and pooling != POOLING_RANK:
            raise errors.NotAReranker(
                f"{self.path.name} is not a cross-encoder: it declares pooling "
                f"type {pooling}, and a reranker declares {POOLING_RANK}. Used "
                f"as one it returns numbers that are not relevance, and ranks "
                f"silently and plausibly wrong.",
                role="reranker",
            )
        self._llm = Llama(
            model_path=str(self.path),
            embedding=True,
            # Rank pooling turns the model into a scorer: one number per pair
            # rather than a vector. Without it this is just a language model.
            pooling_type=LLAMA_POOLING_TYPE_RANK,
            n_ctx=n_ctx,
            n_batch=n_ctx,
            n_ubatch=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.template_name = template or template_for(self.path,
                                                       said.get("architecture"))
        if self.template_name not in TEMPLATES:
            raise ValueError(
                f"unknown reranker template {self.template_name!r}; "
                f"expected one of {sorted(TEMPLATES)}"
            )
        self.template = TEMPLATES[self.template_name]
        self.name = self.path.stem

    def score(self, query: str, passages: list[str], progress=None) -> list[float]:
        """One relevance score per passage, larger meaning more relevant.

        `progress` is asked between passages whether anyone is still waiting.
        This is the loop that makes a stop button honest: measured on `neuro`,
        rescoring 20 passages is 25.7s of the 25.9s a reranked search takes, so
        a checkpoint anywhere else in the pipeline is a checkpoint that never
        fires while the search is actually slow.
        """
        out = []
        for passage in passages:
            if progress is not None:
                progress.check()
            raw = self._llm.embed(self.template.format(query=query, passage=passage))
            while isinstance(raw, list):
                raw = raw[0]
            out.append(float(raw))
        return out


def passages_for(conn: sqlite3.Connection, hits: list[Hit]) -> list[str | None]:
    """The text behind each hit, read by seek and verified against its hash."""
    texts = []
    for hit in hits:
        row = conn.execute(
            "SELECT byte_offset, byte_length, content_hash FROM chunks WHERE id = ?",
            (hit.chunk_id,),
        ).fetchone()
        located = db.locate(conn, hit.chunk_id)
        texts.append(
            None
            if row is None or located is None
            else read_span(
                located["path"], row["byte_offset"], row["byte_length"], row["content_hash"]
            )
        )
    return texts


def rerank(
    conn: sqlite3.Connection,
    reranker: Reranker,
    query: str,
    hits: list[Hit],
    k: int = 5,
    progress=None,
) -> list[Hit]:
    """Reorder `hits` by cross-encoder score and return the best `k`.

    A hit whose source has vanished keeps its retrieval rank rather than being
    dropped: the reranker cannot judge what it cannot read, and silently losing
    a result is worse than leaving it where retrieval put it.
    """
    if not hits:
        return []

    texts = passages_for(conn, hits)
    readable = [(hit, text) for hit, text in zip(hits, texts) if text is not None]
    if not readable:
        return hits[:k]

    scores = reranker.score(query, [text for _, text in readable], progress)
    scored = [Hit(hit.chunk_id, score) for (hit, _), score in zip(readable, scores)]
    scored.sort(key=lambda h: -h.score)

    # Unreadable passages go after everything the reranker could judge, keeping
    # their relative retrieval order.
    unreadable = [hit for hit, text in zip(hits, texts) if text is None]
    return (scored + unreadable)[:k]
