"""Reranking reorders a shortlist, and never silently loses one."""

import pytest

from dyprys import db
from dyprys.embed import store_for
from dyprys.rerank import passages_for, rerank
from dyprys.search import Hit

DIM = 8


class StubReranker:
    """Scores by how often a query word appears — deterministic and inspectable."""

    name = "stub-reranker"

    def __init__(self):
        self.calls = []

    def score(self, query, passages, progress=None):
        # `progress` is asked between passages whether anyone is still waiting;
        # a stub that scores instantly has nothing to check, but it has to
        # accept the argument or it is no longer the same interface.
        self.calls.append((query, len(passages)))
        words = query.lower().split()
        return [sum(p.lower().count(w) for w in words) / (len(p) or 1) for p in passages]


@pytest.fixture
def indexed(conn, tmp_path, library):
    model = db.model_id(conn, "stub", DIM)
    store_for(conn, tmp_path / "index", model, DIM)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    return model


def test_passages_are_read_back_for_the_hits_given(conn, indexed):
    texts = passages_for(conn, [Hit(1, 0.5), Hit(2, 0.4)])
    assert len(texts) == 2
    assert all(t and t.strip() == t for t in texts)


def test_it_reorders_by_the_rerankers_opinion(conn, indexed):
    """Retrieval order must not survive when the reranker disagrees."""
    reranker = StubReranker()
    # chunk 5 mentions book1; a query about book1 should pull it up from last
    hits = [Hit(1, 0.9), Hit(2, 0.8), Hit(6, 0.7)]

    out = rerank(conn, reranker, "book1", hits, k=3)

    assert {h.chunk_id for h in out} == {1, 2, 6}
    assert [h.score for h in out] == sorted((h.score for h in out), reverse=True)


def test_only_k_come_back(conn, indexed):
    out = rerank(conn, StubReranker(), "paragraph", [Hit(i, 0.5) for i in range(1, 9)], k=3)
    assert len(out) == 3


def test_the_shortlist_is_what_gets_scored(conn, indexed):
    """The reranker's cost is one pass per candidate, so depth is the price."""
    reranker = StubReranker()
    rerank(conn, reranker, "paragraph", [Hit(i, 0.5) for i in range(1, 7)], k=2)
    assert reranker.calls == [("paragraph", 6)]


def test_an_empty_shortlist_stays_empty(conn, indexed):
    assert rerank(conn, StubReranker(), "anything", [], k=5) == []


def test_a_hit_whose_source_vanished_is_kept_not_dropped(conn, indexed, library):
    """The reranker cannot judge what it cannot read; losing it would be worse."""
    (library / "book0.txt").unlink()
    hits = [Hit(i, 0.5) for i in range(1, 9)]

    out = rerank(conn, StubReranker(), "paragraph", hits, k=8)

    assert len(out) == len(hits)
    assert {h.chunk_id for h in out} == {h.chunk_id for h in hits}


def test_unreadable_hits_sort_after_judged_ones(conn, indexed, library):
    (library / "book0.txt").unlink()
    book0 = [
        r["id"] for r in conn.execute(
            "SELECT c.id FROM chunks c JOIN segments seg "
            " ON c.id >= seg.chunk_start AND c.id < seg.chunk_start + seg.chunk_count "
            "JOIN sources src ON src.id = seg.source_id "
            "JOIN books b ON b.id = src.book_id WHERE b.title = 'book0'")
    ]
    hits = [Hit(i, 0.5) for i in range(1, 13)]

    out = rerank(conn, StubReranker(), "paragraph", hits, k=12)

    judged = [h.chunk_id for h in out if h.chunk_id not in book0]
    unread = [h.chunk_id for h in out if h.chunk_id in book0]
    assert out[: len(judged)] == [h for h in out if h.chunk_id in judged]
    assert unread  # they are still present, just last


def test_everything_unreadable_falls_back_to_retrieval_order(conn, indexed, library):
    for name in ("book0.txt", "book1.txt"):
        (library / name).unlink()
    hits = [Hit(3, 0.9), Hit(1, 0.8), Hit(2, 0.7)]

    out = rerank(conn, StubReranker(), "anything", hits, k=2)

    assert [h.chunk_id for h in out] == [3, 1]


# --- the template is part of the model, not decoration ------------------------


def test_the_template_is_chosen_from_the_model_name(conn):
    from pathlib import Path
    from dyprys.rerank import template_for

    assert template_for(Path("hf_ggml-org_qwen3-reranker-0.6b-q8_0.gguf")) == "qwen3"
    assert template_for(Path("bge-reranker-v2-m3-Q8_0.gguf")) == "bge"
    assert template_for(Path("jina-reranker-v2.gguf")) == "bge"


def test_every_template_carries_both_halves_of_the_pair(conn):
    """A template that drops the query or the passage scores nothing useful."""
    from dyprys.rerank import TEMPLATES

    for name, template in TEMPLATES.items():
        filled = template.format(query="QQQ", passage="PPP")
        assert "QQQ" in filled, name
        assert "PPP" in filled, name


def test_cli_rerank_default_matches_the_module():
    """The default depth lives in one place, and `--rerank` must not drift from it.

    `cli.py` spells it as a literal rather than importing it, because
    `dyprys.rerank` reaches `dyprys.search` and therefore numpy, and `dyp --help`
    should not pay for that. The cost of the literal is this test.
    """
    import re
    from pathlib import Path

    from dyprys.rerank import DEFAULT_DEPTH

    source = Path(__file__).parent.parent / "src" / "dyprys" / "cli.py"
    text = source.read_text()
    consts = re.findall(r'"--rerank".*?const=(\d+)', text)
    assert consts, "no --rerank default found in cli.py"
    assert {int(c) for c in consts} == {DEFAULT_DEPTH}, (
        f"cli.py offers --rerank default {consts}, rerank.DEFAULT_DEPTH is {DEFAULT_DEPTH}"
    )
    # And the sentence the user actually reads. These drifted apart the first
    # time the default moved: const said 10 while the help still said 20, which
    # is worse than either being wrong on its own.
    helps = re.findall(r"cross-encoder \(default (\d+)\)", text)
    assert helps, "no --rerank help text found in cli.py"
    assert {int(h) for h in helps} == {DEFAULT_DEPTH}, (
        f"--rerank help says {helps}, rerank.DEFAULT_DEPTH is {DEFAULT_DEPTH}"
    )


def test_rescoring_stops_when_nobody_is_waiting():
    """The loop where a stop button either works or does not.

    Exercised against the real `Reranker.score` rather than `StubReranker`,
    because the checkpoint lives in the loop that costs the time and a stub that
    scores instantly replaces exactly that loop -- testing it through the stub
    would assert only that the stub was written to pass.

    Measured on a real corpus, rescoring 20 passages is 25.7s of the 25.9s a
    reranked search takes, so a checkpoint anywhere else in the pipeline never
    fires while the search is actually slow. Hanging up mid-search took the
    *next* search from 28.0s to 1.7s once this was here.
    """
    from dyprys import errors
    from dyprys.progress import Progress
    from dyprys.rerank import Reranker

    class GoneAfter(Progress):
        def __init__(self, after):
            self.after, self.asked = after, 0

        def event(self, event):
            pass

        def cancelled(self):
            self.asked += 1
            return self.asked > self.after

    class Weights:
        def __init__(self):
            self.scored = 0

        def embed(self, _text):
            self.scored += 1
            return [0.5]

    # Built without a GGUF: `score` needs the template and the weights, and
    # loading half a gigabyte to assert a loop counter would be absurd.
    reranker = object.__new__(Reranker)
    reranker._llm = Weights()
    reranker.template = "{query} {passage}"

    say = GoneAfter(after=2)
    with pytest.raises(errors.Abandoned):
        reranker.score("neurons", ["one", "two", "three", "four", "five"], say)

    assert reranker._llm.scored == 2, (
        f"scored {reranker._llm.scored} passages after being abandoned at 2 -- "
        f"the check must sit inside the per-passage loop, not around it")


def test_a_model_that_is_not_a_cross_encoder_is_refused(monkeypatch, tmp_path):
    """The refusal that exists because the alternative is a silent wrong answer.

    `llama.cpp` honours `pooling_type=RANK` on *any* model, so an embedding
    model loads as a reranker without complaint and returns one number per pair
    that looks exactly like a relevance score. Measured on one obvious pair --
    an on-topic passage against a passage about bananas -- embeddinggemma-300M
    put the bananas first (-27.4 against -35.8) and raised nothing.

    So the check is on what the file *declares*: `pooling_type` is written into
    the GGUF at conversion, and a cross-encoder says RANK (4) where an embedding
    model says MEAN (1) or CLS (2). That is a fact from the file, not the
    filename guess this module deliberately refuses to make elsewhere.
    """
    from dyprys import errors, rerank as rerank_mod

    weights = tmp_path / "embedder.gguf"
    weights.write_bytes(b"not really a gguf")
    monkeypatch.setattr(rerank_mod, "declares",
                        lambda path: {"architecture": "bert", "pooling_type": 1})

    with pytest.raises(errors.NotAReranker) as refused:
        rerank_mod.Reranker(weights)

    assert "cross-encoder" in refused.value.message
    assert refused.value.role == "reranker", (
        "a frontend offers a picker off `role`; without it this is just a string")


def test_a_file_that_says_nothing_about_itself_is_allowed(monkeypatch, tmp_path):
    """Unknown is not no.

    A `.gguf` converted before the key existed, or one this machine cannot open
    to read metadata, declares nothing -- and refusing on that would lock
    someone out of a reranker that works. The load below fails for its own
    reasons; what matters is that it is not `NotAReranker`.
    """
    from dyprys import errors, rerank as rerank_mod

    weights = tmp_path / "quiet.gguf"
    weights.write_bytes(b"not really a gguf")
    monkeypatch.setattr(rerank_mod, "declares", lambda path: {})

    with pytest.raises(Exception) as raised:
        rerank_mod.Reranker(weights)

    assert not isinstance(raised.value, errors.NotAReranker)


def test_the_template_falls_back_to_the_declared_architecture(tmp_path):
    """A BERT cross-encoder handed the Qwen3 chat template ranks worse than not
    reranking at all -- measured, and recorded above `TEMPLATES`.

    The filename decides when it names a family, because that is what the
    measurement was taken against. When it does not, the architecture the file
    declares is a far better answer than falling straight through to `qwen3`.
    """
    from dyprys.rerank import template_for

    assert template_for(tmp_path / "bge-reranker-v2.gguf") == "bge"
    assert template_for(tmp_path / "jina-reranker-v2.gguf") == "bge"
    # Says nothing recognisable, and the file says "bert".
    assert template_for(tmp_path / "cross-encoder-q8.gguf", "bert") == "bge"
    assert template_for(tmp_path / "cross-encoder-q8.gguf", "jina-bert-v2") == "bge"
    assert template_for(tmp_path / "cross-encoder-q8.gguf", "qwen3") == "qwen3"
    assert template_for(tmp_path / "cross-encoder-q8.gguf") == "qwen3"
