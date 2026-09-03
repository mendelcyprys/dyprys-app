"""Parsing what an expansion model actually said, not what it should have said."""

from dyprys.expand import MAX_LINES, Expansion, parse

# Real output, from the first two queries ever put to the model. Kept verbatim
# because both are malformed in ways that were not predicted: the first
# truncates a line mid-phrase, the second never closes its <think> block and
# then repeats itself.
REAL_CLEAN = """<think>
</think>

lex: understanding enzyme-linked receptors
lex: how do enzyme-linked
vec: understanding enzyme-linked receptors
vec: how do enzyme-linked receptors function
hyde: Enzyme linked receptors is an important concept that relates to role of \
enzyme-linked receptors in signaling."""

REAL_MALFORMED = """<think>
lex: ball vs sword
lex: what is better
vec: ball vs sword
vec: what is better for death: sword or ball
hyde: The wise woman asked whether the young laird would die by sword or by ball
lex: comparison of death by sword and ball
vec: comparison of death by sword and ball
hyde: The process of the wise woman's questioning involves several steps. \
First, what is better for death: sword or ball."""


def test_it_reads_the_three_kinds_apart():
    got = parse(REAL_CLEAN)

    assert "understanding enzyme-linked receptors" in got.lexical
    assert "how do enzyme-linked receptors function" in got.vector
    assert got.hyde and got.hyde[0].startswith("Enzyme linked receptors")


def test_a_truncated_line_is_dropped_as_a_prefix_of_a_whole_one():
    """Generation stops mid-phrase, and the fragment searches as its own query.

    "how do enzyme-linked" is not a shorter way of asking the question, it is
    half of one — and as a BM25 phrase it matches things the full line would
    not. Prefix containment catches it without guessing at where truncation
    happened.
    """
    got = parse(REAL_CLEAN)

    assert "how do enzyme-linked" not in got.vector
    assert "how do enzyme-linked receptors function" in got.vector


def test_an_unclosed_think_block_does_not_swallow_the_output():
    """The tag is stripped, not the rest of the generation.

    Treating <think> as opening a region to discard loses everything when the
    model forgets to close it — which it did on the second query ever asked.
    """
    got = parse(REAL_MALFORMED)

    assert got.lexical, "an unclosed <think> hid every line"
    assert "ball vs sword" in got.lexical


def test_a_closed_think_block_is_discarded():
    """Reasoning is not an expansion, even when it is phrased like one."""
    got = parse("<think>\nlex: a stray thought about the question\n</think>\nlex: the real one")

    assert got.lexical == ["the real one"]


def test_repeated_lines_are_kept_once():
    """The same phrase under one kind twice adds a search and no information."""
    got = parse("vec: alpha beta\nvec: alpha beta\nvec: gamma delta")

    assert got.vector == ["alpha beta", "gamma delta"]


def test_the_same_phrase_under_two_kinds_is_kept_in_both():
    """Deliberate: one goes to BM25 and one is embedded. They are not duplicates."""
    got = parse("lex: alpha beta\nvec: alpha beta")

    assert got.lexical == ["alpha beta"] and got.vector == ["alpha beta"]


def test_single_words_are_not_a_phrasing_of_a_question():
    got = parse("vec: receptors\nvec: enzyme linked receptors")

    assert got.vector == ["enzyme linked receptors"]


def test_lines_are_capped_per_kind():
    """Each accepted line is another whole search, so this is a latency budget."""
    text = "\n".join(f"vec: phrase number {n} here" for n in range(10))

    assert len(parse(text).vector) == MAX_LINES


def test_prose_around_the_lines_is_ignored():
    """A model that starts explaining itself must not break the query."""
    got = parse("Here are some rewrites:\n\nlex: alpha beta\n\nHope that helps!")

    assert got.lexical == ["alpha beta"]


def test_nothing_usable_is_an_empty_expansion_not_an_error():
    """Expansion is optional; a bad generation costs the improvement, not the query."""
    for text in ("", "I'm sorry, I can't help with that.", "<think>", None):
        got = parse(text or "")
        assert got.empty
        assert got.searches == 0


def test_a_colon_inside_a_line_survives():
    """`what is better for death: sword or ball` is one line, not a kind marker."""
    got = parse("vec: what is better for death: sword or ball")

    assert got.vector == ["what is better for death: sword or ball"]


def test_searches_counts_what_it_will_cost():
    got = Expansion(lexical=["a b"], vector=["c d", "e f"], hyde=["g h"])

    assert got.searches == 4
    assert not got.empty


# --- the search path, against a stub expander --------------------------------


class StubExpander:
    """Returns a fixed expansion, and records what it was asked to expand."""

    def __init__(self, expansion):
        self.expansion, self.asked = expansion, []

    def expand(self, query):
        self.asked.append(query)
        return self.expansion


def _run(conn, tmp_path, library, embedder, expander, mode="hybrid"):
    from dyprys import db as _db
    from dyprys.cli import _searcher
    from dyprys.embed import store_for
    from dyprys.lexical import backfill

    model = _db.model_id(conn, "stub", embedder.dim)
    store = store_for(conn, tmp_path / "ix", model, embedder.dim)
    with conn:
        for seg in conn.execute("SELECT id, chunk_count FROM segments").fetchall():
            _db.set_embedded_prefix(conn, model, seg["id"], seg["chunk_count"])
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    for cid in range(1, total + 1):
        store.write(cid, embedder.embed_query(f"chunk {cid}"))
    store.flush()
    backfill(conn)
    return _searcher(conn, store, embedder, model, mode, None,
                     expander=expander)("about neurons", 5)


def test_the_question_expanded_is_the_question_asked(conn, tmp_path, library, embedder):
    """Expansion sees the original, not something already rewritten."""
    expander = StubExpander(Expansion(vector=["something else entirely"]))

    _run(conn, tmp_path, library, embedder, expander)

    assert expander.asked == ["about neurons"]


def test_an_expansion_that_repeats_the_question_cannot_reorder_anything(
        conn, tmp_path, library, embedder):
    """The original leads its half, so a duplicate of it is a no-op.

    This is the check that the original ranking is passed to fusion *first* and
    is not merely one voice among several: fusing a list with an identical copy
    of itself must return that same order.
    """
    plain = _run(conn, tmp_path, library, embedder, None)
    echoed = _run(conn, tmp_path, library, embedder,
                  StubExpander(Expansion(vector=["about neurons"])))

    assert [h.chunk_id for h in echoed] == [h.chunk_id for h in plain]


def test_a_hypothetical_answer_is_embedded_as_a_document_not_a_query(
        conn, tmp_path, library, embedder):
    """`hyde` lines are passages. EmbeddingGemma prefixes the two differently.

    Embedding a hypothetical answer with the query prefix would ask the model to
    place a passage in query space, which is precisely the mismatch the two
    prompts exist to avoid.
    """
    expander = StubExpander(Expansion(hyde=["Neurons conduct signals along axons."]))

    _run(conn, tmp_path, library, embedder, expander)

    # The stub records only what reaches embed_documents, so its presence there
    # *is* the proof that the document path was taken.
    assert "Neurons conduct signals along axons." in embedder.seen


def test_an_empty_expansion_changes_nothing(conn, tmp_path, library, embedder):
    """A model that says nothing usable costs the improvement, not the query."""
    plain = _run(conn, tmp_path, library, embedder, None)
    empty = _run(conn, tmp_path, library, embedder, StubExpander(Expansion()))

    assert [h.chunk_id for h in plain] == [h.chunk_id for h in empty]


def test_expansion_is_off_unless_an_expander_is_given(conn, tmp_path, library, embedder):
    """The default path must not construct one, let alone call it."""
    from dyprys.cli import _load_expander

    class Args:
        expand = False
        expander = "/nonexistent/model.gguf"

    assert _load_expander(Args()) == (None, None)


def test_the_instructions_echoed_back_are_not_an_expansion():
    """A reasoning model restated the request as its answer, and it parsed.

    That is worse than output which fails to parse: three extra searches for the
    instruction text itself, silently polluting the fusion with rankings for
    "two to six words likely to appear verbatim in the passage". Seen from
    qwen3:4b, which reasons rather than answers.
    """
    from dyprys.expand import INSTRUCTION

    echoed = "\n".join(
        line for line in INSTRUCTION.splitlines()
        if line.startswith(("lex:", "vec:", "hyde:"))
    )
    assert echoed, "the instruction no longer contains the lines it asks for"

    assert parse(echoed).searches == 0


def test_a_real_expansion_is_not_mistaken_for_an_echo():
    """The echo guard must not reject an answer for sharing a word or two."""
    got = parse("lex: wise woman, young laird, sword, ball\n"
                "vec: the death of the young nobleman by blade or shot")

    assert got.searches == 2


def test_an_unreachable_expander_is_reported_not_skipped(monkeypatch):
    """`--expand` is an explicit request; silently ignoring it is the bug.

    A search that skips expansion looks exactly like a normal search — same
    results, same exit code, 468 ms instead of 5 s — so the user has no way to
    learn that the thing they asked for did not happen.
    """
    from dyprys.cli import _load_expander
    from dyprys.expand import OllamaExpander

    monkeypatch.setattr(OllamaExpander, "unavailable",
                        lambda self: "cannot reach ollama at http://localhost:11434")

    class Args:
        expand = True
        expander = "gemma3:4b"

    expander, problem = _load_expander(Args())
    assert expander is None
    assert "ollama" in problem


def test_a_reachable_expander_loads(monkeypatch):
    from dyprys.cli import _load_expander
    from dyprys.expand import OllamaExpander

    monkeypatch.setattr(OllamaExpander, "unavailable", lambda self: None)

    class Args:
        expand = True
        expander = "gemma3:4b"

    expander, problem = _load_expander(Args())
    assert problem is None and expander.model == "gemma3:4b"


def test_the_two_backends_send_the_same_prompt():
    """Which prompt a model needs is a property of its weights, not its backend.

    An earlier version decided it by backend: the direct-GGUF path sent the bare
    question and hardcoded ChatML, while the ollama path sent the instruction
    and used the model's own template. The same weights therefore expanded
    differently depending on how they were reached, and only one of those two
    ways was ever measured.
    """
    from dyprys.expand import PROMPTS, Expander, OllamaExpander

    for name in PROMPTS:
        direct = Expander.__new__(Expander)
        direct.instruction, direct.instruct = PROMPTS[name], True
        served = OllamaExpander("stub", prompt=name)
        assert direct.instruction is served.instruction, f"{name} differs by backend"


def test_the_direct_path_uses_the_models_own_chat_template():
    """Hardcoding one family's template is the failure that looks like success.

    A model that never sees its own template continues the prompt instead of
    answering it — the same thing `/api/generate` did, which produced output
    that parsed into three clean lines and was worth nothing.
    """
    import inspect

    from dyprys.expand import Expander

    source = inspect.getsource(Expander.expand)
    assert "create_chat_completion" in source
    assert "<|im_start|>" not in source
