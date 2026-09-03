"""Checking an answer against its sources, without needing a model to make one."""

from dataclasses import dataclass

from dyprys.summarise import (
    NO_ANSWER,
    Answer,
    answer_from,
    find_claims,
    normalise,
    render,
    verify,
)


@dataclass
class FakePassage:
    """Enough of `search.Passage` to verify against."""

    chunk_id: int
    title: str
    text: str | None
    offset: int = 0


MYELIN = FakePassage(
    101, "Principles of Neural Science",
    "The action potential in a myelinated axon is said to move by saltatory "
    "conduction, from the Latin saltare, to jump. Because the lack of myelin "
    "slows conduction, demyelinating diseases are devastating.")
SCHWANN = FakePassage(
    202, "Neuroscience, Third Edition",
    "Myelinating Schwann cells surround large-diameter axons to furnish the "
    "myelin sheaths that insulate specific axonal populations in the PNS.")
PASSAGES = [MYELIN, SCHWANN]


def test_a_quote_that_is_really_there_verifies():
    text = 'Conduction jumps: "move by saltatory conduction" [1].'

    got = answer_from(text, PASSAGES)

    assert len(got.verified) == 1
    assert got.trustworthy


def test_a_quote_attributed_to_the_wrong_passage_is_caught():
    """The real failure seen in the first probe, not a hypothetical one.

    The model quoted a sentence that exists — in a different passage from the
    one it cited. Every word is genuine, so no check on the *words* finds it;
    only checking the words against the cited source does.
    """
    text = 'Schwann cells: "move by saltatory conduction" [2].'

    got = answer_from(text, PASSAGES)

    assert got.rejected and not got.verified
    assert not got.trustworthy


def test_an_invented_quote_is_caught():
    text = '"myelin doubles the speed of every axon" [1].'

    assert not answer_from(text, PASSAGES).verified


def test_a_rejected_quote_is_removed_from_the_prose():
    """The sentence is what the quote was offered as evidence for.

    Leaving the prose and dropping only the citation would present an
    unsupported claim as though it had a source.
    """
    text = 'Myelin is vital: "myelin doubles the speed of every axon" [1].'

    got = answer_from(text, PASSAGES)

    assert "doubles the speed" not in got.prose
    assert "unverifiable quotation removed" in got.prose


def test_typography_is_not_a_mismatch():
    """Models re-type curly quotes and en-dashes while copying faithfully.

    Normalising both sides keeps the check about words. Refusing these would
    reject correct quotations and teach the reader to ignore the warning.
    """
    text = '“move by saltatory conduction” [1].'

    assert answer_from(text, PASSAGES).verified


def test_pdf_ligatures_survive_the_round_trip():
    """Extracted text carries ﬁ and ﬂ; a model retypes them as fi and fl."""
    passage = FakePassage(1, "b", "the ﬁnal conﬂict was decisive")

    assert answer_from('"the final conflict was decisive" [1]', [passage]).verified


def test_an_apostrophe_does_not_open_a_quotation():
    """The bug in the first version of this regex.

    "Here's an answer" made everything after the apostrophe a quotation, so
    output whose quotes were all genuine reported zero verified.
    """
    text = "Here's an answer, based on the passages: \"move by saltatory conduction\" [1]."

    claims = find_claims(text)

    assert len(claims) == 1
    assert claims[0].quote == "move by saltatory conduction"


def test_a_citation_out_of_range_is_not_reinterpreted():
    """A model that invents passage 7 of 2 has said something about itself."""
    got = answer_from('"move by saltatory conduction" [7].', PASSAGES)

    assert got.rejected and got.rejected[0].chunk_id is None


def test_locations_come_from_the_index_not_the_model():
    """The model picks a passage; the index says where that passage is.

    Nothing in the citation is generated, so there is nothing in it to
    hallucinate.
    """
    got = answer_from('"myelin sheaths that insulate" [2].', PASSAGES)

    claim = got.verified[0]
    assert claim.chunk_id == 202
    assert claim.title == "Neuroscience, Third Edition"
    assert "chunk 202" in claim.location


def test_an_answer_with_no_quotes_is_not_trustworthy():
    """Fluent prose with nothing to check is the output to distrust most."""
    got = answer_from("Myelin speeds up conduction considerably.", PASSAGES)

    assert not got.trustworthy and not got.claims


def test_saying_there_is_no_answer_is_a_valid_outcome():
    """Without this a model always finds something, and 'nothing here' is useful."""
    got = answer_from(NO_ANSWER, PASSAGES)

    assert got.prose.strip() == NO_ANSWER
    assert not got.claims


def test_a_passage_that_could_not_be_read_verifies_nothing():
    """An unreadable source cannot support a quotation about it."""
    gone = FakePassage(9, "moved", None)

    assert not answer_from('"anything at all here" [1]', [gone]).verified


def test_rendering_numbers_passages_from_one():
    """Citation n is passages[n-1], and the model is shown exactly that."""
    block = render(PASSAGES)

    assert block.startswith("[1] The action potential")
    assert "[2] Myelinating Schwann cells" in block


def test_normalise_collapses_whitespace_not_words():
    assert normalise("a  b\n c") == "a b c"
    assert normalise("unchanged") == "unchanged"


def test_verify_is_pure():
    """It returns new claims rather than marking the ones it was given."""
    claims = find_claims('"move by saltatory conduction" [1]')

    checked = verify(claims, PASSAGES)

    assert claims[0].verified is False and checked[0].verified is True


def test_an_answer_holds_both_halves():
    got = Answer(prose="x", claims=verify(find_claims(
        '"move by saltatory conduction" [1] and "nonsense" [1]'), PASSAGES))

    assert len(got.verified) == 1 and len(got.rejected) == 1
    assert not got.trustworthy


def test_a_short_quote_is_checked_like_any_other():
    """It used to be skipped entirely, which is worse than rejecting it.

    A twelve-character floor meant `"saltatory" [1]` was neither verified nor
    rejected — it stayed in the prose with a citation beside it and nothing had
    looked at it. Length is not what makes the match precise; the citation is.
    """
    assert answer_from('"saltatory" [1]', PASSAGES).verified
    assert answer_from('"gibberish" [1]', PASSAGES).rejected


# --- citations without quotation marks ---------------------------------------


def test_a_citation_without_quotes_is_still_checked():
    """What the model actually does, and what the first version let through.

    Asked a real question it replied `neuropeptides, and neurosteroids ... [1]`
    — copied verbatim from passage 1, with no quotation marks. The parser found
    no `"..." [n]` pair, reported "nothing here has been checked", and left the
    sentence standing under its citation looking sourced.
    """
    text = "The action potential in a myelinated axon is said to move by " \
           "saltatory conduction [1]."

    got = answer_from(text, PASSAGES)

    assert got.claims, "a bare citation was not treated as a claim"
    assert got.verified, "verbatim text should verify even unquoted"
    assert got.verified[0].quoted is False


def test_a_bare_citation_that_is_a_paraphrase_is_marked_not_deleted():
    """It may be a fair reading; it is not a quotation, and must not look like one."""
    text = "Myelin makes signals travel faster [1]."

    got = answer_from(text, PASSAGES)

    assert got.rejected and not got.verified
    assert "paraphrase, not in the source" in got.prose
    assert "Myelin makes signals travel faster" in got.prose, "a reading was deleted"


def test_a_bare_citation_reaches_back_only_to_the_previous_sentence():
    """Otherwise one citation claims every sentence before it."""
    text = ("Neurons are complex. Myelinating Schwann cells surround "
            "large-diameter axons [2].")

    got = answer_from(text, PASSAGES)

    assert len(got.claims) == 1
    assert "Neurons are complex" not in got.claims[0].quote


def test_a_quoted_claim_is_not_also_counted_as_a_bare_one():
    text = '"move by saltatory conduction" [1].'

    got = answer_from(text, PASSAGES)

    assert len(got.claims) == 1 and got.claims[0].quoted is True


# --- asking again when the search found nothing -------------------------------


def test_a_rephrasing_is_taken_from_a_chatty_reply():
    """Told "no preamble", a model supplies one anyway."""
    from dyprys.summarise import rephrase

    said = ("Sure! Here's a different way to ask it:\n\n"
            "Describe the glymphatic system's role in cerebral waste clearance.\n")

    assert rephrase("x", lambda _p: said) == \
        "Describe the glymphatic system's role in cerebral waste clearance."


def test_a_rephrasing_that_is_only_preamble_yields_nothing():
    """Better to stop than to search the library for "Sure, here you go"."""
    from dyprys.summarise import rephrase

    assert rephrase("x", lambda _p: "Sure! Here is one:") == ""
    assert rephrase("x", lambda _p: "") == ""


def test_a_rephrasing_is_a_question_not_an_essay():
    """A paragraph is not a query; the length bounds keep it usable."""
    from dyprys.summarise import rephrase

    assert rephrase("x", lambda _p: "two words") == ""
    assert rephrase("x", lambda _p: " ".join(["word"] * 60)) == ""


def test_the_retry_is_the_refusal_path_only():
    """It costs nothing when the first search works, which is the point.

    NO_ANSWER is the whole trigger: a search that returned something is not
    retried however weak its answer, because a second opinion is not what the
    refusal signal means.
    """
    from dyprys.summarise import NO_ANSWER

    fine = answer_from('"move by saltatory conduction" [1]', PASSAGES)
    assert fine.prose.strip() != NO_ANSWER

    refused = answer_from(NO_ANSWER, PASSAGES)
    assert refused.prose.strip() == NO_ANSWER
