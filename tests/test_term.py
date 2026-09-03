"""Emphasis in a terminal, and none anywhere else."""

import pytest

from dyprys import term


@pytest.fixture
def coloured(monkeypatch):
    monkeypatch.setattr(term, "COLOUR", True)


def test_nothing_is_styled_when_output_is_not_a_terminal(monkeypatch):
    """`dyp ask ... > out.txt` and `| grep` are ordinary; escape codes are not.

    A file full of escape sequences is corruption, not decoration.
    """
    monkeypatch.setattr(term, "COLOUR", False)

    assert term.bold("Title") == "Title"
    assert term.mark("the receptor here", ["receptor"]) == "the receptor here"
    assert "\033" not in term.rule()


def test_styling_wraps_and_closes(coloured):
    out = term.bold("Title")

    assert out.startswith("\033[1m") and out.endswith("\033[0m")
    assert "Title" in out


def test_no_color_is_honoured(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("CLICOLOR_FORCE", raising=False)

    assert term._enabled() is False


def test_clicolor_force_wins_over_a_pipe(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("CLICOLOR_FORCE", "1")

    assert term._enabled() is True


def test_common_words_are_not_marked():
    """Highlighting every `the` makes a passage harder to read, not easier."""
    assert term.terms("what is the role of the receptor in a neuron") == \
        sorted({"role", "receptor", "neuron"}, key=len, reverse=True)


def test_short_words_are_not_marked():
    """Two letters match inside longer words constantly, mid-word."""
    assert "on" not in term.terms("on my axon")


def test_a_stem_matches_its_plural(coloured):
    marked = term.mark("many receptors here", term.terms("receptor"))

    assert "\033" in marked
    assert "receptors" in marked.replace("\033[1;33m", "").replace("\033[0m", "")


def test_marking_lands_on_word_boundaries(coloured):
    """`conduction` must not light up because the query said `duct`."""
    marked = term.mark("saltatory conduction", ["duct"])

    assert marked == "saltatory conduction"


def test_overlapping_stems_do_not_nest_escape_codes(coloured):
    """One pass with a single alternation, so codes cannot strand inside each other."""
    marked = term.mark("neurotransmitter release", ["neuro", "neurotransmitter"])

    assert marked.count("\033[0m") == marked.count("\033[1;33m")


def test_marking_is_a_no_op_without_terms(coloured):
    assert term.mark("anything at all", []) == "anything at all"


# --- choosing which part of a passage to show ---------------------------------


def test_the_extract_is_taken_from_where_the_answer_is():
    """A chunk is 3,500 bytes; the opening is wherever the chunker happened to cut.

    The part worth showing is the part the query's words are in, which is as
    likely to be in the middle.
    """
    text = ("Some preamble about nothing much at all. More filler here. "
            "The myelin sheath is what makes saltatory conduction possible. "
            "And then some more unrelated closing text follows.")

    extract, before, after = term.best_extract(text, ["myelin", "saltatory"], budget=70)

    assert "saltatory conduction" in extract
    assert before is True


def test_a_passage_shorter_than_the_budget_is_returned_whole():
    text = "Short enough already."

    assert term.best_extract(text, ["short"], budget=700) == (text, False, False)


def test_the_extract_begins_and_ends_on_a_sentence():
    text = ("Alpha one here. Beta two here. The receptor binds its ligand. "
            "Delta four here. Epsilon five here.")

    extract, _, _ = term.best_extract(text, ["receptor"], budget=40)

    assert extract.startswith("The receptor")
    assert extract.rstrip().endswith(".")


def test_with_no_query_words_it_shows_the_opening():
    """Nothing in the passage is more relevant than anything else, so say so."""
    text = "First sentence. " * 80

    extract, before, after = term.best_extract(text, ["absent"], budget=100)

    assert extract.startswith("First sentence")
    assert before is False and after is True


def test_it_reports_whether_it_cut_either_end():
    text = "Aaa one. Bbb two. TARGET three. Ddd four. Eee five."

    _, before, after = term.best_extract(text, ["target"], budget=20)

    assert before is True and after is True


def test_empty_text_is_handled():
    assert term.best_extract("", ["anything"]) == ("", False, False)


# --- progress bars ------------------------------------------------------------


def test_a_bar_is_proportional():
    assert term.bar(0, 100, width=10).rstrip("·") == ""
    assert term.bar(100, 100, width=10) == "█" * 10
    assert term.bar(50, 100, width=10).count("█") == 5


def test_a_bar_moves_between_whole_cells():
    """Eighth-width blocks, so it advances on most updates rather than sticking."""
    a = term.bar(1, 100, width=10)
    b = term.bar(3, 100, width=10)

    assert a != b, "a bar that only moves per whole cell looks frozen"


def test_a_bar_is_plain_text_without_colour(monkeypatch):
    monkeypatch.setattr(term, "COLOUR", False)

    drawn = term.bar(30, 100, width=10)

    assert "\033" not in drawn
    assert len(drawn) == 10


def test_a_bar_handles_no_work_and_overshoot():
    assert len(term.bar(0, 0, width=8)) == 8
    assert term.bar(200, 100, width=8) == "█" * 8


def test_a_long_title_is_cut_in_the_middle():
    """Titles differ at the end; trimming the tail hides what tells them apart."""
    a = term.elide("Springer.Neuroscience.A.Mathematical.Primer.2002.eBook", 34)
    b = term.elide("Springer.Neuroscience.A.Mathematical.Primer.2009.eBook", 34)

    assert len(a) == 34 and len(b) == 34
    assert a != b, "two books that differ only at the end elided identically"
    assert a.startswith("Springer.Neuro") and a.endswith("2002.eBook")


def test_a_short_title_is_untouched():
    assert term.elide("pg553", 34) == "pg553"


def test_elide_copes_with_absurd_widths():
    assert len(term.elide("a long title here", 3)) == 3
    assert len(term.elide("a long title here", 6)) == 6
