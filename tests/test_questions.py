"""The question sets are data, and data goes wrong quietly.

`eval/README.md` has always claimed two rules were checked -- no question
contains its own answer, and each stands alone -- and nothing checked them. A
question set is the one artefact in this project that cannot be verified by
running it: a broken question does not raise, it just scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SETS = sorted((Path(__file__).parent.parent / "eval").glob("questions*.json"))
KINDS = {"lookup", "descriptive", "mechanism", "oblique"}


def load(path):
    rows = json.loads(path.read_text())
    out = []
    for row in rows:
        if isinstance(row, dict):
            out.append((row["q"], row["a"], row.get("kind", "lookup")))
        else:  # the older ["question", "answer"] pair format still loads
            out.append((row[0], row[1], "lookup"))
    return out


def test_there_are_question_sets():
    assert SETS, "no eval/questions*.json found"


@pytest.mark.parametrize("path", SETS, ids=lambda p: p.stem)
def test_no_question_contains_its_own_answer(path):
    """Otherwise BM25 answers it for free and the question measures nothing."""
    bad = [(q, a) for q, a, _ in load(path) if a.lower() in q.lower()]
    assert not bad, f"{path.name}: questions containing their answer: {bad}"


@pytest.mark.parametrize("path", SETS, ids=lambda p: p.stem)
def test_no_duplicate_questions(path):
    rows = load(path)
    seen, dupes = set(), []
    for q, _, _ in rows:
        if q in seen:
            dupes.append(q)
        seen.add(q)
    assert not dupes, f"{path.name}: duplicated: {dupes}"


@pytest.mark.parametrize("path", SETS, ids=lambda p: p.stem)
def test_every_question_stands_alone(path):
    """A question that only makes sense after the previous one is broken.

    Search sees one question at a time, so a back-reference has no referent.
    """
    openers = ("and ", "also ", "what about", "the same ", "then ", "but ")
    bad = [q for q, _, _ in load(path)
           if q.lower().startswith(openers) or len(q.split()) < 4]
    assert not bad, f"{path.name}: not standalone: {bad}"


@pytest.mark.parametrize("path", SETS, ids=lambda p: p.stem)
def test_kinds_and_fields_are_well_formed(path):
    for q, a, kind in load(path):
        assert q.strip() and a.strip(), f"{path.name}: empty field in {(q, a)!r}"
        assert kind in KINDS, f"{path.name}: unknown kind {kind!r} on {q!r}"
        assert a == a.strip().lower() or a == a.strip(), f"{path.name}: {a!r} has stray space"


@pytest.mark.parametrize("path", SETS, ids=lambda p: p.stem)
def test_answers_use_the_corpus_spelling(path):
    """The corpus is American-spelled: fiber 2828 vs fibre 542, edema 379 vs 13.

    A term spelled the other way scores as a miss however good the passage is,
    which measures the speller rather than the retrieval.
    """
    british = ("fibre", "oedema", "ischaem", "haem", "paraesthes",
               "grey matter", "colour", "behaviour", "tumour", "anaemi")
    bad = [(q, a) for q, a, _ in load(path)
           if any(b in a.lower() for b in british)]
    assert not bad, f"{path.name}: British spelling in answer terms: {bad}"


def test_the_extended_set_contains_the_frozen_one():
    """questions.json is frozen: every recorded number in this project is "of 110".

    The extended set is a superset so the original subset can still be scored on
    its own, and the two remain comparable.
    """
    base = Path(__file__).parent.parent / "eval" / "questions.json"
    ext = base.parent / "questions-extended.json"
    if not ext.exists():
        pytest.skip("no extended set")
    have = {q for q, _, _ in load(ext)}
    missing = [q for q, _, _ in load(base) if q not in have]
    assert not missing, f"extended set dropped {len(missing)} original questions"
