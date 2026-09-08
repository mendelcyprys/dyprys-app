"""The seam itself: what may print, exit, and read the environment.

A service layer is only worth extracting if it stays extracted. The two ways it
rots are both mechanical and both invisible in review of a large diff: a `print`
left in a moved function, and an `os.environ.get` that quietly re-couples a
"pure" module to the process it happens to be running in. The first turns a web
request into a line on the server's terminal; the second means the API answers
according to the shell that started it rather than the request that arrived.

This checks the rule with `ast` rather than by reading, so it keeps holding
after the phase that established it.

Note what is deliberately *not* forbidden. `term`, `registry` and `db.data_dir`
read the environment today -- `NO_COLOR`, `XDG_CONFIG_HOME`, `DYPRYS_DATA` --
and they are right to: those name where configuration and colour live, which is
a property of the machine, not of a request. The rule that matters is narrower
and is checked separately below: **no module below the frontends may read the
variables that choose a model**, because those are per-request in an API and
per-shell in a CLI, and a function that reads them itself can only serve one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src" / "dyprys"

# Everything that is not a frontend. `cli` and `api` are the two allowed to
# print, exit and consult the environment; every other module is called by both
# and may do none of it.
CORE = [
    "backup", "check", "chunker", "compact", "db", "embed", "embedder",
    "evaluate", "expand", "ingest", "jobs", "lexical", "library", "lock",
    "progress", "registry", "rerank", "routing", "search", "service",
    "summarise", "term", "text", "vectors",
]

# The variables that pick a model. In a server these arrive in the request body;
# in the CLI they come from the shell. Either way the resolution belongs to the
# frontend, and a core module that reads one can serve only one caller.
MODEL_ENV = ("DYPRYS_MODEL", "DYPRYS_RERANKER", "DYPRYS_EXPANDER", "DYPRYS_SUMMARISER")


def tree_of(module: str) -> ast.Module:
    path = SRC / f"{module}.py"
    if not path.exists():
        pytest.skip(f"{module}.py is not written yet")
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def lines_of(tree: ast.Module, predicate) -> list[int]:
    return sorted({node.lineno for node in ast.walk(tree) if predicate(node)})


@pytest.mark.parametrize("module", CORE)
def test_no_core_module_prints(module):
    """A print is an answer delivered to whoever owns the terminal.

    Over HTTP nobody does. Today `_router` reports unreachable books by printing
    -- CLAUDE.md calls that the one failure a reader cannot otherwise detect --
    so the move has to turn it into something a caller can carry, not drop it.
    """
    tree = tree_of(module)
    printed = lines_of(tree, lambda n: isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Name) and n.func.id == "print")
    assert not printed, f"{module}.py prints at line(s) {printed}"


@pytest.mark.parametrize("module", CORE)
def test_no_core_module_exits_the_process(module):
    """`_where` raises SystemExit today, which is why a server cannot call it.

    An exception that unwinds to the interpreter is a decision about the whole
    process. A library gets to say what went wrong; only a frontend gets to say
    what that costs.
    """
    tree = tree_of(module)

    def exits(node):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            return isinstance(node.exc.func, ast.Name) and node.exc.func.id == "SystemExit"
        return isinstance(node, ast.Name) and node.id == "SystemExit"

    where = lines_of(tree, exits)
    assert not where, f"{module}.py raises SystemExit at line(s) {where}"


@pytest.mark.parametrize("module", CORE)
def test_no_core_module_reads_argv(module):
    """argv belongs to the process. There is one, and two frontends."""
    tree = tree_of(module)
    where = lines_of(tree, lambda n: isinstance(n, ast.Attribute) and n.attr == "argv")
    assert not where, f"{module}.py reads sys.argv at line(s) {where}"


@pytest.mark.parametrize("module", CORE)
def test_no_core_module_resolves_a_model_from_the_environment(module):
    """Which model to use is an argument, not an ambient fact.

    `_weights_for` reads $DYPRYS_MODEL and `resolve_model` reads $DYPRYS_*
    today. Moved as they are, a server would answer every request with whatever
    model the shell that launched it named, and two libraries needing different
    models could not both be served -- while the CLI would keep working, so
    nothing would fail loudly enough to be noticed.
    """
    path = SRC / f"{module}.py"
    if not path.exists():
        pytest.skip(f"{module}.py is not written yet")
    source = path.read_text(encoding="utf-8")
    found = [name for name in MODEL_ENV if name in source]
    assert not found, (
        f"{module}.py names {found}; a core module takes the model as an "
        f"argument and lets its caller decide where that came from")


def test_the_frontends_are_the_only_place_a_typed_error_becomes_an_exit_code():
    """One handler each, so the two frontends cannot disagree about a failure.

    Scattered `print(...); return 2` is how the CLI says things now. Once the
    prose lives on the exception, a second `except DyprysError` anywhere below
    is a second opinion about what the error means.
    """
    errors = SRC / "errors.py"
    if not errors.exists():
        pytest.skip("errors.py is not written yet")
    for module in CORE:
        path = SRC / f"{module}.py"
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        caught = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.type is not None
            and "DyprysError" in ast.dump(node.type)
        ]
        assert not caught, (
            f"{module}.py catches DyprysError at line(s) {caught}; the "
            f"frontends translate errors, the core only raises them")
