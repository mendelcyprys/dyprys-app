"""Many libraries, one installation.

A library is a directory holding an index; nothing stops you keeping several,
and once backups start moving between people you will. This maps a short name
to a path so `-L neuro` replaces remembering where the directory went, and
records which one to use when no name is given.

The registry holds names and paths only. It never holds an index, so losing it
loses nothing — every library is still openable by `--data DIR`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


def config_home() -> Path:
    """Where the registry lives: $DYPRYS_HOME, else the XDG config directory."""
    explicit = os.environ.get("DYPRYS_HOME")
    if explicit:
        return Path(explicit)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return Path(xdg or Path.home() / ".config") / "dyprys"


def registry_path() -> Path:
    return config_home() / "libraries.json"


@dataclass(frozen=True)
class Library:
    name: str
    path: Path
    is_default: bool = False

    @property
    def exists(self) -> bool:
        # By content, not by filename: an index is one of ours because of the
        # tables in it. Asking for `dyprys.sqlite` by name reported a renamed
        # or forked index as absent, so `dyp library list` printed "no index
        # yet" beside a library every other command opened and read.
        from dyprys import db

        return db.index_exists(self.path)


# Where an unreadable registry is kept when a write would otherwise erase it.
UNREADABLE_SUFFIX = ".unreadable"


def _empty() -> dict:
    return {"libraries": {}, "default": None}


def _read() -> dict | None:
    """The registry as stored, or None when it is there but cannot be used.

    Valid JSON of the wrong shape counts as unusable: a hand-edited `{}` or `[]`
    would otherwise raise KeyError or TypeError out of whichever command
    happened to touch it first.
    """
    path = registry_path()
    if not path.exists():
        return _empty()
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict) or not isinstance(state.get("libraries"), dict):
        return None
    state.setdefault("default", None)
    return state


def readable() -> bool:
    """False when a registry file exists that cannot be parsed.

    Worth asking before reporting that nothing is registered, which is what an
    unreadable file otherwise looks like.
    """
    return _read() is not None


def _load() -> dict:
    """A usable registry, empty when the file cannot be read.

    Reading fails soft on purpose: a corrupt registry costs you the *names*, and
    every library is still openable by `--data DIR`. Writing is where the care
    goes -- see `_save`, which will not overwrite what it could not read.
    """
    state = _read()
    return _empty() if state is None else state


def _preserve_unreadable() -> Path | None:
    """Move an unusable registry aside so that writing cannot erase it.

    Half of a truncated file is usually repairable by eye; nothing at all is
    not. Numbered rather than overwritten, so preserving twice does not undo the
    first rescue.
    """
    path = registry_path()
    if not path.exists() or _read() is not None:
        return None
    kept = path.with_name(path.name + UNREADABLE_SUFFIX)
    attempt = 1
    while kept.exists():
        attempt += 1
        kept = path.with_name(f"{path.name}{UNREADABLE_SUFFIX}.{attempt}")
    path.replace(kept)
    return kept


def _save(state: dict) -> None:
    """Replace the registry atomically, and never over a file we could not read.

    `write_text` truncates before it writes, so a crash or a full disk leaves a
    half-written file -- which `_load` then reads as an empty registry, and the
    very next `add` writes over it, losing every other name. Two defects that
    compose into losing the lot. Writing a neighbouring temp file and renaming
    makes the swap atomic, so a reader sees the old registry or the new one and
    never a torn one; and an unreadable file is set aside first rather than
    overwritten.
    """
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _preserve_unreadable()

    # Same directory, so os.replace is a rename within one filesystem.
    handle = tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name + ".", suffix=".tmp",
        delete=False, encoding="utf-8",
    )
    try:
        with handle:
            handle.write(json.dumps(state, indent=1) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def libraries() -> list[Library]:
    state = _load()
    return [
        Library(name, Path(path), name == state.get("default"))
        for name, path in sorted(state["libraries"].items())
    ]


def add(name: str, path: Path, make_default: bool | None = None) -> Library:
    """Register `name`.  The first library registered becomes the default."""
    state = _load()
    first = not state["libraries"]
    state["libraries"][name] = str(Path(path).resolve())
    if make_default or (make_default is None and first):
        state["default"] = name
    _save(state)
    return Library(name, Path(state["libraries"][name]), state["default"] == name)


def remove(name: str) -> bool:
    """Forget a name.  The library's files are not touched."""
    state = _load()
    if name not in state["libraries"]:
        return False
    del state["libraries"][name]
    if state.get("default") == name:
        state["default"] = next(iter(sorted(state["libraries"])), None)
    _save(state)
    return True


@dataclass(frozen=True)
class Contents:
    """What a library holds, read without touching it."""

    books: int
    chunks: int
    models: list[tuple[str, int, int]]   # (name, chunks embedded, chunks it could)

    @property
    def coverage(self) -> float:
        """How far the best-covered model has got, against its own chunking.

        Not against the library. A model embeds one chunking, so a library split
        two ways would show every model permanently short of complete if this
        divided by every chunk in it.
        """
        best = max(((done, whole) for _n, done, whole in self.models),
                   key=lambda p: p[0] / p[1] if p[1] else 0.0, default=(0, 0))
        return best[0] / best[1] if best[1] else 0.0


def summarise(path: Path) -> Contents | None:
    """Books, chunks and per-model coverage for a library on disk.

    Opened **read-only**. `db.connect` runs the schema script and any pending
    migration, which is the right thing when you mean to use an index and the
    wrong thing when you are only listing what is there -- a summary must not
    write to five libraries as a side effect of naming them.
    """
    import sqlite3 as _sqlite3

    from dyprys import db

    # Resolved the same way every other command resolves it -- by the tables in
    # the file rather than its name -- so a listing agrees with what opening it
    # would find. db_path raises when a directory holds two indexes; a listing
    # says nothing about that one rather than failing the whole listing.
    try:
        database = db.db_path(Path(path))
    except ValueError:
        return None
    if not database.exists():
        return None
    try:
        conn = _sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    except _sqlite3.Error:
        return None
    try:
        books = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        chunks = conn.execute(
            "SELECT COALESCE(SUM(chunk_count), 0) FROM segments").fetchone()[0]
        models = [
            (name, done, whole or chunks)
            for name, done, whole in conn.execute(
                "SELECT m.name, COALESCE(SUM(p.n_embedded), 0), "
                "  (SELECT COALESCE(SUM(seg.chunk_count), 0) FROM segments seg "
                "   WHERE m.chunking_id IS NULL OR seg.chunking_id = m.chunking_id) "
                "FROM models m LEFT JOIN segment_progress p ON p.model_id = m.id "
                "GROUP BY m.id ORDER BY m.id")
        ]
        return Contents(books, chunks, models)
    except _sqlite3.Error:
        return None            # a half-written or foreign database is not fatal
    finally:
        conn.close()


def contents(path: Path) -> tuple[int, int]:
    """(files, bytes) of an index directory — what dropping it would remove."""
    files = [p for p in Path(path).glob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def use(name: str) -> bool:
    state = _load()
    if name not in state["libraries"]:
        return False
    state["default"] = name
    _save(state)
    return True


def resolve(name: str | None) -> Path | None:
    """The path for `name`, or for the default when `name` is None."""
    state = _load()
    wanted = name or state.get("default")
    if not wanted:
        return None
    found = state["libraries"].get(wanted)
    return Path(found) if found else None
