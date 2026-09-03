"""Wrapping an index up so it can be moved, kept, or restored elsewhere.

An index is three things that must travel together: the database, the vector
files, and the source text the offsets point into. Any two without the third is
not a library — vectors without sources can rank but cannot show a passage, and
sources without vectors are hundreds of hours from being searchable again.

The archive is gzipped, which is not merely tidiness. Vector files are sized to
the whole library and mostly unwritten until embedding finishes, and those
regions are zeros: 8 MB of untouched rows compress to 7.8 KB. Source text
compresses about 3x and is the bulk of a real library. The vectors themselves
barely compress at all, which is what one would expect of them.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from dyprys import db

MANIFEST = "manifest.json"
FORMAT = 1


@dataclass
class BackupReport:
    path: Path
    bytes_written: int
    books: int
    chunks: int
    models: int
    sources: int


def source_prefix(conn: sqlite3.Connection) -> str | None:
    """The deepest directory containing every source, or None if there are none."""
    paths = [row["path"] for row in conn.execute("SELECT path FROM sources")]
    if not paths:
        return None
    if len(paths) == 1:
        return str(Path(paths[0]).parent)
    return os.path.commonpath(paths)


def write(
    conn: sqlite3.Connection,
    directory: Path,
    out: Path,
    include_sources: bool = True,
    progress=None,
) -> BackupReport:
    """Archive the index, and optionally the text it points at.

    The database is copied through SQLite's online backup rather than read off
    disk, so a backup taken while `dyp embed` is running still captures a
    consistent snapshot. The vectors it captures may run *ahead* of that
    snapshot's progress counts, which is harmless: the extra rows are simply
    re-embedded on the next run, exactly as after any interruption.
    """
    directory = Path(directory)
    out = Path(out)
    counts = conn.execute(
        "SELECT (SELECT COUNT(*) FROM books), (SELECT COUNT(*) FROM chunks), "
        "       (SELECT COUNT(*) FROM models), (SELECT COUNT(*) FROM sources)"
    ).fetchone()
    prefix = source_prefix(conn)
    sources = [row["path"] for row in conn.execute("SELECT path FROM sources ORDER BY path")]

    manifest = {
        "format": FORMAT,
        "schema_version": int(db.get_meta(conn, "schema_version") or 0),
        "created_at": db.now(),
        "source_prefix": prefix,
        "includes_sources": bool(include_sources and prefix),
        "books": counts[0],
        "chunks": counts[1],
        "models": counts[2],
        "sources": counts[3],
        # In the manifest as well as the database, so someone can read what
        # weights they need straight out of the archive without restoring it.
        "weights": db.model_provenance(conn),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        snapshot = Path(staging) / db.DB_FILENAME
        target = sqlite3.connect(snapshot)
        with target:
            conn.backup(target)      # consistent even mid-write
        target.close()

        done = 0
        with tarfile.open(out, "w:gz") as archive:
            info = tarfile.TarInfo(MANIFEST)
            payload = json.dumps(manifest, indent=1).encode()
            info.size = len(payload)
            info.mtime = int(time.time())
            archive.addfile(info, io.BytesIO(payload))

            archive.add(snapshot, arcname=f"index/{db.DB_FILENAME}")
            for name in sorted(p.name for p in directory.glob("*.f32")):
                archive.add(directory / name, arcname=f"index/{name}")

            if manifest["includes_sources"]:
                for path in sources:
                    relative = os.path.relpath(path, prefix)
                    if Path(path).exists():
                        archive.add(path, arcname=f"sources/{relative}")
                    done += 1
                    if progress:
                        progress(done, len(sources))

    return BackupReport(out, out.stat().st_size, counts[0], counts[1], counts[2], counts[3])


@dataclass
class RestoreReport:
    data_dir: Path
    library: Path | None
    books: int
    chunks: int
    relocated: int
    missing: list[str]


def read_manifest(archive: Path) -> dict:
    with tarfile.open(archive, "r:*") as tar:
        try:
            member = tar.extractfile(MANIFEST)
        except KeyError:
            member = None
        if member is None:
            raise ValueError("not a dyprys backup: no manifest")
        return json.loads(member.read())


def restore(
    archive: Path,
    data_dir: Path,
    library: Path | None = None,
    progress=None,
) -> RestoreReport:
    """Unpack an index, put its sources somewhere, and point it at them.

    Refuses to write into a directory that already holds an index: a restore
    that silently merged with what was there would be unrecoverable, and the
    caller can always choose an empty directory.
    """
    from dyprys.library import relocate, unreadable_sources

    archive, data_dir = Path(archive), Path(data_dir)
    manifest = read_manifest(archive)
    if manifest.get("format") != FORMAT:
        raise ValueError(f"backup format {manifest.get('format')}, this build reads {FORMAT}")
    if (data_dir / db.DB_FILENAME).exists():
        raise ValueError(f"{data_dir} already holds an index; restore into an empty directory")

    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        index = [m for m in members if m.name.startswith("index/")]
        sources = [m for m in members if m.name.startswith("sources/")]

        for member in index:
            member.name = member.name[len("index/") :]
        # filter="data" refuses absolute paths and traversal outside the target.
        tar.extractall(data_dir, members=index, filter="data")

        if sources and library:
            library = Path(library)
            library.mkdir(parents=True, exist_ok=True)
            for done, member in enumerate(sources, 1):
                member.name = member.name[len("sources/") :]
                tar.extract(member, library, filter="data")
                if progress:
                    progress(done, len(sources))

    conn = db.connect(data_dir)
    try:
        relocated = 0
        if library and manifest.get("source_prefix"):
            _, relocated = relocate(conn, manifest["source_prefix"], str(Path(library).resolve()))
        counts = conn.execute(
            "SELECT (SELECT COUNT(*) FROM books), (SELECT COUNT(*) FROM chunks)"
        ).fetchone()
        return RestoreReport(
            data_dir, Path(library) if library else None,
            counts[0], counts[1], relocated, unreadable_sources(conn),
        )
    finally:
        conn.close()
