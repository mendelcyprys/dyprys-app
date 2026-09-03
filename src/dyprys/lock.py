"""One writer per index.

Two embed runs against the same index do not corrupt it -- they would compute
identical vectors and write them to identical rows -- but they would burn hours
of GPU doing the same work twice and leave the progress counts racing each
other backwards. Cheaper to refuse.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
    import msvcrt

    def _take(handle):
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _release(handle):
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _take(handle):
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release(handle):
        fcntl.flock(handle, fcntl.LOCK_UN)


class AlreadyRunning(RuntimeError):
    pass


def holder(directory: Path, what: str = "embed") -> int | None:
    """The pid of the process holding this lock, or None if nobody holds it.

    The pid in the file is not evidence on its own -- it is left behind when the
    holder exits, because a released lock is not an erased file. The lock itself
    is the evidence: if taking it succeeds, nobody is there, and we release it
    again immediately.
    """
    path = Path(directory) / f".{what}.lock"
    if not path.exists():
        return None
    try:
        handle = open(path, "r+")
    except OSError:
        return None
    try:
        try:
            _take(handle)
        except OSError:
            handle.seek(0)
            text = handle.read().strip()
            return int(text) if text.isdigit() else -1   # held, pid unreadable
        _release(handle)
        return None
    finally:
        handle.close()


@contextmanager
def exclusive(directory: Path, what: str = "embed"):
    """Hold an advisory lock on `directory` for the duration of the block.

    The lock is a file lock, so it is released by the kernel if the process is
    killed -- a crashed run must never leave the index permanently unusable.
    Taken through fcntl on POSIX and msvcrt on Windows; both are advisory locks
    the kernel drops when the holder dies.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f".{what}.lock"
    handle = open(path, "w")
    try:
        _take(handle)
    except OSError:
        handle.close()
        raise AlreadyRunning(
            f"another `dyp {what}` is running against this index "
            f"(lock: {path}). Wait for it, or stop it with Ctrl-C."
        ) from None
    handle.write(str(os.getpid()))
    handle.flush()
    try:
        yield
    finally:
        _release(handle)
        handle.close()
