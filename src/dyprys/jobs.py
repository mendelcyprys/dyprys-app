"""The long mutations, run as detached subprocesses and followed from outside.

`ask` runs in-process because the warm model is the entire point. Everything
that *writes* runs as a real `dyp` subprocess, and the asymmetry is deliberate:

  * `lock.exclusive` is a per-process advisory lock, so one process may only
    embed one index at a time — and a server that embedded in-process could
    not answer a question while it did;
  * a run can last days. It must survive the API restarting, and it does,
    because nothing here holds a handle on it: progress is committed to the
    index per batch, so `state` is a *read of the index*, not a pipe;
  * spawning the real CLI means a web-started run and a terminal-started run
    are the same run. `dyp watch` in another window follows one this module
    started, and this module reports on one a terminal started.

Cancellation is safe because embedding is resumable. `stop` sends SIGINT, which
`dyp embed` handles by finishing the batch in flight and committing it; nothing
is lost and the next run picks up where it left off. That is why a UI may offer
a stop button at all.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dyprys import db, errors
from dyprys import embed as embed_mod
from dyprys import lock

# Every mutation the API is allowed to start. Anything not listed is not a job,
# which is the point of a list rather than a prefix check: `kind` arrives in a
# URL path, and it becomes an argv element.
KINDS = ("embed", "add", "route", "lexical", "compact")

# What each kind will accept, as {name: (flag, kind_of_value)}. An allowlist
# rather than a translation of whatever arrived, because these become arguments
# to a subprocess: an option this table does not name cannot reach argv at all,
# and a value is coerced to the declared type before it gets there.
#
# `bool` means a switch — present when true, absent when false, never a value.
FLAGS: dict[str, dict[str, tuple[str, type]]] = {
    "embed": {
        "model": ("--model", str), "limit": ("--limit", int),
        "batch": ("--batch", int), "int8": ("--int8", bool),
        "for": ("--for", str), "duty": ("--duty", int),
        "target": ("--target", int), "collection": ("--collection", str),
    },
    "add": {
        "chapters": ("--chapters", bool), "ext": ("--ext", str),
        "target": ("--target", int), "overlap": ("--overlap", int),
        "deep": ("--deep", bool),
    },
    "route": {"centroids": ("--centroids", int), "model": ("--model", str)},
    "lexical": {},
    "compact": {"yes": ("--yes", bool)},
}

# `compact` rewrites vector files and cannot be asked twice at once either, so
# every kind takes the same lock `embed` does unless it names its own.
LOCKS = {"embed": "embed", "add": "embed", "route": "embed",
         "lexical": "embed", "compact": "embed"}


@dataclass(frozen=True)
class JobRef:
    """A run that was started. The pid is the subprocess's, not the lock's.

    They differ for a moment: `dyp embed` loads its weights *before* it takes
    the lock, so a job is running and unlocked for the seconds that takes.
    `state` waits that gap out rather than reporting "nothing is running" for a
    run that is plainly starting.
    """

    kind: str
    pid: int
    log: Path
    started_at: str


def log_dir(directory) -> Path:
    return Path(directory) / ".jobs"


def _argv(directory, kind: str, options: dict | None) -> list[str]:
    """The command line, built only from names this module already knew."""
    if kind not in KINDS:
        raise errors.NoSuchJob(f"no such job {kind!r}; try one of {', '.join(KINDS)}")
    argv = [sys.executable, "-m", "dyprys.cli", "--data", str(directory), kind]
    # `add` is the one kind with a positional, and it is a path into the user's
    # own filesystem rather than a flag.
    paths = (options or {}).get("paths") or []
    if kind == "add":
        if not paths:
            raise errors.NoSuchBook("`add` needs at least one path to read text from")
        argv.extend(str(p) for p in paths)
    for name, (flag, want) in FLAGS[kind].items():
        value = (options or {}).get(name)
        if value is None or value is False:
            continue
        if want is bool:
            argv.append(flag)
        else:
            argv.extend((flag, str(want(value))))
    return argv


def start(directory, kind: str = "embed", options: dict | None = None) -> JobRef:
    """Spawn `dyp KIND` against this index, detached, logging to `.jobs/`.

    Refused if the lock is already held. The subprocess would refuse too — but
    only after spending a minute loading a GGUF to discover it, and a UI would
    show a job that starts and then dies for no visible reason.
    """
    directory = Path(directory)
    argv = _argv(directory, kind, options)
    held = lock.holder(directory, LOCKS.get(kind, "embed"))
    if held is not None:
        raise lock.AlreadyRunning(
            f"another `dyp {kind}` is running against this index (pid {held}). "
            f"Wait for it, or stop it.")

    logs = log_dir(directory)
    logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    log = logs / f"{kind}-{stamp}.log"
    handle = open(log, "ab")
    try:
        # `start_new_session` detaches it from this process group, so the run
        # outlives the server and a Ctrl-C in the server's terminal does not
        # reach it. Stopping it is `stop`, which is deliberate and graceful.
        child = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, cwd=str(directory),
                                 start_new_session=True)
    finally:
        handle.close()
    return JobRef(kind=kind, pid=child.pid, log=log, started_at=db.now())


# The previous reading of a run's progress, per (index, model), so a polling
# caller gets a rate from the gap between its own polls. `dyp watch` samples in
# a loop it owns; an HTTP handler cannot sleep to get a second sample, and
# guessing one would be worse than saying "not yet".
#
# Shared by the module, which is safe *only* because it holds observations
# rather than anything belonging to a request: two callers polling the same job
# see slightly different rates and neither sees the other's job.
_SAMPLES: dict[tuple[str, int], tuple[float, int]] = {}


def _rate(conn, directory, model_id: int, done: int) -> float | None:
    """Chunks per second, from this caller's previous poll or from history."""
    key = (str(directory), model_id)
    now = time.monotonic()
    before = _SAMPLES.get(key)
    _SAMPLES[key] = (now, done)
    if before is not None:
        elapsed, moved = now - before[0], done - before[1]
        if elapsed > 1 and moved > 0:
            return moved / elapsed
    # No usable second sample yet. The median of what this model has actually
    # managed on this machine is a better first answer than nothing, and it is
    # the same number `dyp check` estimates from.
    return db.observed_rate(conn, model_id)


def state(conn, directory, kind: str = "embed") -> dict:
    """How a run is going, read from the index. "Nothing" is a state.

    Deliberately not a 404 when idle: a UI polls this, and an error code for
    the ordinary case would make every poll look like a failure.
    """
    directory = Path(directory)
    pid = lock.holder(directory, LOCKS.get(kind, "embed"))
    log = latest_log(directory, kind)
    answer = {
        "kind": kind,
        "running": pid is not None,
        "pid": pid if (pid or 0) > 0 else None,
        "model": None, "done": None, "live": None, "share": None,
        "rate": None, "eta_seconds": None, "book_in_flight": None,
        "log": str(log) if log else None,
    }
    if kind != "embed":
        return answer

    following = embed_mod.model_in_progress(conn)
    if following is None:
        return answer
    model_id, model_name, chunking_id = following
    live, done, inflight = embed_mod.progress_of(conn, model_id, chunking_id)
    rate = _rate(conn, directory, model_id, done)
    answer.update(
        model=model_name, done=done, live=live,
        share=(done / live) if live else 0.0,
        rate=rate,
        eta_seconds=((live - done) / rate) if rate else None,
        book_in_flight=inflight["title"] if inflight else None,
    )
    return answer


def states(conn, directory) -> dict:
    """Every kind at once, which is what a dashboard actually asks for."""
    return {kind: state(conn, directory, kind) for kind in KINDS}


def latest_log(directory, kind: str) -> Path | None:
    logs = sorted(log_dir(directory).glob(f"{kind}-*.log"))
    return logs[-1] if logs else None


def log_tail(directory, kind: str = "embed", lines: int = 40) -> str:
    """The end of the most recent log for this kind, or "".

    Read from the end rather than by reading the file: an embed's log grows for
    days, and `dyp watch`'s in-place redraw writes a line per tick.
    """
    log = latest_log(directory, kind)
    if log is None:
        return ""
    try:
        size = log.stat().st_size
        with open(log, "rb") as handle:
            handle.seek(max(0, size - 64_000))
            raw = handle.read()
    except OSError:
        return ""
    text = raw.decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def stop(directory, kind: str = "embed", force: bool = False) -> dict:
    """Ask the run holding this index to stop, and say whether anyone heard.

    SIGINT, because `dyp embed` handles it by finishing the batch in flight and
    committing it. `force` escalates to SIGTERM for a second attempt; there is
    no third, because killing a run outright is safe here anyway -- the lock is
    a file lock the kernel drops, and an embed loses at most the batch it was
    part way through.
    """
    directory = Path(directory)
    pid = lock.holder(directory, LOCKS.get(kind, "embed"))
    if pid is None:
        return {"stopped": False, "pid": None, "why": "nothing is running"}
    if pid < 0:
        raise errors.JobUnreachable(
            "something holds this index and its pid could not be read; "
            "stop it from the terminal that started it")
    try:
        os.kill(pid, signal.SIGTERM if force else signal.SIGINT)
    except ProcessLookupError:
        return {"stopped": False, "pid": pid, "why": "that process is already gone"}
    except PermissionError as refused:
        raise errors.JobUnreachable(
            f"cannot signal pid {pid}: {refused}") from None
    return {"stopped": True, "pid": pid,
            "signal": "SIGTERM" if force else "SIGINT",
            # Said here rather than left implied: a caller deciding whether to
            # escalate needs to know the first signal is not instant.
            "note": "it finishes the batch in flight and commits before stopping"}
