"""What a slow search is doing, said once and rendered per frontend.

An expanded and summarised query takes ten seconds or more, and silence for ten
seconds is indistinguishable from a hang. `cli.Stages` solved that for a
terminal by writing to stderr, which is the right answer for a terminal and no
answer over HTTP -- a `print` in a request handler narrates to whoever started
the server, not to whoever asked.

So the pipeline emits `Event`s and something else decides where they go: the
terminal formatter that reproduces today's stderr exactly, a list for a
non-streaming JSON reply, a queue for an NDJSON stream, or nothing at all.

Note that `Stderr` writes to a stream rather than calling `print`. The
difference matters: a stream is an argument, so a test can capture it and a
second frontend can substitute one, whereas `print` is a decision about the
process. `tests/test_layering.py` holds that rule for every module below the
frontends, this one included.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from dyprys import term


@dataclass(frozen=True)
class Event:
    """One thing that happened, before anyone decided how to show it.

    `kind` is what happened ("routed", "expanded", "rescoring"); `label` and
    `detail` are the two halves `Stages` has always printed as `label: detail`;
    `indent` is nesting, one step per level.
    """

    kind: str
    label: str
    detail: str = ""
    indent: int = 0


class Progress:
    """The seam. Subclasses override `event` and inherit the two verbs.

    `note` and `working` are what the pipeline actually calls, and they are the
    two `cli.Stages` had -- kept, rather than replaced by `event` everywhere, so
    the moved code reads as a move.
    """

    def event(self, event: Event) -> None:
        raise NotImplementedError

    def note(self, label: str, detail: str = "", *, indent: int = 0,
             kind: str = "note") -> None:
        self.event(Event(kind, label, detail, indent))

    def working(self, message: str, *, kind: str = "working") -> None:
        """Announced before a slow stage, so the wait has a name."""
        self.event(Event(kind, message))


class Silent(Progress):
    """The default. A library that is not asked to narrate says nothing."""

    def event(self, event: Event) -> None:
        pass


class Collect(Progress):
    """Keeps the events, for a reply that is assembled before it is sent."""

    def __init__(self):
        self.events: list[Event] = []

    def event(self, event: Event) -> None:
        self.events.append(event)

    def as_json(self) -> list[dict]:
        return [{"kind": e.kind, "label": e.label, "detail": e.detail,
                 "indent": e.indent} for e in self.events]


class Stderr(Progress):
    """Today's terminal output, moved: dim label, plain detail, nothing erased.

    An earlier version wrote a transient line and erased it, which meant the
    interesting part -- the rephrasing the model chose, the books routing picked
    -- flashed past and was gone. Nothing is erased now: every line is something
    worth having kept. stderr, so `dyp ask ... > results.txt` still captures only
    results.

    The stream is resolved at write time rather than bound at construction, so
    a test that replaces `sys.stderr` after this object exists still captures it.
    """

    def __init__(self, on: bool = True, stream=None):
        self.on = bool(on)
        self._stream = stream

    def event(self, event: Event) -> None:
        if not self.on:
            return
        stream = self._stream if self._stream is not None else sys.stderr
        pad = "  " + "    " * event.indent
        if event.detail:
            line = f"{pad}{term.dim(event.label + ':')} {event.detail}"
        else:
            line = f"{pad}{term.dim(event.label)}"
        stream.write(line + "\n")
        stream.flush()


class Queue(Progress):
    """Pushes events onto an `asyncio.Queue` for a streaming response.

    The search itself runs on a worker thread -- one per library, because a
    sqlite connection belongs to the thread that opened it and `llama_cpp.Llama`
    is not re-entrant -- so every put has to cross back to the loop's thread.
    `call_soon_threadsafe` is the only safe way to do that, and passing the loop
    in makes the threading explicit rather than discovered.
    """

    def __init__(self, queue, loop=None):
        self.queue, self.loop = queue, loop

    def event(self, event: Event) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, event)
        else:
            self.queue.put_nowait(event)
