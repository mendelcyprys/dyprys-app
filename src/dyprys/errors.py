"""Failures as values, so two frontends can disagree about what to do with them.

`cli.py` says what went wrong by printing to stderr and returning 1 or 2. That
is a complete answer for a terminal and no answer at all for a web server: the
prose lands on whichever console launched the process, and the caller gets a
number it cannot map to a status code without re-deriving what the number meant.

So the wording moves here, verbatim -- these are the sentences `cli.py` already
prints, and they are unusually good ones. What changes is who decides the
consequence. A raise says *what* is wrong; a frontend decides whether that is an
exit code, a 404, or a picker in a UI.

Two rules make that split hold:

  * **the message is written where the fact is known.** `_weights_for` knows the
    remembered weights are the wrong size and can say so exactly; a handler
    three layers up would have to guess.
  * **constructing an error does no I/O.** `cli._no_model_for` opens a socket to
    localhost:11434 while building its message, which is a one-off pause before
    exit on a terminal and a blocking syscall on a request thread in a server.
    An exception carries `role` instead, and the frontend adds the suggestions
    it is able to make.
"""

from __future__ import annotations

import re

from dyprys.lock import AlreadyRunning  # noqa: F401  -- one name for the caller

_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


class DyprysError(Exception):
    """Something dyprys refuses to do, with the explanation already written.

    `kind` is the stable slug a machine matches on; it is derived from the class
    name so it cannot drift from it. `message` is the prose a person reads.
    `exit_code` is the CLI's 1-or-2, kept on the class so the one handler in
    `main` reproduces today's codes without a table to maintain.
    """

    kind = "error"
    exit_code = 2

    def __init__(self, message: str, *, hint: str | None = None,
                 choices: list | None = None, role: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        # Which model the failure is about ("model", "reranker", "expander",
        # "summariser"), so a frontend can add the suggestions only it can make
        # -- what ollama has installed, which environment variable sets a
        # default -- without the core naming either.
        self.role = role
        # Never None. A caller that has to test before iterating will one day
        # forget to, and the UI that offers a picker is the one that suffers.
        self.choices = list(choices) if choices else []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "kind" not in cls.__dict__:
            cls.kind = _BOUNDARY.sub("_", cls.__name__).lower()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r})"


# --- the query -------------------------------------------------------------


class EmptyQuery(DyprysError):
    """Nothing to search for.

    Refused rather than run: an empty query embeds to a meaningless vector and
    matches no words, so hybrid search returns whatever the vector half drifts
    to -- noise presented as answers, at exit 0.
    """


class NoSuchBook(DyprysError):
    """A collection pattern, or a source path, that this index does not hold.

    Exit 1 rather than 2: the command was well formed, it simply found nothing,
    which is the same shape of answer as a search that matched no passage.
    """

    exit_code = 1


class Abandoned(DyprysError):
    """Whoever asked stopped listening, so the search stopped working.

    Not a fault: the answer had nowhere to go. It matters because a search runs
    on the one worker thread that library has, so a search nobody is waiting for
    is a search the next one is queued behind -- and a rescoring or a summary
    can hold that thread for half a minute after the browser tab that asked for
    it has closed.
    """

    exit_code = 1


class NothingEmbedded(DyprysError):
    """Text has been added but no vectors exist yet."""

    exit_code = 1


# --- where the index is ----------------------------------------------------


class NoSuchLibrary(DyprysError):
    """A name that resolves to nothing in the registry."""


class RegistryUnreadable(DyprysError):
    """The registry file exists and could not be parsed.

    Distinct from `NoSuchLibrary` on purpose: an unreadable registry makes
    *every* name fail, and telling someone "no library named x" when the real
    problem is a corrupt file sends them to fix the wrong thing.
    """


# --- which model ------------------------------------------------------------


class ModelAmbiguous(DyprysError):
    """More than one model has embedded this index and none was named.

    `choices` carries handles that can be pasted back into `--model`, because a
    UI can only offer the picker that unblocks the user if the options arrive as
    data. Search refuses rather than guessing which vectors to answer from.
    """


class ModelMissing(DyprysError):
    """A name was given and matches neither a file nor a model in this index."""


class WeightsAbsent(DyprysError):
    """The index knows which weights made its vectors; they are not here now."""


class WeightsMismatch(DyprysError):
    """The file at the remembered path is not the size the index recorded.

    Loading it anyway would register it as a *different* model rather than
    corrupting this one -- so nothing breaks, and the symptom is a very long run
    that looks entirely normal and shares none of the work already done.
    """


class ChunkingAmbiguous(DyprysError):
    """The index holds more than one chunking and the command needs one."""


class OptionalModelMissing(DyprysError):
    """A reranker, expander or summariser was asked for and none was named.

    Carries `role` and nothing else about how to fix it. The suggestions worth
    printing -- what ollama has installed right now -- need a network round trip
    the frontend can afford and a request thread cannot.
    """


class NotAReranker(DyprysError):
    """A file was named as a reranker and declares itself something else.

    Not `ModelUnavailable`: the file is here and it loads. It is the wrong kind
    of model, and llama.cpp will not say so -- rank pooling can be forced onto
    any model, and the result is a number per pair that looks exactly like a
    score. So this is the only place the difference is ever noticed.
    """


class ModelUnavailable(DyprysError):
    """A model was named and cannot be reached.

    Separate from `OptionalModelMissing` because the fix is different and the
    message already says what it is -- "cannot reach ollama at ..." replaced by
    a list of installed models would be worse than useless.
    """


# --- what a search needs ----------------------------------------------------


class NoRoutingProfile(DyprysError):
    """`--route` was asked for before `dyp route` has ever run."""


class SourceUnavailable(DyprysError):
    """This index owns the path, and the bytes behind it cannot be read."""

    exit_code = 1


# --- the long mutations -----------------------------------------------------


class NoSuchJob(DyprysError):
    """A job kind that is not one of the mutations this tool runs."""


class JobUnreachable(DyprysError):
    """Something holds the index and this process cannot signal it."""


class BadRequest(DyprysError):
    """A request naming something this tool has no option for."""
