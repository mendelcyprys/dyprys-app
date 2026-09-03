"""Step 09: ask a small model for other ways to say the question.

A query is one phrasing of an information need, and the corpus is not obliged to
share it. "What are enzyme linked receptors" and a textbook's "receptor tyrosine
kinases dimerise on ligand binding" are the same subject in disjoint vocabulary,
and the vector half exists precisely to bridge that -- but it bridges what one
embedding can reach, and one embedding of a short question is a thin thing to
bridge with.

Expansion produces several phrasings and searches with all of them. It fits this
architecture without forcing anything, because fusion already takes a *list* of
rankings and cares only about positions: a query rewritten three ways is three
more rankings, ranked and fused exactly like the vector and lexical halves
already are.

**Measured. It works, with a general model, and is still off by default.**
Main question set, n=110, frozen index of 204,076 chunks, paired McNemar on
recall@1:

    expander              recall@1   recall@5   lexical safety   ms/query
    none                    77/110    100/110       18/20            438
    fine-tuned 1.7B         71/110     94/110       16/20          1,873
    qwen3:4b-instruct       82/110    104/110       17/20         ~4,900
    gemma3:4b               86/110    104/110       18/20         ~4,700

gemma3:4b is +10/-2, p = 0.0386, and reproduced exactly across two independent
runs. It lands where it should: *oblique* questions -- phrased unlike the
passage answering them, which is the whole point -- go 52% to 68% at recall@1,
and *mechanism*, the weakest area in the project, 45% to 55%.

**The first version of this measurement said the opposite, and was wrong.** It
tested one expander, a model fine-tuned for another tool, and concluded the
technique did not pay. One general instruct model overturned that. A single
model is evidence about that model.

**Three things are necessary, and each was found by a failure.** A *general*
instruct model, not a task-fine-tuned one. A *non-reasoning* model: qwen3:4b
deliberates for 36 seconds and never reaches an answer. And the model's own chat
template, which is why `OllamaExpander` exists -- given a raw prompt with no
template, qwen3 handed the instructions back verbatim, output that parsed into
three clean lines and was worth nothing.

**Expansions go to the vector half only.** See `EXPAND_LEXICAL` in `cli.py`:
giving BM25 a rewritten query cost 4 of 20 exact-phrase probes. A phrase is a
lookup, and a lookup cannot be improved by rewording the thing being looked up
-- the same argument that already sends the exact-phrase attempt past the
router. Vector-only beats both-halves on every measure, so it is not a trade.

**Off by default because of latency, not quality.** ~4.7 s a query against 438
ms is ~11x, and a five-second interactive query is a different product rather
than a tuning choice. It is the cheaper of the two optional query-time stages by
an order of magnitude -- reranking costs ~9 s -- and the only one that moves
recall@1 on a search that already ranks well, which is why it is kept as a
supported option: right for a considered query, wrong for typing at a prompt.

Parsing is the whole of the difficulty and is kept a pure function, so its tests
need no model, no GPU and no fixtures -- the same reason `chunk_bytes` is pure.
Model output is not a protocol: it repeats itself, truncates mid-phrase, and
sometimes never closes the reasoning block it opened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# What an expansion line is for. `lex` goes to BM25, `vec` and `hyde` are
# embedded -- a hypothetical answer is a document, not a question, so it is
# tracked separately even though both end up in the vector half.
KINDS = ("lex", "vec", "hyde")

# A line has to say something. Two words is the floor at which a phrase is worth
# a search of its own; below that it is a fragment of a truncated line.
MIN_WORDS = 2

# Per kind. Each accepted line is another full search, so this is a latency
# budget as much as a quality one.
MAX_LINES = 3

_LINE = re.compile(r"^\s*(lex|vec|hyde)\s*:\s*(.+?)\s*$", re.I | re.M)

# Reasoning models emit a <think> block. It is not always closed -- one of the
# first two real outputs sampled had an opening tag and no closing one, with the
# expansion lines inside it -- so an unclosed block is stripped down to its tag
# rather than swallowing the rest of the output.
_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_TAG = re.compile(r"</?think>", re.I)


@dataclass(frozen=True)
class Expansion:
    """Alternative phrasings of one query, by what they are for."""

    lexical: list[str] = field(default_factory=list)
    vector: list[str] = field(default_factory=list)
    hyde: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.lexical or self.vector or self.hyde)

    @property
    def searches(self) -> int:
        """Extra searches this expansion costs, over the unexpanded query."""
        return len(self.lexical) + len(self.vector) + len(self.hyde)


def _is_echo(line: str) -> bool:
    """Whether the model handed the instructions back instead of following them.

    A reasoning model asked for `lex:`/`vec:`/`hyde:` lines will sometimes
    restate the request as its answer — "lex: two to six words likely to appear
    verbatim in the passage that answers this". That parses perfectly and is
    worth nothing, which makes it more dangerous than output that fails to
    parse: three extra searches for the instruction text itself, silently
    polluting the fusion. Checked against the instruction rather than a list of
    phrases, so it cannot fall out of step with what was asked for.
    """
    return line.casefold() in INSTRUCTION.casefold()


def _worth_keeping(line: str, already: list[str]) -> bool:
    """Whether a line adds anything to the ones already accepted.

    Four things get dropped, all seen in real output:

    * duplicates, which the model emits freely -- the same phrase under `lex`
      and again under `vec` is intended, but the same phrase twice under one
      kind is not;
    * fragments, where generation stopped mid-phrase. "how do enzyme-linked" is
      a prefix of "how do enzyme-linked receptors function", so prefix
      containment catches it without needing to guess at truncation;
    * anything under two words, which cannot be a phrasing of a question;
    * the instructions themselves, echoed back as though they were an answer.
    """
    if len(line.split()) < MIN_WORDS or _is_echo(line):
        return False
    folded = line.casefold()
    for seen in already:
        other = seen.casefold()
        if folded == other or other.startswith(folded) or folded.startswith(other):
            return False
    return True


def parse(text: str) -> Expansion:
    """Pull expansion lines out of whatever the model produced.

    Tolerant on purpose. Anything that is not a recognised line is ignored
    rather than treated as an error, because the failure this guards against is
    a malformed generation costing a user their query -- expansion is an
    optional improvement, and returning nothing is a valid outcome that simply
    falls back to searching the question as asked.
    """
    if not text:
        return Expansion()
    body = _THINK_CLOSED.sub(" ", text)
    body = _THINK_TAG.sub(" ", body)

    found: dict[str, list[str]] = {kind: [] for kind in KINDS}
    for match in _LINE.finditer(body):
        kind, line = match.group(1).lower(), match.group(2).strip()
        bucket = found[kind]
        if len(bucket) < MAX_LINES and _worth_keeping(line, bucket):
            bucket.append(line)
    return Expansion(lexical=found["lex"], vector=found["vec"], hyde=found["hyde"])


# A model fine-tuned for this task needs no instructions; a general one needs
# all of them. Three prompts are kept because the prompt is a tuning knob with
# a latency cost, and this project does not adopt a knob it has not measured.
#
# `lex` is asked for by the first prompt and then discarded: `EXPAND_LEXICAL` in
# cli.py is False, because handing BM25 a rewritten query cost 4 of 20
# exact-phrase probes. So the original prompt pays generation time for a line
# nothing reads, which is roughly a third of the output.
PROMPTS = {
    # What the measured +9 was obtained with. Kept verbatim as the baseline any
    # replacement has to beat, not because it is good.
    "full": """\
You help search a library of book-length texts. The user's query is searched \
two ways: by keyword match, and by embedding similarity against passages.

Reply with exactly these three lines and nothing else:

lex: two to six words likely to appear verbatim in the passage that answers this
vec: the query restated in the vocabulary a book on the subject would use
hyde: one sentence, written as it might appear in the book, answering the query

Query: {query}""",

    # The same instruction with the discarded line and the explanation removed.
    "compact": """\
Rewrite this book-search query. Two lines, nothing else:
vec: the query in the vocabulary a book on the subject would use
hyde: one sentence as that book would write it

{query}""",

    # Says what is wanted and lets the model choose how much of it to give. A
    # capable model may know better than a fixed recipe which query needs a
    # restatement, which needs an imagined passage, and which needs both.
    "judge": """\
Expand this book-search query to help find the passage answering it.
Use whichever helps, one per line, nothing else:
vec: <restatement in the book's own vocabulary>
hyde: <a sentence as the book would write it>

{query}""",
}

DEFAULT_PROMPT = "full"
INSTRUCTION = PROMPTS[DEFAULT_PROMPT]

# Long enough for three lines of each kind plus a reasoning block, short enough
# that a model which starts repeating itself is cut off rather than indulged.
MAX_TOKENS = 220


class OllamaExpander:
    """A general instruct model, reached through a local ollama server.

    Two reasons for HTTP rather than loading the GGUF directly. Ollama applies
    each model's own chat template, which differs between families and is the
    kind of detail that silently degrades output when guessed at -- the reranker
    lost to a wrong template before. And it keeps a 2.5 GB model out of this
    process while a 200,000-chunk embedding run is using the GPU.

    Uses `urllib` rather than `requests`, so it adds no dependency.
    """

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 timeout: float = 120.0, keep_alive: str = "30m",
                 prompt: str = DEFAULT_PROMPT):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.instruction = PROMPTS.get(prompt, PROMPTS[DEFAULT_PROMPT])
        # Ollama loads a model on first use and unloads it after five minutes
        # idle. That default costs ~10 s on the first query of every session --
        # it is the whole of the difference between the 14.3 s and 4.7 s
        # per-query figures measured for the same arm. Asking it to stay
        # resident makes the second query as fast as the tenth.
        self.keep_alive = keep_alive

    def unavailable(self) -> str | None:
        """Why this expander cannot be used, or None if it can.

        Checked once when the expander is built, not per query. `--expand` is an
        explicit request: if it cannot be honoured the user has to be told,
        because a search that silently skips expansion is indistinguishable from
        a normal one -- same output, same exit code, 468 ms instead of 5 s.
        Per-query failures still fail soft, which is a different case: there the
        model answered and the answer was unusable.
        """
        import json
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as response:
                names = {m["name"] for m in json.loads(response.read()).get("models", [])}
        except (urllib.error.URLError, OSError, ValueError):
            return (f"cannot reach ollama at {self.host} — start it with `ollama serve`, "
                    f"or pass a .gguf path to --expander instead")
        if self.model in names or f"{self.model}:latest" in names:
            return None
        return (f"ollama has no model named {self.model!r} — "
                f"pull it with `ollama pull {self.model}`"
                + (f", or use one of: {', '.join(sorted(names)[:4])}" if names else ""))

    def expand(self, query: str) -> Expansion:
        import json
        import urllib.error
        import urllib.request

        # /api/chat, not /api/generate: the latter passes the prompt through
        # raw, and a model that never sees its own chat template continues the
        # text instead of answering it. Qwen3 echoed the instructions back
        # verbatim as its "expansion" -- output that parses cleanly and is
        # entirely worthless, which is the worst kind.
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": self.instruction.format(query=query)}],
            "stream": False,
            # Reasoning is not wanted here: the task is a rewrite, and thinking
            # costs seconds per query for output the parser then discards.
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.0, "num_predict": MAX_TOKENS, "seed": 0},
        }).encode()
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError):
            # An expander that cannot be reached costs the improvement, never
            # the query -- the same contract as an unparseable generation.
            return Expansion()
        return parse((payload.get("message") or {}).get("content", ""))


class Expander:
    """A small instruct model that rewrites a query several ways.

    Loaded like `Embedder` and `Reranker`: this is the only class that knows a
    model handle exists, so everything downstream deals in `Expansion` and can
    be tested against a stub.
    """

    def __init__(self, model_path: str | Path, n_gpu_layers: int = -1,
                 n_ctx: int = 2048, seed: int = 0, instruct: bool = True,
                 prompt: str = DEFAULT_PROMPT):
        from llama_cpp import Llama

        self.path = Path(model_path)
        self._llm = Llama(
            model_path=str(self.path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=False,
        )
        # A general model is told what to produce; a model fine-tuned for this
        # task already knows and is given the bare question. Which of those is
        # right is a property of the *weights*, not of how they are reached --
        # an earlier version decided it by backend, so the same model expanded
        # differently through ollama than from a file.
        self.instruct = instruct
        self.instruction = PROMPTS.get(prompt, PROMPTS[DEFAULT_PROMPT])

    def expand(self, query: str) -> Expansion:
        """Alternative phrasings, or an empty expansion if nothing usable came back.

        Temperature 0, so the same question expands the same way twice. A
        measurement that moves when nothing changed cannot decide anything, and
        this project has already been caught once by a number that drifted for
        reasons outside the change being tested.

        Goes through `create_chat_completion`, which applies the chat template
        stored in the GGUF itself. The first version hardcoded ChatML, which is
        Qwen's -- correct for the fine-tuned expander and wrong for every other
        family, and wrong in the way that does not look like an error: a model
        that never sees its own template continues the prompt instead of
        answering it. That is the same failure `/api/generate` produced, and it
        would have made this path quietly useless for exactly the general models
        that measured best.
        """
        content = self.instruction.format(query=query) if self.instruct else query
        out = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": content}],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        return parse(out["choices"][0]["message"]["content"] or "")
