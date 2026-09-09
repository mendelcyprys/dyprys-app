"""Step 10: an answer assembled from passages, where every quote is checked.

The design doc put answer synthesis out of scope, and the objection was
reliability: a model that paraphrases a source into something it does not say,
or attributes a real sentence to the wrong book, is worse than no answer at all
because it looks like one.

That objection is answerable, and the answer is not a better model. **Nothing
here trusts the model.**

* Every quote must be a literal substring of the passage it cites. It is checked
  against the bytes `read_span` returned, and a quote that does not verify is
  removed from the answer. Measured on a first probe: 5 of 7 quotes verified,
  and one of the failures was a real misattribution -- text quoted from one
  passage and cited to another. A trusting implementation ships that.
* **Locations never come from the model at all.** It chooses among numbered
  passages; the book, chapter and byte offset are attached afterwards from the
  index, which already holds them. There is nothing there for it to get wrong.

So the model's job is narrowed to the one thing it is good at -- deciding which
passage bears on the question and how to say why -- and everything checkable is
checked. What cannot be checked is its prose, so the prose is kept subordinate
to the quotes rather than the other way round.

Verification is pure, so its tests need no model, no GPU and no index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Only true quotation marks. An apostrophe is not one, and treating it as one
# makes "Here's an answer" open a quotation that swallows the rest of the line
# -- which is exactly what the first version of this regex did, reporting zero
# verified quotes on output whose quotes were all fine.
_QUOTE = "[\"“”]"

# A quote followed by the passage it is attributed to: "…" [2].
#
# The minimum length is deliberately tiny. An earlier version required twelve
# characters to avoid matching stray quoted words, and the cost was that a short
# quotation -- "saltatory" [1] -- was never checked at all: not verified, not
# rejected, and left standing in the prose looking supported. Precision comes
# from requiring the citation immediately after the closing quote, not from
# length, so anything offered as a quotation is now checked as one.
_CITED = re.compile(_QUOTE + r"(.{2,}?)" + _QUOTE + r"\s*\[(\d+)\]", re.S)

# A citation with no quotation marks around what it supports. Models do this
# constantly -- they copy a sentence verbatim, append `[1]`, and omit the
# quotes -- and the first version of this module saw no `"..." [n]` pair, said
# "nothing here has been checked", and left the sentence standing under its
# citation looking sourced. Unverified text presented as sourced is the exact
# failure this module exists to prevent, so a bare citation is now treated as a
# claim: the sentence in front of it is checked against the passage it names,
# and it either verifies as a quotation or is marked as a paraphrase.
_BARE = re.compile(r"(?<![\"\u201c\u201d])\s*\[(\d+)\]")

# How far back a bare citation reaches for the claim it supports: to the end of
# the previous sentence, or the start of the line.
_SENTENCE_END = re.compile(r"(?:^|[.!?]\s+|\n)")

# Typographic substitutions a model makes freely while believing it is copying
# verbatim. Normalising both sides is the difference between "does not verify"
# and "verifies", and neither answer would be wrong about the *words*.
_SUBSTITUTIONS = (
    ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
    ("—", "-"), ("–", "-"), ("…", "..."),
    ("ﬁ", "fi"), ("ﬂ", "fl"),      # PDF extraction leaves ligatures
)


def normalise(text: str) -> str:
    """Fold the differences that are typography rather than content."""
    for fancy, plain in _SUBSTITUTIONS:
        text = text.replace(fancy, plain)
    return " ".join(text.split())


@dataclass(frozen=True)
class Claim:
    """One quoted span and the passage the model attributed it to."""

    quote: str
    cited: int                      # 1-based, as shown to the model
    quoted: bool = True             # False when the model cited without quoting
    verified: bool = False
    chunk_id: int | None = None     # filled from the index, never from the model
    title: str | None = None
    offset: int | None = None

    @property
    def location(self) -> str:
        """Where this sits, in the index's words rather than the model's."""
        if self.chunk_id is None:
            return "unattributed"
        where = self.title or "?"
        return f"{where} (chunk {self.chunk_id}, byte {self.offset})"


@dataclass(frozen=True)
class Answer:
    prose: str
    claims: list[Claim]

    @property
    def verified(self) -> list[Claim]:
        return [c for c in self.claims if c.verified]

    @property
    def rejected(self) -> list[Claim]:
        return [c for c in self.claims if not c.verified]

    @property
    def trustworthy(self) -> bool:
        """At least one checked quote, and nothing that failed checking.

        Deliberately strict. A single unverifiable quote means the model
        attributed something to a passage that does not contain it, and that is
        the failure this whole module exists to catch -- not a blemish on an
        otherwise fine answer.
        """
        return bool(self.verified) and not self.rejected


def find_claims(text: str) -> list[Claim]:
    """Every claim the output attributes to a passage, quoted or not.

    A quoted claim is `"words" [n]`. A bare one is a sentence followed by `[n]`
    with no quotation marks, which is what models actually produce most of the
    time. Both are checked the same way; the difference is only that a bare
    claim which fails to verify is a paraphrase rather than a misquotation.
    """
    if not text:
        return []
    claims, consumed = [], []
    for match in _CITED.finditer(text):
        claims.append(Claim(quote=match.group(1).strip(), cited=int(match.group(2)),
                            quoted=True))
        consumed.append(match.span())

    for match in _BARE.finditer(text):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        before = text[:match.start()]
        cut = max((m.end() for m in _SENTENCE_END.finditer(before)), default=0)
        sentence = before[cut:].strip().strip("\u201c\u201d\"")
        if sentence:
            claims.append(Claim(quote=sentence, cited=int(match.group(1)), quoted=False))
    return claims


def verify(claims: list[Claim], passages: list) -> list[Claim]:
    """Check each quote against the passage it cites, and attach its location.

    `passages` are `search.Passage` objects in the order they were shown, so
    citation *n* is `passages[n - 1]`. A citation outside that range is not
    charitably reinterpreted -- a model that invents a passage number has told
    you something about how much of the rest to believe.
    """
    checked = []
    for claim in claims:
        index = claim.cited - 1
        if not (0 <= index < len(passages)):
            checked.append(claim)
            continue
        passage = passages[index]
        body = normalise(passage.text) if passage.text else ""
        ok = bool(body) and normalise(claim.quote) in body
        checked.append(Claim(
            quote=claim.quote,
            cited=claim.cited,
            quoted=claim.quoted,
            verified=ok,
            chunk_id=passage.chunk_id,
            title=passage.title,
            offset=getattr(passage, "offset", None),
        ))
    return checked


def redact(text: str, rejected: list[Claim]) -> str:
    """Remove quotes that did not verify, leaving a mark where they were.

    The prose around a bad quote is not salvaged and not shown as though it
    were sound: the sentence is what the quote was offered as evidence for.
    """
    for claim in rejected:
        cite = r"\s*\[" + str(claim.cited) + r"\]"
        if claim.quoted:
            pattern = re.compile(_QUOTE + re.escape(claim.quote) + _QUOTE + cite)
            text = pattern.sub("[unverifiable quotation removed]", text)
        else:
            # A bare citation that does not verify is a paraphrase, not a
            # misquotation. The sentence may still be a fair reading, so it is
            # marked rather than deleted -- but it must never keep a citation
            # implying the passage says it in those words.
            pattern = re.compile(re.escape(claim.quote) + cite)
            text = pattern.sub(claim.quote + " [paraphrase, not in the source]", text)
    return text


# What the model is asked. Numbered passages, quotes required, and an explicit
# way to say the passages do not answer -- without which a model will always
# find something, and "nothing here answers this" is a useful search result.
PROMPT = """\
Answer the question using ONLY the numbered passages below.

Rules:
- Support every claim with an exact quotation, copied character for character,
  followed by the passage number: "exact words here" [2].
- Do not quote anything that is not in the passages.
- If the passages do not answer the question, reply exactly: NO ANSWER IN PASSAGES

Question: {question}

Passages:
{passages}"""

NO_ANSWER = "NO ANSWER IN PASSAGES"

# How much of each passage the model is shown. Raised from 900 once passages
# began arriving with context either side: 900 was a quarter of a chunk, and a
# chunk that starts mid-sentence gave the model a fragment to work from. The
# first summary this project produced opened "neuropeptides, and neurosteroids,
# as well as..." because that is where the chunk started and the model copied
# what it was handed. Prompt tokens are cheap next to generated ones.
PASSAGE_CHARS = 1600


def render(passages: list) -> str:
    """The numbered block the model reads.  Numbering is 1-based and positional."""
    return "\n\n".join(
        f"[{n}] {' '.join(p.text.split())[:PASSAGE_CHARS]}"
        for n, p in enumerate(passages, 1) if p.text
    )


def refused(prose: str) -> bool:
    """Did the model say the passages do not answer the question?

    The sentinel on a line of its own, rather than as the whole of the output.
    `prose.strip() == NO_ANSWER` was the test in two places -- the retry in
    `service` and the refusal panel in the browser -- and a model that obeys
    the instruction *and* keeps talking defeats both at once. Asking a real
    library "fairness opinion" produced:

        Goldman Sachs advised the Special Committee of Cox Communications. [2]

        NO ANSWER IN PASSAGES

    which was read as an answer. So the rephrase-and-try-again that
    `--summarise` is documented to do never ran, and the token itself was
    rendered to the reader as prose. A model that emits it at all is saying it
    could not answer; that reading is also the safe one, because the retry
    costs nothing when the first attempt was fine.
    """
    return any(line.strip() == NO_ANSWER for line in prose.splitlines())


def answer_from(text: str, passages: list) -> Answer:
    """Turn a generation into a checked answer.  Pure: no model, no index.

    A refusal is normalised to the bare sentinel, so every caller that has to
    recognise one -- and there is one in each frontend -- can do it by equality
    and none of them can drift from this rule.
    """
    if refused(text):
        return Answer(prose=NO_ANSWER, claims=[])
    claims = verify(find_claims(text), passages)
    rejected = [c for c in claims if not c.verified]
    return Answer(prose=redact(text, rejected), claims=claims)


# How long the answer may be. Short enough that the model quotes rather than
# retells: given room to write an essay it writes one, and an essay has more
# unquotable prose in it, which is the part nothing can check.
MAX_TOKENS = 400


def ask_ollama(model: str, prompt: str, host: str = "http://localhost:11434",
               timeout: float = 300.0) -> str:
    """One completion from a local ollama model, or "" if it cannot be had.

    Deliberately duplicates a few lines of `expand.OllamaExpander` rather than
    sharing a transport with it. This is a prototype for a step the design doc
    put out of scope; if it earns a place the two should share one client, and
    if it does not, nothing else has been bent around it.
    """
    import json
    import urllib.error
    import urllib.request

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.0, "num_predict": MAX_TOKENS, "seed": 0},
    }).encode()
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return (json.loads(response.read()).get("message") or {}).get("content", "")
    except (urllib.error.URLError, OSError, ValueError):
        return ""


# Asked of the same model when the first search comes back with nothing. It is
# the expansion task, but reached from a different place: the search has already
# failed, and that failure is the signal to try other words.
REPHRASE = """\
A library search for this question found nothing that answers it.
Give one different way to ask it, using the vocabulary a book on the subject \
would use. Reply with the rephrasing alone, no preamble.

Question: {question}"""


def rephrase(question: str, ask) -> str:
    """One other way to ask, or "" if nothing usable came back."""
    said = (ask(REPHRASE.format(question=question)) or "").strip()
    # A model told "no preamble" supplies one anyway; take the longest line that
    # is not obviously a sentence about the task.
    lines = [line.strip(" \"\u201c\u201d*-") for line in said.splitlines() if line.strip()]
    usable = [line for line in lines
              if 3 <= len(line.split()) <= 40 and not line.lower().startswith(
                  ("here", "sure", "one ", "a different", "rephrase"))]
    return usable[0] if usable else ""


def summarise(question: str, passages: list, ask) -> Answer:
    """Answer `question` from `passages`, keeping only what verifies.

    `ask` is any callable from prompt to text, so this is testable against a
    stub and indifferent to which model or transport is behind it.
    """
    usable = [p for p in passages if p.text]
    if not usable:
        return Answer(prose=NO_ANSWER, claims=[])
    text = ask(PROMPT.format(question=question, passages=render(usable)))
    return answer_from(text, usable)
