"""Properties the chunker must hold, whatever the tuning."""

import pytest

from dyprys.chunker import chunk_bytes


def test_spans_are_ordered_and_disjoint():
    data = (b"paragraph one. " * 40 + b"\n\n") * 30
    spans = chunk_bytes(data)

    assert len(spans) > 1
    ends = [offset + length for offset, length in spans]
    starts = [offset for offset, _ in spans]
    assert starts == sorted(starts)
    assert all(end <= next_start for end, next_start in zip(ends, starts[1:]))


def test_no_text_is_lost():
    """Every byte outside a span is whitespace -- nothing droppable is dropped."""
    data = b"\n\n".join(f"Paragraph {n} about neurons. ".encode() * 20 for n in range(200))
    spans = chunk_bytes(data)

    covered = 0
    for offset, length in spans:
        assert data[covered:offset].isspace() or covered == offset
        covered = offset + length
    assert data[covered:] == b"" or data[covered:].isspace()


def test_chunks_respect_the_target():
    data = b"word " * 20_000
    spans = chunk_bytes(data, target=1000)

    # Every chunk but the last fits the budget; none is absurdly short.
    for _, length in spans[:-1]:
        assert 750 <= length <= 1000


def test_multibyte_characters_are_never_split():
    """A hard cut with no whitespace nearby must still land on a boundary."""
    data = "é".encode() * 5000  # two bytes each, not a space in sight
    spans = chunk_bytes(data, target=101)  # odd target, so cuts want to land mid-character

    for offset, length in spans:
        data[offset : offset + length].decode("utf-8")  # raises if split


def test_spans_never_start_or_end_on_whitespace():
    data = b"\n\n\n   sentence one.\n\n\n\n   sentence two.   \n\n\n" * 300
    for offset, length in chunk_bytes(data, target=200):
        assert not data[offset : offset + 1].isspace()
        assert not data[offset + length - 1 : offset + length].isspace()


def test_prefers_a_paragraph_break_over_a_space():
    body = b"x" * 500
    data = body + b"\n\n" + body + b" " + body
    spans = chunk_bytes(data, target=600)

    assert spans[0] == (0, 500)  # cut at the blank line, not at 600


def test_whitespace_only_input_yields_nothing():
    assert chunk_bytes(b"   \n\n\t  \n") == []


def test_empty_input_yields_nothing():
    assert chunk_bytes(b"") == []


def test_short_input_is_one_chunk():
    assert chunk_bytes(b"  a short book.  ") == [(2, len(b"a short book."))]


def test_overlap_repeats_text_between_neighbours():
    data = b"word " * 4000
    spans = chunk_bytes(data, target=1000, overlap=200)

    for (offset, length), (next_offset, _) in zip(spans, spans[1:]):
        assert next_offset < offset + length


def test_overlap_does_not_shred_the_tail():
    """The last chunk ends the file; it is not re-cut once per overlap byte.

    The bug this pins: `start` stepped back by the overlap even when the span
    had already reached the end of the input, so the tail was emitted again and
    again, one byte shorter each time -- exactly `overlap` extra spans per file.
    They were degenerate (down to a single byte) and, because BM25 normalises by
    document length, they outranked the paragraphs they were cut from.
    """
    data = b"".join(b"Paragraph %d about the hippocampus.\n\n" % i for i in range(400))

    for overlap in (0, 100, 360, 900):
        spans = chunk_bytes(data, target=3600, overlap=overlap)
        lengths = [length for _, length in spans]

        # One chunk per (target - overlap) bytes of text, give or take a break.
        assert len(spans) < 2 * len(data) // (3600 - overlap) + 4, (
            f"overlap={overlap} produced {len(spans)} spans for {len(data)} bytes"
        )
        # No runts. The final span is a legitimate remainder and may be any
        # size; every span before it must be a real passage.
        assert min(lengths[:-1]) > 3600 // 4, (
            f"overlap={overlap} produced a {min(lengths[:-1])}-byte span before the end"
        )
        assert len(set(spans)) == len(spans), f"overlap={overlap} repeated a span"


def test_the_last_span_reaches_the_end_of_the_text():
    data = b"Sentence one. " * 500 + b"The final clause."

    for overlap in (0, 200, 700):
        offset, length = chunk_bytes(data, target=2000, overlap=overlap)[-1]
        assert offset + length == len(data)


def test_it_is_deterministic():
    data = b"neurons fire together. " * 5000
    assert chunk_bytes(data) == chunk_bytes(data)


@pytest.mark.parametrize("target,overlap", [(0, 0), (-1, 0), (100, 100), (100, -1)])
def test_bad_parameters_are_rejected(target, overlap):
    with pytest.raises(ValueError):
        chunk_bytes(b"text", target=target, overlap=overlap)


def test_ascii_separators_count_as_whitespace():
    """PDF extraction leaves 0x1c-0x1f about; str.isspace() treats them as blank."""
    data = b"\x1f\x1d First passage.\x1e\x1c" * 200
    for offset, length in chunk_bytes(data, target=300):
        passage = data[offset : offset + length].decode()
        assert passage == passage.strip()
