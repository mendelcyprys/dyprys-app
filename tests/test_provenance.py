"""A backup should tell you which weights it needs, and let you check a candidate."""

import json
import tarfile

import pytest

from dyprys import db
from dyprys.backup import read_manifest, write
from dyprys.embed import store_for

DIM = 8
WEIGHTS = {
    "file_name": "some-model-Q8_0.gguf",
    "file_bytes": 333_590_944,
    "file_sha256": "b5ce9d77a3fc" + "0" * 52,
}


def test_registering_records_what_the_weights_were(conn):
    db.model_id(conn, "m@abc", DIM, provenance=WEIGHTS)

    row = db.model_provenance(conn)[0]
    assert row["file_name"] == WEIGHTS["file_name"]
    assert row["file_bytes"] == WEIGHTS["file_bytes"]
    assert row["file_sha256"] == WEIGHTS["file_sha256"]


def test_an_older_index_learns_provenance_from_a_later_run(conn):
    """Indexes built before this existed should not stay unidentifiable."""
    model = db.model_id(conn, "m@abc", DIM)
    assert db.model_provenance(conn)[0]["file_sha256"] is None

    again = db.model_id(conn, "m@abc", DIM, provenance=WEIGHTS)

    assert again == model
    assert db.model_provenance(conn)[0]["file_sha256"] == WEIGHTS["file_sha256"]


def test_recorded_provenance_is_not_overwritten(conn):
    """The first recording wins; a later run cannot silently rewrite history."""
    db.model_id(conn, "m@abc", DIM, provenance=WEIGHTS)
    db.model_id(conn, "m@abc", DIM, provenance={**WEIGHTS, "file_name": "other.gguf"})

    assert db.model_provenance(conn)[0]["file_name"] == WEIGHTS["file_name"]


def test_a_source_can_be_recorded_and_read_back(conn):
    model = db.model_id(conn, "m@abc", DIM, provenance=WEIGHTS)
    db.set_model_source(conn, model, "hf:org/repo/file.gguf")

    assert db.model_provenance(conn)[0]["source_uri"] == "hf:org/repo/file.gguf"


def test_a_model_without_provenance_reports_nothing_rather_than_guessing(conn):
    db.model_id(conn, "m@abc", DIM)
    row = db.model_provenance(conn)[0]
    assert row["file_sha256"] is None
    assert row["source_uri"] is None


# --- it travels in the archive ----------------------------------------------


@pytest.fixture
def archived(conn, tmp_path, library):
    model = db.model_id(conn, "m@abc", DIM, provenance=WEIGHTS)
    db.set_model_source(conn, model, "hf:org/repo/file.gguf")
    store_for(conn, tmp_path / "index", model, DIM).flush()
    out = tmp_path / "backup.tar.gz"
    write(conn, tmp_path / "index", out)
    return out


def test_the_manifest_says_which_weights_are_needed(archived):
    weights = read_manifest(archived)["weights"]

    assert len(weights) == 1
    assert weights[0]["file_sha256"] == WEIGHTS["file_sha256"]
    assert weights[0]["source_uri"] == "hf:org/repo/file.gguf"
    assert weights[0]["dim"] == DIM


def test_the_manifest_can_be_read_without_restoring(archived):
    """Someone deciding whether they can use an archive should not have to unpack it."""
    with tarfile.open(archived) as tar:
        manifest = json.loads(tar.extractfile("manifest.json").read())

    assert manifest["weights"][0]["file_name"] == WEIGHTS["file_name"]


def test_the_digest_recorded_is_the_whole_one(conn):
    """Twelve characters name a model; sixty-four verify a download."""
    db.model_id(conn, "m@abc", DIM, provenance=WEIGHTS)
    assert len(db.model_provenance(conn)[0]["file_sha256"]) == 64


# --- being able to refer to a model without repeating its path ---------------


def test_a_model_can_be_given_a_short_name(conn):
    model = db.model_id(conn, "hf_org_embeddinggemma-300M-Q8_0@b5ce9d77a3fc", 768)

    db.set_alias(conn, model, "gemma")

    assert db.find_model(conn, "gemma")["id"] == model
    assert db.find_model(conn, "embeddinggemma")["id"] == model, "substring still works"
    assert db.find_model(conn, "nothing-like-this") is None


def test_an_ambiguous_substring_matches_nothing(conn):
    """Better to ask than to guess which of two models was meant."""
    db.model_id(conn, "gemma-300M@aaaaaaaaaaaa", 768)
    db.model_id(conn, "gemma-700M@bbbbbbbbbbbb", 768)

    assert db.find_model(conn, "gemma") is None
    assert db.find_model(conn, "300M") is not None


def test_an_alias_beats_a_substring_of_a_hash(conn):
    """A nickname a person chose must never be ambiguous with part of a digest."""
    one = db.model_id(conn, "alpha@abc123def456", 768)
    db.model_id(conn, "beta@999abc123fff", 768)
    db.set_alias(conn, one, "abc123")

    assert db.find_model(conn, "abc123")["id"] == one


def test_the_sole_model_is_only_sole_when_it_is(conn):
    assert db.sole_model(conn) is None
    first = db.model_id(conn, "only@aaaaaaaaaaaa", 768)
    assert db.sole_model(conn)["id"] == first
    db.model_id(conn, "second@bbbbbbbbbbbb", 768)
    assert db.sole_model(conn) is None


def test_where_the_weights_were_last_opened_is_remembered(conn, tmp_path):
    model = db.model_id(conn, "gemma@aaaaaaaaaaaa", 768)
    weights = tmp_path / "gemma.gguf"
    weights.write_bytes(b"x" * 64)

    db.remember_weights(conn, model, weights)

    assert db.find_model(conn, "gemma")["file_path"] == str(weights)


def test_the_remembered_path_is_a_hint_and_the_size_is_checked(conn, tmp_path):
    """Wrong weights at the remembered path must not be loaded silently.

    They would register as a *different* model rather than corrupting this one,
    so nothing breaks — but the symptom would be a very long run that looks
    entirely normal and shares none of the work already done.
    """
    from dyprys.cli import _weights_for

    model = db.model_id(conn, "gemma@aaaaaaaaaaaa", 768,
                        provenance={"file_name": "gemma.gguf", "file_bytes": 1000,
                                    "file_sha256": "a" * 64})
    weights = tmp_path / "gemma.gguf"
    weights.write_bytes(b"x" * 999)
    db.remember_weights(conn, model, weights)

    path, why = _weights_for(conn, None)

    assert path is None
    assert "not the" in why and "--model" in why


def test_a_usable_remembered_path_needs_no_argument(conn, tmp_path):
    from dyprys.cli import _weights_for

    model = db.model_id(conn, "gemma@aaaaaaaaaaaa", 768,
                        provenance={"file_name": "gemma.gguf", "file_bytes": 64,
                                    "file_sha256": "a" * 64})
    weights = tmp_path / "gemma.gguf"
    weights.write_bytes(b"x" * 64)
    db.remember_weights(conn, model, weights)

    assert _weights_for(conn, None) == (str(weights), None)


def test_with_no_model_at_all_it_says_how_to_start(conn):
    from dyprys.cli import _weights_for

    path, why = _weights_for(conn, None)

    assert path is None and "--model" in why


def test_two_models_cannot_share_a_nickname(conn):
    import sqlite3

    a = db.model_id(conn, "one@aaaaaaaaaaaa", 768)
    b = db.model_id(conn, "two@bbbbbbbbbbbb", 768)
    db.set_alias(conn, a, "gemma")

    with pytest.raises(sqlite3.IntegrityError):
        db.set_alias(conn, b, "gemma")


def test_models_without_a_nickname_do_not_collide(conn):
    """A unique index permits any number of NULLs, which is what we want here."""
    db.model_id(conn, "one@aaaaaaaaaaaa", 768)
    db.model_id(conn, "two@bbbbbbbbbbbb", 768)

    assert conn.execute("SELECT COUNT(*) FROM models WHERE alias IS NULL").fetchone()[0] == 2


def test_every_handle_cyp_models_offers_actually_resolves(conn, tmp_path):
    """Printing a handle that does not work is worse than printing none.

    `dyp models` lists what you can pass to --model. This asserts the listing
    and the resolver agree, for a named model and an unnamed one, over names
    long enough that the displayed form is a truncation of the real one.
    """
    from dyprys.library import models as inspect

    long_name = "hf_ggml-org_some-really-long-model-name-v2@abc123def456"
    first = db.model_id(conn, long_name, 384)
    second = db.model_id(conn, "hf_org_embeddinggemma-300M-Q8_0@b5ce9d77a3fc", 768)
    db.set_alias(conn, second, "gemma")

    for info in inspect(conn, tmp_path):
        handles = ([info.alias] if info.alias else []) + [
            info.name.split("@")[0][-28:] if "@" in info.name else info.name
        ]
        for handle in handles:
            found = db.find_model(conn, handle)
            assert found is not None, f"{handle!r} is offered but resolves to nothing"
            assert found["id"] == info.id, f"{handle!r} resolves to the wrong model"

    assert db.find_model(conn, "gemma")["id"] == second
    assert db.find_model(conn, "me-really-long-model-name-v2")["id"] == first


def test_the_listing_reports_the_alias_and_the_weights_path(conn, tmp_path):
    from dyprys.library import models as inspect

    model = db.model_id(conn, "gemma@aaaaaaaaaaaa", 768)
    db.set_alias(conn, model, "g")
    weights = tmp_path / "g.gguf"
    weights.write_bytes(b"x")
    db.remember_weights(conn, model, weights)

    info = inspect(conn, tmp_path)[0]

    assert info.alias == "g"
    assert info.file_path == str(weights)
