"""One writer per index, and being able to tell from outside that it is there."""

import pytest

from dyprys.lock import AlreadyRunning, exclusive, holder


def test_nothing_holds_a_fresh_directory(tmp_path):
    assert holder(tmp_path) is None


def test_the_holder_is_visible_while_the_lock_is_held(tmp_path):
    import os

    with exclusive(tmp_path, "embed"):
        assert holder(tmp_path, "embed") == os.getpid()


def test_the_lock_is_free_again_afterwards(tmp_path):
    """A released lock is not an erased file: the pid stays behind.

    So the pid on its own is not evidence, and `holder` asks the lock rather
    than the file — otherwise `dyp watch` would follow a run that ended hours
    ago and report a rate of zero forever.
    """
    with exclusive(tmp_path, "embed"):
        pass

    assert (tmp_path / ".embed.lock").exists()
    assert (tmp_path / ".embed.lock").read_text().strip().isdigit()
    assert holder(tmp_path, "embed") is None


def test_a_second_holder_is_refused(tmp_path):
    with exclusive(tmp_path, "embed"):
        with pytest.raises(AlreadyRunning):
            with exclusive(tmp_path, "embed"):
                pass


def test_locks_of_different_names_do_not_collide(tmp_path):
    with exclusive(tmp_path, "embed"):
        assert holder(tmp_path, "compact") is None
