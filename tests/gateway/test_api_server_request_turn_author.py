"""``body.author`` is the API server's per-turn attribution input, and only that.

The hub authenticates the caller with the API bearer token and then names the human in
``body.author`` on /v1/runs and the legacy session stream. These tests pin the request-parsing
contract feeding ``run_conversation(turn_author=...)``: a usable object normalizes, junk is
rejected or dropped to None (never inherited), and nothing about the author is authorization.
"""
import pytest

from agent.turn_author import parse_turn_author
from gateway.platforms.api_server import _request_turn_author

MENG = {"id": "meng", "name": "Meng", "is_bot": False}
LENA = {"id": "lena", "name": "Lena", "is_bot": False}


@pytest.mark.parametrize("author", [MENG, LENA])
def test_valid_author_normalizes(author):
    assert _request_turn_author({"author": author}) == author


def test_author_is_normalized_not_passed_through_raw():
    parsed = _request_turn_author({"author": {"id": " meng ", "name": "Meng\n", "extra": "ignored"}})
    assert parsed == {"id": "meng", "name": "Meng", "is_bot": False}
    assert "extra" not in parsed


@pytest.mark.parametrize("body", [{}, {"author": None}])
def test_absent_or_null_author_is_none(body):
    # None means "this turn has no author" — the caller then omits turn_author entirely,
    # so a cached agent resets to None rather than keeping the previous caller's author.
    assert _request_turn_author(body) is None


@pytest.mark.parametrize("raw", ["meng", 7, ["meng"], True])
def test_non_object_author_is_a_400_not_a_silent_default(raw):
    with pytest.raises(ValueError):
        _request_turn_author({"author": raw})


@pytest.mark.parametrize("raw", [{}, {"id": "  "}, {"id": None, "name": None}, {"is_bot": True}])
def test_object_without_usable_identity_is_none(raw):
    assert _request_turn_author({"author": raw}) is None


def test_is_bot_defaults_false_for_a_human_turn():
    assert _request_turn_author({"author": {"id": "meng"}})["is_bot"] is False


def test_client_supplied_is_bot_is_a_label_only():
    # A client may claim is_bot; it changes attribution shape, never a permission.
    assert _request_turn_author({"author": {"id": "meng", "is_bot": True}})["is_bot"] is True


def test_parse_matches_the_shared_normalizer():
    assert _request_turn_author({"author": MENG}) == parse_turn_author(MENG)


def test_author_carries_no_credential_or_permission_keys():
    parsed = _request_turn_author({"author": {
        "id": "meng", "name": "Meng", "token": "t", "api_key": "k",
        "allowed_tools": ["terminal"], "is_admin": True}})
    assert set(parsed) == {"id", "name", "is_bot"}


def test_other_body_fields_are_untouched():
    body = {"input": "hi", "author": MENG}
    _request_turn_author(body)
    assert body == {"input": "hi", "author": MENG}
