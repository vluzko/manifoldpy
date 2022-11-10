"""Tests of prepared requests for the API.
These are used as fast tests for the POST endpoints, which require setting up a development server to test properly.
"""
from manifoldpy import api


def test_bet_prepared():
    wrapper = api.APIWrapper("no_key")
    prepped = wrapper._prep_make_bet(10, "1", "YES")
    assert prepped.headers == {
        "Content-Type": "application/json",
        "Authorization": "Key no_key",
        "Content-Length": "51",
    }
    assert prepped.body == b'{"amount": 10, "contractId": "1", "outcome": "YES"}'


def test_cancel_prepared():
    wrapper = api.APIWrapper("no_key")
    prepped = wrapper._prep_cancel_bet("2")
    assert prepped.headers == {
        "Content-Type": "application/json",
        "Authorization": "Key no_key",
        "Content-Length": "0",
    }
    assert prepped.body == None
    assert prepped.url == "https://manifold.markets/api/v0/bet/cancel/2"


def test_create_market_prepared():
    wrapper = api.APIWrapper("no_key")
    prepped = wrapper._prep_create_market(
        "BINARY",
        "Test question",
        "Some elaboration.",
        1659896688,
        tags=None,
        initialProb=50,
    )
    assert prepped.headers == {
        "Content-Type": "application/json",
        "Authorization": "Key no_key",
        "Content-Length": "226",
    }
    assert (
        prepped.body
        == b'{"outcomeType": "BINARY", "question": "Test question", "description": {"type": "doc", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Some elaboration."}]}]}, "closeTime": 1659896688, "initialProb": 50}'
    )


def test_me_prepared():
    wrapper = api.APIWrapper("no_key")
    prepped = wrapper._prep_me()
    assert prepped.headers == {
        "Content-Type": "application/json",
        "Authorization": "Key no_key",
    }
    assert prepped.body == None
    assert prepped.url == "https://manifold.markets/api/v0/me"


def test_resolve_market_prepared():
    wrapper = api.APIWrapper("no_key")
    prepped = wrapper._prep_resolve("1", "YES")
    assert prepped.headers == {
        "Content-Type": "application/json",
        "Authorization": "Key no_key",
        "Content-Length": "18",
    }
    assert prepped.body == b'{"outcome": "YES"}'


def test_sell_prepared():
    wrapper = api.APIWrapper("no_key")
    prepped = wrapper._prep_sell("1", "YES", 5)
    assert prepped.headers == {
        "Content-Type": "application/json",
        "Authorization": "Key no_key",
        "Content-Length": "31",
    }

    assert prepped.body == b'{"outcome": "YES", "shares": 5}'
