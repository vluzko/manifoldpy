import numpy as np
import random
import pytest
from requests import HTTPError
from manifoldpy import api


def test_get_user_by_name():
    user = api.get_user_by_name("vluzko")
    assert user.username == "vluzko"
    assert user.id == "acvO0NAsghTTgGjnsdwt94O44OT2"


def test_get_user_by_id():
    user = api.get_user_by_id("acvO0NAsghTTgGjnsdwt94O44OT2")
    assert user.username == "vluzko"
    assert user.id == "acvO0NAsghTTgGjnsdwt94O44OT2"


def test_get_users():
    api.get_users()


def test_get_markets():
    markets = api.get_markets()
    for market in markets:
        assert market.bets is None
        assert market.comments is None


def test_get_markets_limit():
    markets = api.get_markets(limit=3)
    assert len(markets) == 3


def test_get_market():
    market = api.get_market("6qEWrk0Af7eWupuSWxQm")
    assert market.bets is not None
    assert market.comments is not None


def test_by_slug():
    market = api.get_slug("will-any-model-pass-an-undergrad-pr")
    assert market.createdTime == 1657570778443
    assert (
        market.question
        == 'Will any model pass an "undergrad proofs exam" Turing test by 2027?'
    )


def test_get_bets():
    api.get_bets()


def test_broken():
    # If this stops breaking, the API has been updated
    with pytest.raises(HTTPError):
        market = api.get_market("YVDsNCQWr7hUrAiFiKIV")


def test_get_free_response():
    market = api.get_market("kbCU0NTSe22jMWWwD4i5")
    assert (
        market.question
        == "When will 100 babies be born whose embryos were selected for genetic scores for intelligence?"
    )
    assert market.createdTime == 1656552954430
    assert len(market.answers) >= 5


def test_get_market_noisy():
    """Randomly sample k markets and check the data matches no matter how we get it"""
    k = 15
    markets = api.get_markets()
    choices = random.sample(range(len(markets)), k)

    for i in choices:
        lite_market = markets[i]
        full_market_id = api.get_market(lite_market.id)
        full_market_slug = api.get_slug(lite_market.slug)
        assert full_market_id == full_market_slug
        assert full_market_id.question == lite_market.question


def test_get_probabilities():
    """Grabs a closed market and checks it
    Could break if the market ever gets deleted.
    """
    # Permalink: https://manifold.markets/guzey/will-i-create-at-least-one-more-pre
    market_id = "8Lt9ZTHCPCK58gtn0Y8n"
    market = api.get_market(market_id)
    times, probs = market.probability_history()
    assert len(times) == len(probs) == 23
    assert probs[0] == 0.33
    assert times[0] == market.createdTime

    assert np.isclose(probs[-1], 0.56, atol=0.01)
    assert times[-1] == 1652147977243


def test_get_markets():
    markets = api.get_markets()
    for market in markets:
        assert market.bets is None
        assert market.comments is None

def test_get_all_bets():
    bets = api.get_all_bets('LiquidityBonusBot')
    assert len(bets) == 1056 # Not going to change because I forgot the password :D


def test_me_prepared():
    wrapper = api.APIWrapper("no_key")
    prepped = wrapper._prep_me()
    assert prepped.headers == {
        "Content-Type": "application/json",
        "Authorization": "Key no_key",
    }
    assert prepped.body == None
    assert prepped.url == "https://manifold.markets/api/v0/me"


def test_bet_prepared():
    wrapper = api.APIWrapper("no_key")
    prepped = wrapper._prep_bet(10, "1", "YES")
    assert prepped.headers == {
        "Content-Type": "application/json",
        "Authorization": "Key no_key",
        "Content-Length": "51",
    }
    assert prepped.body == b'{"amount": 10, "contractId": "1", "outcome": "YES"}'


def test_cancel_prepared():
    wrapper = api.APIWrapper("no_key")
    prepped = wrapper._prep_cancel("2")
    assert prepped.headers == {
        "Content-Type": "application/json",
        "Authorization": "Key no_key",
        "Content-Length": "0",
    }
    assert prepped.body == None
    assert prepped.url == "https://manifold.markets/api/v0/bet/cancel/2"


def test_create_market_prepared():
    wrapper = api.APIWrapper("no_key")
    prepped = wrapper._prep_create(
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