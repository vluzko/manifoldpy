"""Tests of the GET endpoints of the API"""
import numpy as np
import random
import pytest
from requests import HTTPError
from manifoldpy import api


def test_get_bets():
    bets = api.get_bets(limit=100)
    assert len(bets) == 100


def test_get_comments():
    comments = api.get_comments(marketId="6qEWrk0Af7eWupuSWxQm")
    assert comments == []


def test_get_groups():
    groups = api.get_groups()
    assert len(groups) >= 480


def test_get_group_by_slug():
    slug = "uk-policies-by-the-next-election"
    group = api.get_group_by_slug(slug)
    assert group.id == "023SKlBd1yv7btKfwjHy"


def test_get_group_by_id():
    group = api.get_group_by_id("023SKlBd1yv7btKfwjHy")
    assert group.name == "UK policies by the next election"
    assert group.slug == "uk-policies-by-the-next-election"


def test_get_group_markets():
    markets = api.get_group_markets("023SKlBd1yv7btKfwjHy")
    assert len(markets) >= 26


def test_get_markets():
    markets = api.get_markets(limit=100)
    for market in markets:
        assert market.bets is None
        assert market.comments is None


def test_get_markets_limit():
    markets = api.get_markets(limit=3)
    assert len(markets) == 3


# Will be activated once the API is updated to deprecate previous get_market route.
# def test_get_market():
#     market = api.get_market("6qEWrk0Af7eWupuSWxQm")
#     assert market.bets is not None
#     assert market.comments is not None


def test_get_full_market():
    market = api.get_full_market("6qEWrk0Af7eWupuSWxQm")
    assert market.bets is not None
    assert market.comments is not None


def test_market_broken():
    # If this stops breaking, the API has been updated
    with pytest.raises(HTTPError):
        api.get_market("YVDsNCQWr7hUrAiFiKIV")


def test_get_free_response_market():
    market = api.get_market("kbCU0NTSe22jMWWwD4i5")
    assert (
        market.question
        == "When will 100 babies be born whose embryos were selected for genetic scores for intelligence?"
    )
    assert market.createdTime == 1656552954430
    assert isinstance(market, api.FreeResponseMarket)
    assert market.answers is not None
    assert len(market.answers) >= 5


def test_get_market_noisy():
    """Randomly sample k markets and check the data matches no matter how we get it"""
    k = 5
    # We filter for resolved markets because if they're not resolved then
    # the market can update in between the two API calls
    markets = [m for m in api.get_markets(limit=100) if m.isResolved]

    choices = random.sample(range(len(markets)), k)

    for i in choices:
        lite_market = markets[i]
        full_market_id = api.get_market(lite_market.id)
        full_market_slug = api.get_slug(lite_market.slug)
        assert full_market_id == full_market_slug


def test_by_slug():
    market = api.get_slug("will-any-model-pass-an-undergrad-pr")
    assert market.createdTime == 1657570778443
    assert (
        market.question
        == 'Will any model pass an "undergrad proofs exam" Turing test by 2027?'
    )


def test_get_user_by_name():
    user = api.get_user_by_name("vluzko")
    assert user.username == "vluzko"
    assert user.id == "acvO0NAsghTTgGjnsdwt94O44OT2"


def test_get_user_by_id():
    user = api.get_user_by_id("acvO0NAsghTTgGjnsdwt94O44OT2")
    assert user.username == "vluzko"
    assert user.id == "acvO0NAsghTTgGjnsdwt94O44OT2"


def test_get_users():
    users = api.get_users(limit=100)
    assert len(users) == 100


def test_get_all_users():
    api.get_all_users()


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


def test_get_all_bets():
    bets = api.get_all_bets("LiquidityBonusBot")
    assert len(bets) == 1056  # Not going to change because I forgot the password :D
