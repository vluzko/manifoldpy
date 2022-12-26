"""Tests of the GET endpoints of the API"""
import numpy as np
import pytest
from manifoldpy import api
from api_test import SKIP_LONG


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
    markets = api.get_markets()
    for market in markets:
        assert market.bets is None
        assert market.comments is None


def test_get_markets_limit():
    markets = api.get_markets(limit=3)
    assert len(markets) == 3


def test_get_markets_after():
    markets = api.get_markets(limit=4)
    after = markets[-1].createdTime + 1
    after_markets = api.get_all_markets(after=after)
    assert markets[-2].id == after_markets[-1].id


# Will be activated once the API is updated to deprecate previous get_market route.
def test_get_market():
    market = api.get_market("6qEWrk0Af7eWupuSWxQm")
    assert market.bets is not None
    assert market.comments is not None


def test_get_full_market():
    market = api.get_full_market("6qEWrk0Af7eWupuSWxQm")
    assert market.bets is not None
    assert market.comments is not None


def test_get_binary_market():
    market = api.get_market("L4IuKRctNWewm6CjJGx4")
    assert isinstance(market, api.BinaryMarket)


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


def test_get_pseudo_numeric_market():
    market = api.get_market("z5Azjkk0pDw1C905REGd")
    assert isinstance(market, api.PseudoNumericMarket)
    assert market.min == -1
    assert market.max == 3


def test_get_multiple_choice_market():
    market = api.get_market("TFMgBCrTM5RLZUd95zRW")
    assert isinstance(market, api.MultipleChoiceMarket)


def test_id_slug_comparison():
    by_slug = api.get_slug("will-any-model-pass-an-undergrad-pr")
    by_id = api.get_market("Le040Y0ZGkyAYCIEnpA2")
    assert by_id == by_slug


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


def test_binary_probabilities():
    """Grabs a closed market and checks it
    Could break if the market ever gets deleted.
    """
    # Permalink: https://manifold.markets/guzey/will-i-create-at-least-one-more-pre
    market_id = "8Lt9ZTHCPCK58gtn0Y8n"
    market = api.get_full_market(market_id)
    times, probs = market.probability_history()
    assert len(times) == len(probs) == 23
    assert probs[0] == 0.33
    assert times[0] == market.createdTime

    assert np.isclose(probs[-1], 0.56, atol=0.01)
    assert times[-1] == 1652147977243


def test_free_response_outcomes():
    """Grabs a closed market and checks it
    Could break if the market ever gets deleted.
    """
    # market: api.FreeResponseMarket = api.get_market("kbCU0NTSe22jMWWwD4i5")  # type: ignore
    market: api.FreeResponseMarket = api.get_slug("after-how-many-unique-traders-will")  # type: ignore
    outcomes, times = market.outcome_history()
    # import pdb

    # pdb.set_trace()
    assert set(outcomes) == {
        "10",
        "1",
        "5",
        "250",
        "2",
        "3",
        "4",
        "0",
        "6",
        "7",
        "8",
        "9",
    }
    assert list(times) == [
        1666423162327,
        1666478042455,
        1666479166006,
        1666482554004,
        1666489679648,
        1666489699034,
        1666489751907,
        1666491644393,
        1666595438015,
        1666595479549,
        1666595524918,
        1666595532354,
    ]


def test_free_response_full_history():
    market: api.FreeResponseMarket = api.get_full_market("3SNUOKvgVzkSNYPgBdMX")  # type: ignore
    times, probabilities = market.full_history()
    assert len(times) == probabilities.shape[1]
    assert times[0] == 1666421235253
    assert times[-1] == 1666944777490
    assert np.isclose(probabilities[5, -1], 0.38642623)


def test_free_response_probabilities():
    """Grabs a closed market and checks it
    Could break if the market ever gets deleted.
    """
    market: api.FreeResponseMarket = api.get_full_market("3SNUOKvgVzkSNYPgBdMX")  # type: ignore
    times1, final_probs = market.probability_history()
    times2, full_probs = market.full_history()
    assert (times1 == times2).all()
    assert (full_probs[int(market.resolution)] == final_probs).all()  # type: ignore


def test_get_all_bets():
    bets = api.get_all_bets("LiquidityBonusBot")
    assert len(bets) == 1056  # Not going to change because I forgot the password :D


@SKIP_LONG
def test_get_all_markets():
    markets = api.get_all_markets()
    unique_ids = set(x.id for x in markets)
    assert len(markets) == len(unique_ids)


def test_get_all_markets_limit():
    markets = api.get_all_markets(limit=1005)
    unique_ids = set(x.id for x in markets)
    assert len(unique_ids) == 1005
    assert len(markets) == 1005
