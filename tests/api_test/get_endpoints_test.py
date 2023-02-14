"""Tests of the GET endpoints of the API"""
import numpy as np
import pytest
from api_test import SKIP_LONG

from manifoldpy import api


def test_get_bets():
    bets = api.get_bets(limit=100)
    assert len(bets) == 100


def test_get_bets_username():
    bets = api.get_bets(username="vluzko")
    # Number of bets I've made at time of test creation
    assert len(bets) >= 504


def test_get_bets_user_id():
    bets = api.get_bets(userId="acvO0NAsghTTgGjnsdwt94O44OT2")
    assert len(bets) >= 504


def test_get_bets_market_id():
    bets = api.get_bets(marketId="pBPJS5ebbd3QD3RVi8AN")
    assert len(bets) == 177


def test_get_bets_market_slug():
    bets = api.get_bets(marketSlug="will-mrna4157-go-to-phase-3-by-2025")
    assert len(bets) == 177


def test_get_bets_limit():
    bets = api.get_bets(limit=3)
    assert len(bets) == 3


def test_get_comments():
    comments = api.get_comments(marketId="pBPJS5ebbd3QD3RVi8AN")
    assert len(comments) == 6


def test_get_comments_slug():
    comments = api.get_comments(marketSlug="will-mrna4157-go-to-phase-3-by-2025")
    assert len(comments) == 6


def test_get_groups():
    groups = api.get_groups()
    assert len(groups) >= 480


def test_get_group_by_slug():
    slug = "technical-ai-timelines"
    group = api.get_group_by_slug(slug)
    assert group.id == "GbbX9U5pYnDeftX9lxUh"


def test_get_group_by_id():
    group = api.get_group_by_id("GbbX9U5pYnDeftX9lxUh")
    assert group.name == "Technical AI Timelines"
    assert group.slug == "technical-ai-timelines"


def test_get_group_markets():
    markets = api.get_group_markets("GbbX9U5pYnDeftX9lxUh")
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


def test_get_market():
    market = api.get_market("pBPJS5ebbd3QD3RVi8AN")
    assert market.bets is None
    assert market.comments is None


def test_get_full_market():
    market = api.get_full_market("pBPJS5ebbd3QD3RVi8AN")
    assert market.bets is not None
    assert market.comments is not None


def test_get_binary_market():
    market = api.get_market("pBPJS5ebbd3QD3RVi8AN")
    assert isinstance(market, api.BinaryMarket)


def test_get_free_response_market():
    market = api.get_slug("monthly-paper-search-1-which-ai-pap-57c34bfc81c7")
    assert isinstance(market, api.FreeResponseMarket)
    assert market.createdTime == 1665676857961
    assert market.answers is not None
    assert len(market.answers) == 10


def test_get_pseudo_numeric_market():
    market = api.get_slug("benchmark-gap-4-once-a-single-ai-mo")
    assert isinstance(market, api.PseudoNumericMarket)
    assert market.min == 0
    assert market.max == 240


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
    market: api.FreeResponseMarket = api.get_slug("after-how-many-unique-traders-will")  # type: ignore
    outcomes, times = market.outcome_history()

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


@SKIP_LONG
def test_get_all_markets():
    markets = api.get_all_markets()
    unique_ids = set(x.id for x in markets)
    assert len(markets) == len(unique_ids)


def test_get_all_markets_limit():
    markets = api.get_all_markets(limit=1005)
    unique_ids = set(x.id for x in markets)
    assert len(unique_ids) == 1005


def test_get_all_markets_json():
    markets = api._get_all_markets(limit=1005)
    unique_ids = set(x["id"] for x in markets)
    assert len(unique_ids) == 1005


def test_get_all_bets_limit():
    bets = api.get_all_bets(limit=1005)
    unique = set((x.id, x.createdTime) for x in bets)
    assert len(unique) == 1005


def test_get_all_bets_json():
    bets = api._get_all_bets(limit=1005)
    unique = set((x["id"], x["createdTime"]) for x in bets)
    assert len(unique) == 1005


def test_get_all_users_limit():
    users = api.get_all_users(limit=1005)
    unique = set(x.id for x in users)
    assert len(unique) == 1005


def test_weak_unstructure():
    b = api.get_bets(limit=1)
    m = api.get_markets(limit=1)[0]
    m.bets = b
    m.comments = []

    as_json = api.weak_unstructure(m)
    assert len(as_json["bets"]) == 1
    b2 = as_json["bets"][0]
    assert isinstance(b2, dict)
