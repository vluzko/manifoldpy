"""Tests of the GET endpoints of the API"""
import numpy as np
import pytest

from manifoldpy import api


def test_get_bets():
    bets = api.get_bets(limit=100)
    assert len(bets) == 100
    for i, bet in enumerate(bets[:-1]):
        assert bet.createdTime >= bets[i + 1].createdTime


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


@pytest.mark.skip(reason="This endpoint breaks upstream.")
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


def test_get_free_response_market():
    market = api.get_slug("monthly-paper-search-1-which-ai-pap-57c34bfc81c7")
    assert market.createdTime == 1665676857961
    assert market.answers is not None
    assert len(market.answers) == 11


def test_get_pseudo_numeric_market():
    market = api.get_slug("benchmark-gap-4-once-a-single-ai-mo")
    assert market.min == 0
    assert market.max == 240


def test_get_multiple_choice_market():
    market = api.get_market("TFMgBCrTM5RLZUd95zRW")


def test_get_bountied_market():
    market = api.get_market("ofhEyRqeO8AXY6SNteOO")


def test_search_markets():
    markets = api.search_markets(["will"])
    assert len(markets) <= 100


def test_get_positions():
    api.get_market_positions("pBPJS5ebbd3QD3RVi8AN")


def test_get_user_positions():
    user_id = "acvO0NAsghTTgGjnsdwt94O44OT2"
    positions = api.get_market_positions("pBPJS5ebbd3QD3RVi8AN", userId=user_id)
    for pos in positions:
        assert pos.userId == user_id


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


def test_get_full_data():
    m = api.get_markets(limit=1)[0]
    m.get_full_data()
    assert m.bets is not None
    assert m.comments is not None
