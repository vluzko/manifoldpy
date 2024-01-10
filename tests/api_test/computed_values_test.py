"""Tests for the probability calculations"""
from pytest import fixture

from manifoldpy import api


@fixture
def lite_mkt():
    return api.BinaryMarket(
        **{
            "id": "none",
            "creatorUsername": "TestUserName",
            "creatorName": "Test Name",
            'slug': 'will-this-test-pass'
            "createdTime": 0,
            "question": "Will this test pass",
            "url": "none",
            "pool": {"NO": 38.461538461538474, "YES": 65},
            "volume": 15,
            "volume24Hours": 0,
            "outcomeType": "BINARY",
            "mechanism": "cpmm-1",
            "isResolved": False,
            "lastUpdatedTime": 0,
            "closeTime": 1,
            "creatorId": "none",
            "creatorAvatarUrl": "none",
            "p": 0.5,
            "totalLiquidity": 50,
            'uniqueBettorCount': 0,
            "probability": 0.3717472118959108,
        }
    )


def test_no_bets(lite_mkt):
    assert lite_mkt.num_traders() == 0
