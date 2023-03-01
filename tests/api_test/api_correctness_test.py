from api_test import SKIP_LONG

from manifoldpy import api


@SKIP_LONG
def test_mechanisms():
    markets = api.get_all_markets()
    all_mechanisms = {x.mechanism for x in markets}
    assert all_mechanisms <= {"cpmm-1", "dpm-2"}


def test_comment_correctness_small():
    json = api._get_comments(marketId="pBPJS5ebbd3QD3RVi8AN")
    expected_keys = {x.name for x in api.Comment.__attrs_attrs__}  # type: ignore
    actual_keys = {y for x in json for y in x}
    assert actual_keys.issubset(expected_keys)


def test_comment_correctness_large():
    markets = api.get_markets(limit=100)
    expected_keys = {x.name for x in api.Comment.__attrs_attrs__}  # type: ignore
    for market in markets:
        json = api._get_comments(marketId=market.id)
        actual_keys = {y for x in json for y in x}
        assert actual_keys.issubset(expected_keys)
