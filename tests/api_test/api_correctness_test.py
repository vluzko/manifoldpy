from typing import List, Set
from manifoldpy import api


def test_comment_correctness_small():
    json = api._get_comments(marketId="pBPJS5ebbd3QD3RVi8AN")
    expected_keys = {x.name for x in api.Comment.__attrs_attrs__}  # type: ignore
    actual_keys = {y for x in json for y in x}
    assert actual_keys.issubset(
        expected_keys
    ), f"Extra keys: {actual_keys - expected_keys}"


def test_comment_correctness_large():
    markets = api.get_markets(limit=100)
    expected_keys = {x.name for x in api.Comment.__attrs_attrs__}  # type: ignore
    always_present = set(expected_keys)
    for market in markets:
        json = api._get_comments(marketId=market.id)
        actual_keys = {y for x in json for y in x}
        assert actual_keys.issubset(
            expected_keys
        ), f"Extra keys: {actual_keys - expected_keys}"
        if len(json) > 0:
            always_present &= actual_keys
    print(f"Always present keys: {always_present}")


def check_markets(market_name: str, markets: List[dict], expected_keys: Set[str]):
    always_present = set(expected_keys)
    never_present = set(expected_keys)
    sometimes_present = set()
    for market in markets:
        actual_keys = set(market)
        assert actual_keys.issubset(expected_keys), f"Extra keys: {actual_keys - expected_keys}"

        always_present &= actual_keys
        never_present -= actual_keys
        sometimes_present |= actual_keys

    print(f"Always present keys on {market_name} markets: {always_present}")
    print(f"Never present keys on {market_name} markets: {never_present}")
    print(
        f"Sometimes present keys on {market_name} markets: {sometimes_present - always_present}"
    )
