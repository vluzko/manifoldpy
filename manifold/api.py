import requests

from typing import List
from manifold.markets import Market, BinaryMarket, MultiMarket


ALL_MARKETS_URL = "https://manifold.markets/api/v0/markets"
SINGLE_MARKET_URL = "https://manifold.markets/api/v0/market/{}"


def get_markets() -> List[Market]:
    json = requests.get(ALL_MARKETS_URL).json()

    # If this fails, the code is out of date.
    all_mechanisms = {x["mechanism"] for x in json}
    assert all_mechanisms == {"cpmm-1", "dpm-2"}

    markets = [BinaryMarket.from_json(x) if 'probability' in x else MultiMarket.from_json(x) for x in json]

    return markets


def get_market(market_id: str) -> Market:
    market = requests.get(SINGLE_MARKET_URL.format(market_id)).json()
    if "probability" in market:
        return BinaryMarket.from_json(market)
    else:
        return MultiMarket.from_json(market)


def get_full_markets() -> List[Market]:
    """Get all markets, including bets and comments.
    Not part of the API, but handy. Takes a while to run.
    """
    markets = get_markets()

    return [get_market(x.id) for x in markets]
