import attr
import pickle
import requests
import json
from time import time
from typing import Dict, Any, List, TypedDict
from manifoldpy import api, config


class Cache(TypedDict):
    latest_market: int
    latest_bet: int
    lite_markets: Dict[int, Dict[str, Any]]
    bets: Dict[int, List[Dict[str, Any]]]


def load_cache():
    try:
        cache = json.load(config.JSON_CACHE_LOC.open("r"))
    except FileNotFoundError:
        cache = Cache(latest_market=0, latest_bet=0, lite_markets=[], bets={})
    return cache


def update_lite_markets():
    cache = load_cache()
    lite_markets = api.get_all_markets(after=cache["latest_market"])
    json = [api.weak_unstructure(m) for m in lite_markets]


def get_full_markets(
    reset_cache: bool = False, cache_every: int = 100
) -> List[api.Market]:
    """Get all full markets, and cache the results.

    Args:
        reset_cache: Whether or not to overwrite the existing cache
        cache_every: How frequently to cache the updated markets.
    """
    if not reset_cache:
        full_markets = load_cache()
    else:
        full_markets = {}

    lite_markets = {x.id: x for x in api.get_all_markets()}
    print(f"Found {len(lite_markets)} lite markets")

    cached_ids = {x["market"].id for x in full_markets.values()}
    missing_markets = set(lite_markets.keys()) - cached_ids
    print(f"Need to fetch {len(missing_markets)} new markets.")
    missed = []
    for i, lmarket_id in enumerate(missing_markets):
        try:
            lite_market = lite_markets[lmarket_id]
            full_market = lite_market.get_full_data()
            full_markets[full_market.id] = {
                "market": full_market,
                "cache_time": time(),
            }
        # If we get an HTTP Error, just skip that market
        except requests.HTTPError:
            missed.append(lmarket_id)

        if i % cache_every == 0:
            print(f"Fetched {i} markets, {len(missing_markets) - i} remaining")
            pickle.dump(full_markets, config.CACHE_LOC.open("wb"))
    pickle.dump(full_markets, config.CACHE_LOC.open("wb"))
    market_list = [x["market"] for x in full_markets.values()]
    missed_ids = "\n".join(missed)
    print(f"Could not get {len(missed)} markets. Missing markets:\n {missed_ids}")
    return market_list


def update_cached(cache_every: int = 100):
    """Update all unresolved markets"""
    cache = load_cache()

    unresolved = [x["market"] for x in cache.values() if not x["market"].isResolved]
    print(f"Found {len(unresolved)} unresolved markets")
    missed = []
    for i, market in enumerate(unresolved):
        try:
            full_market = api.get_full_market(market.id)
            cache[full_market.id] = {
                "market": full_market,
                "cache_time": time(),
            }
        # If we get an HTTP Error, just skip that market
        except requests.HTTPError:
            missed.append(market.id)

        if i % cache_every == 0:
            print(f"Fetched {i} markets, {len(unresolved) - i} remaining")
            pickle.dump(cache, config.CACHE_LOC.open("wb"))
    missed_ids = "\n".join(missed)
    print(f"Could not get {len(missed)} markets. Missing markets:\n {missed_ids}")
    pickle.dump(cache, config.CACHE_LOC.open("wb"))
