import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, TypedDict

from manifoldpy import api, config


class Cache(TypedDict):

    latest_market: int
    latest_bet: int
    lite_markets: Dict[str, Dict[str, Any]]
    # Market ID -> {bet ID -> bet}
    bets: Dict[str, Dict[str, Dict[str, Any]]]


def load_cache() -> Cache:
    try:
        with config.JSON_CACHE_LOC.open("r") as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = Cache(latest_market=0, latest_bet=0, lite_markets={}, bets={})
    return cache


def save_cache(cache: Cache):
    """Save the cache to disk"""
    # Two steps because json.dump will write partial output if there's an error partway through
    encoded = json.dumps(cache)
    with config.JSON_CACHE_LOC.open("w") as f:
        f.write(encoded)


def update_lite_markets():
    cache = load_cache()

    lite_markets = api._get_all_markets(after=cache["latest_market"])
    cache["lite_markets"].update({m["id"]: m for m in lite_markets})  # type: ignore
    cache["latest_market"] = max(
        m["createdTime"] for m in cache["lite_markets"].values()
    )
    save_cache(cache)
    return cache


def add_bets_to_cache(cache, bets):
    for b in bets:
        # TODO: Implement this to give unique bet IDs, requires migration first
        # key = (b["id"], b["createdTime"])
        key = b["id"]
        if b["contractId"] in cache["bets"]:
            cache["bets"][b["contractId"]][key] = b
        else:
            cache["bets"][b["contractId"]] = {key: b}
    return cache


def backfill_bets(limit: int = 100000):
    cache = load_cache()

    if len(cache["bets"]) == 0:
        earliest_bet = None
    else:
        first_bets = (
            min((b for b in v.values()), key=lambda x: x["createdTime"])
            for v in cache["bets"].values()
        )
        earliest_bet = min(first_bets, key=lambda x: x["createdTime"])["id"]
    older_bets = api._get_all_bets(before_id=earliest_bet, limit=limit)
    cache = add_bets_to_cache(cache, older_bets)
    save_cache(cache)
    return cache


def update_bets():
    cache = load_cache()

    bets = api._get_all_bets(after=cache["latest_bet"])
    cache = add_bets_to_cache(cache, bets)

    last_bets = (
        max((b["createdTime"] for b in v.values())) for v in cache["bets"].values()
    )
    cache["latest_bet"] = max(last_bets)
    save_cache(cache)
    return cache


def get_full_markets() -> List[api.Market]:
    """Get all full markets, and cache the results."""
    update_lite_markets()
    update_bets()
    cache = load_cache()
    markets = {k: api.Market.from_json(v) for k, v in cache["lite_markets"].items()}
    for k, market in markets.items():
        if k in cache["bets"]:
            bets = cache["bets"][k]
            markets[k].bets = [api.weak_structure(b, api.Bet) for b in bets.values()]
        else:
            market.bets = []

        market.comments = []
        # market.bets = []
    # for k, bets in cache["bets"].items():
    #     markets[k].bets = [api.weak_structure(b, api.Bet) for b in bets.values()]
    return list(markets.values())
