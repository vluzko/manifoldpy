import json
from collections import defaultdict
from typing import Dict, Any, List, TypedDict
from manifoldpy import api, config


class Cache(TypedDict):
    latest_market: int
    latest_bet: int
    lite_markets: Dict[str, Dict[str, Any]]
    bets: Dict[str, List[Dict[str, Any]]]


def load_cache():
    try:
        with config.JSON_CACHE_LOC.open("r") as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = Cache(latest_market=0, latest_bet=0, lite_markets={}, bets={})
    return cache


def save_cache(cache: Cache):
    """Save the cache to disk"""
    with config.JSON_CACHE_LOC.open("w") as f:
        json.dump(cache, f)


def update_lite_markets():
    cache = load_cache()
    lite_markets = api.get_all_markets(after=cache["latest_market"])
    cache["lite_markets"].update({m.id: api.weak_unstructure(m) for m in lite_markets})
    cache["latest_market"] = max(
        m["createdTime"] for m in cache["lite_markets"].values()
    )
    save_cache(cache)


def update_bets():
    cache = load_cache()
    bets = api.get_all_bets(after=cache["latest_bet"])
    bets_dict = defaultdict(list)
    for b in bets:
        bets_dict[b.contractId].append(api.weak_unstructure(b))
    cache["bets"].update(bets_dict)
    cache["latest_bet"] = max(
        max([b["createdTime"] for b in v]) for v in cache["bets"].values()
    )
    save_cache(cache)


def get_full_markets() -> List[api.Market]:
    """Get all full markets, and cache the results."""
    update_lite_markets()
    update_bets()
    cache = load_cache()
    markets = {k: api.Market.from_json(v) for k, v in cache["lite_markets"].items()}
    for k, bets in cache["bets"].items():
        markets[k].bets = [api.weak_structure(b, api.Bet) for b in bets]
    return list(markets.values())
