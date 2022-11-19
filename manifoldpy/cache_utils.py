from collections import defaultdict
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
    lite_markets: Dict[str, Dict[str, Any]]
    bets: Dict[str, List[Dict[str, Any]]]


def load_cache():
    try:
        cache = json.load(config.JSON_CACHE_LOC.open("r"))
    except FileNotFoundError:
        cache = Cache(latest_market=0, latest_bet=0, lite_markets=[], bets={})
    return cache


def save_cache(cache: Cache):
    """Save the cache to disk"""
    json.dump(cache, config.JSON_CACHE_LOC.open("w"))


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
    cache["bets"] = bets_dict
    cache["latest_bet"] = max(
        max([b["createdTime"] for b in v]) for v in cache["bets"].values()
    )
    save_cache(cache)


def get_full_markets() -> List[api.Market]:
    """Get all full markets, and cache the results."""

    raise NotImplementedError
