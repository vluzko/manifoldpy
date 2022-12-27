import json
from pathlib import Path

from pytest import fixture

from manifoldpy import api, cache_utils, config

fake_cache = Path(__file__).parent / ".full_markets.json"


def make_fake_mkts(n: int = 10):
    f = api._get_all_markets

    def fake_mkts(*args, **kwargs):
        kwargs["limit"] = n
        return f(*args, **kwargs)

    return fake_mkts


def make_fake_bets(n: int = 10):
    f = api._get_all_bets

    def fake_bets(*args, **kwargs):
        kwargs["limit"] = n
        return f(*args, **kwargs)

    return fake_bets


def test_cache_markets(monkeypatch):
    monkeypatch.setattr(config, "JSON_CACHE_LOC", fake_cache)
    monkeypatch.setattr(api, "_get_all_markets", make_fake_mkts())
    res1 = cache_utils.update_lite_markets()
    res2 = json.loads(fake_cache.read_text())
    assert res1 == res2


def test_cache_markets_repull(monkeypatch):
    monkeypatch.setattr(config, "JSON_CACHE_LOC", fake_cache)
    monkeypatch.setattr(api, "_get_all_markets", make_fake_mkts())
    res1 = cache_utils.update_lite_markets()
    res2 = cache_utils.update_lite_markets()
    res3 = cache_utils.load_cache()
    assert len(res1["lite_markets"]) == 10
    assert res1 == res2 == res3


def test_cache_bets(monkeypatch):
    monkeypatch.setattr(config, "JSON_CACHE_LOC", fake_cache)
    monkeypatch.setattr(api, "_get_all_bets", make_fake_bets())
    res1 = cache_utils.update_bets()
    res2 = cache_utils.load_cache()

    assert res1 == res2


def test_cache_bets_divisible(monkeypatch):
    monkeypatch.setattr(config, "JSON_CACHE_LOC", fake_cache)
    monkeypatch.setattr(api, "_get_all_bets", make_fake_bets(1000))
    res1 = cache_utils.update_bets()
    res2 = cache_utils.load_cache()

    assert res1 == res2


def test_cache_bets_repull(monkeypatch):
    monkeypatch.setattr(config, "JSON_CACHE_LOC", fake_cache)
    monkeypatch.setattr(api, "_get_all_bets", make_fake_bets())
    res1 = cache_utils.update_bets()
    res2 = cache_utils.update_bets()
    res3 = cache_utils.load_cache()
    assert res1 == res2 == res3


@fixture(autouse=True)
def run_after():
    yield
    fake_cache.unlink()
