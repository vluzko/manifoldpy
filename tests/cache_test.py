import json
from pathlib import Path

from pytest import fixture
from manifoldpy import cache_utils, config

fake_cache = Path(__file__).parent / ".full_markets.json"


def test_cache_markets(monkeypatch):
    monkeypatch.setattr(config, "JSON_CACHE_LOC", fake_cache)
    res1 = cache_utils.update_lite_markets(limit=10)
    res2 = json.loads(fake_cache.read_text())
    assert res1 == res2


def test_cache_markets_repull(monkeypatch):
    monkeypatch.setattr(config, "JSON_CACHE_LOC", fake_cache)
    res1 = cache_utils.update_lite_markets(limit=10)
    res2 = cache_utils.update_lite_markets(limit=10)
    res3 = cache_utils.load_cache()
    assert res1 == res2 == res3


def test_cache_bets(monkeypatch):
    monkeypatch.setattr(config, "JSON_CACHE_LOC", fake_cache)
    res1 = cache_utils.update_bets(limit=10)
    res2 = cache_utils.load_cache()

    assert res1 == res2


def test_cache_bets_repull(monkeypatch):
    monkeypatch.setattr(config, "JSON_CACHE_LOC", fake_cache)
    res1 = cache_utils.update_bets(limit=10)
    res2 = cache_utils.update_bets(limit=10)
    res3 = cache_utils.load_cache()
    assert res1 == res2 == res3


@fixture(autouse=True)
def run_after():
    yield
    fake_cache.unlink()
