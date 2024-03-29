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


def count_bets(cache: cache_utils.Cache) -> int:
    count = 0
    for _mkt_id, bets in cache["bets"].items():
        count += len(bets)
    return count


def patch_cache(func):
    def wrapper(monkeypatch, *args, **kwargs):
        monkeypatch.setattr(config, "JSON_CACHE_LOC", fake_cache)
        return func(monkeypatch, *args, **kwargs)

    return wrapper


@patch_cache
def test_cache_markets(monkeypatch):
    monkeypatch.setattr(api, "_get_all_markets", make_fake_mkts())
    res1 = cache_utils.update_lite_markets()
    res2 = json.loads(fake_cache.read_text())
    assert res1 == res2


@patch_cache
def test_cache_markets_repull(monkeypatch):
    monkeypatch.setattr(api, "_get_all_markets", make_fake_mkts())
    res1 = cache_utils.update_lite_markets()
    res2 = cache_utils.update_lite_markets()
    res3 = cache_utils.load_cache()
    assert len(res1["lite_markets"]) == 10
    assert res1 == res2 == res3


@patch_cache
def test_cache_bets(monkeypatch):
    monkeypatch.setattr(api, "_get_all_bets", make_fake_bets())
    res1 = cache_utils.update_bets()
    res2 = cache_utils.load_cache()

    assert res1 == res2


@patch_cache
def test_cache_bets_divisible(monkeypatch):
    monkeypatch.setattr(api, "_get_all_bets", make_fake_bets(1000))
    res1 = cache_utils.update_bets()
    res2 = cache_utils.load_cache()

    assert res1 == res2


@patch_cache
def test_cache_bets_repull(monkeypatch):
    monkeypatch.setattr(api, "_get_all_bets", make_fake_bets())
    res1 = cache_utils.update_bets()
    res2 = cache_utils.update_bets()
    res3 = cache_utils.load_cache()
    assert res1 == res2 == res3


@patch_cache
def test_get_full_markets(monkeypatch):
    monkeypatch.setattr(api, "_get_all_markets", make_fake_mkts())
    monkeypatch.setattr(api, "_get_all_bets", make_fake_bets(100))
    f = cache_utils.load_cache

    res = cache_utils.get_full_markets()
    assert len(res) == 10


@patch_cache
def test_backfill(_monkeypatch):
    res1 = cache_utils.backfill_bets(limit=10)
    res2 = cache_utils.backfill_bets(limit=10)
    assert count_bets(res1) == 10
    assert count_bets(res2) == 20


@patch_cache
def test_build_df(monkeypatch):
    monkeypatch.setattr(api, "_get_all_markets", make_fake_mkts())
    monkeypatch.setattr(api, "_get_all_bets", make_fake_bets(100))
    cache_utils.update_bets()
    cache_utils.update_lite_markets()
    df = cache_utils.load_binary_markets_as_df()

    # Honestly this is a nothing test, it's just here to run the code regularly.
    assert "creatorId" in df.columns


@fixture(autouse=True)
def run_after():
    yield
    fake_cache.unlink(missing_ok=True)
