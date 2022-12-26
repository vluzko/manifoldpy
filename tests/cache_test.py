import json
from pathlib import Path

from pytest import fixture
from manifoldpy import cache_utils, config

fake_cache = Path(__file__).parent / ".full_markets.json"


def test_cache_markets(monkeypatch):
    monkeypatch.setattr(config, "JSON_CACHE_LOC", fake_cache)
    cache_utils.update_lite_markets(limit=10)
    res1 = cache_utils.load_cache()
    res2 = json.loads(fake_cache.read_text())
    assert res1 == res2


# def test_cache_bets():
#     raise NotImplementedError


@fixture(autouse=True)
def run_after():
    yield
    fake_cache.unlink()
