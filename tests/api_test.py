from manifold import api


def test_get_markets():
    markets = api.get_markets()
    for market in markets:
        assert market.bets is None
        assert market.comments is None


def test_get_market():
    market = api.get_market('6qEWrk0Af7eWupuSWxQm')
    assert market.bets is not None
    assert market.comments is not None
