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


def test_get_probabilities():
    market_id = '8Lt9ZTHCPCK58gtn0Y8n'
    market = api.get_market(market_id)
    market.get_updates()
    # markets = api.get_markets()
    # for market in markets:
    #     full = api.get_market(market.id)
    #     if isinstance(market, api.BinaryMarket) and len(full.bets) > 10:
    #         import pdb
    #         pdb.set_trace()
    import pdb
    pdb.set_trace()
