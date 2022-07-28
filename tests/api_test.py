import numpy as np
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
    """Grabs a closed market and checks it
    Could break if the market ever gets deleted.
    """
    # Permalink: https://manifold.markets/guzey/will-i-create-at-least-one-more-pre
    market_id = '8Lt9ZTHCPCK58gtn0Y8n'
    market = api.get_market(market_id)
    times, probs = market.get_updates()
    assert len(times) == len(probs) == 23
    assert probs[0] == 0.33
    assert times[0] == market.createdTime

    assert np.isclose(probs[-1], 0.56, atol=0.01)
    assert times[-1] == 1652147977243
