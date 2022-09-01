# manifold-markets-python
[![CircleCI](https://circleci.com/gh/vluzko/manifold-markets-python.svg?style=shield)](https://circleci.com/gh/vluzko/manifold-markets-python)
[![Documentation Status](https://readthedocs.org/projects/manifold-markets-python/badge/?version=latest)](https://manifold-markets-python.readthedocs.io/en/latest/?badge=latest)


Tools for analyzing [Manifold Markets](https://manifold.markets/home) data. Currently has bindings for their API, and code for computing various accuracy metrics (Brier score, log score, calibration).\


[Full Documentation](https://manifold-markets-python.readthedocs.io/en/latest/).


## Installation
The package is on PyPI as [`manifoldpy`](https://pypi.org/project/manifoldpy/). It can be installed with:
```
pip install manifoldpy
```
Alternatively, clone this repo and install with pip:
```
git clone https://github.com/vluzko/manifold-markets-python.git
cd manifold-markets-python
pip install .
```

## Basic Usage
Get a list containing every market:
```
from manifoldpy import api
markets = api.get_all_markets()
```

Manifold also has a POST API that lets you make/resolve/bet on markets. This requires you to have an API key (which you can generate on your [Manifold profile page](https://manifold.markets/profile)). Here's an example for making a bet on a binary market:
```
from manifoldpy import api
# Amount to spend
amount = 100
# The ID of the market to bet on
contract_id = "8Lt9ZTHCPCK58gtn0Y8n"
# The outcome you want to bet on.
outcome = "YES"
api.make_bet(YOUR_API_KEY, amount, contract_id, outcome)
```
If you want to avoid passing the key every time, you can instead create an `APIWrapper` and use that:
```
from manifoldpy import api
wrapper = api.APIWrapper(YOUR_API_KEY)
amount = 100
contract_id = "8Lt9ZTHCPCK58gtn0Y8n"
outcome = "YES"
wrapper.make_bet(amount, contract_id, outcome)
```
