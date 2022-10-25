"""Calibration across all resolved markets.
Currently only works for binary markets.
"""
import numpy as np
import pickle

from matplotlib import pyplot as plt
from manifoldpy import api, calibration, config


def group_calibration_at_close():
    market_cache = pickle.load(config.CACHE_LOC.open("rb"))
    full_markets = [x["market"] for x in market_cache.values()]
    binary = [
        m for m in full_markets if isinstance(m, api.BinaryMarket) and m.isResolved
    ]
    df, histories = calibration.build_dataframe(binary)
    # Group calibration
    groups = calibration.markets_by_group(df)
    results = {}
    for group_name, f in groups.items():
        markets = df[f]
        if len(markets) < 10:
            continue
        yes_markets = markets[markets["resolution"] == "YES"]
        yes_count = len(
            yes_markets[(yes_markets["final"] < 0.95) & (yes_markets["final"] > 0.85)]
        )
        no_markets = markets[markets["resolution"] == "NO"]
        no_count = len(
            no_markets[(no_markets["final"] < 0.95) & (no_markets["final"] > 0.85)]
        )
        if yes_count + no_count < 10:
            continue
        res = calibration.binary_calibration(yes_markets["final"], no_markets["final"])[
            -2
        ]
        results[group_name] = res
    print(results)


def overall_calibration():
    """Evaluate calibration at start, end, midway, and for individual groups"""
    market_cache = pickle.load(config.CACHE_LOC.open("rb"))
    full_markets = [x["market"] for x in market_cache.values()]
    binary = [
        m for m in full_markets if isinstance(m, api.BinaryMarket) and m.isResolved
    ]
    df, histories = calibration.build_dataframe(binary)
    # Probabilities at the halfway point
    starts = np.array([h[0][0] for h in histories])
    ends = np.array([h[0][-1] for h in histories])
    midpoints = (starts + ends) * 0.5
    df["midway"] = calibration.probability_at_time(histories, midpoints)

    yes_markets = df[df["resolution"] == "YES"]
    no_markets = df[df["resolution"] == "NO"]

    # Calibration at start
    yes_probs = yes_markets["start"]
    no_probs = no_markets["start"]
    res = calibration.market_set_accuracy(yes_probs, no_probs)

    # Calibration at end
    yes_probs = yes_markets["final"]
    no_probs = no_markets["final"]
    res = calibration.market_set_accuracy(yes_probs, no_probs)

    # Calibration at midpoint
    yes_probs = yes_markets["midway"]
    no_probs = no_markets["midway"]
    res = calibration.market_set_accuracy(yes_probs, no_probs)


if __name__ == "__main__":
    group_calibration_at_close()
