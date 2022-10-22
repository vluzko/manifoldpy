"""Calibration across all resolved markets.
Currently only works for binary markets.
"""
import numpy as np
import pickle

from matplotlib import pyplot as plt
from manifoldpy import api, calibration, config


def plot_calibration(c_table: np.ndarray, bins: np.ndarray):
    _, ax = plt.subplots()
    ax.scatter(bins, c_table)
    # Perfect calibration line
    l = np.arange(0, bins.max(), 0.0001)
    ax.scatter(l, l, color="green", s=0.01, label="Perfect calibration")

    ax.set_xticks(np.arange(0, 1 + 1 / 10, 1 / 10))
    ax.set_xlabel("Market probability")
    ax.set_yticks(np.arange(0, 1 + 1 / 10, 1 / 10))
    ax.set_ylabel("Empirical probability")

    plt.show()


def run_bundle():
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
    print("Start")
    yes_probs = yes_markets["start"]
    no_probs = no_markets["start"]
    calibration.overall_calibration(yes_probs, no_probs)
    print()

    # Calibration at end
    print("End")
    yes_probs = yes_markets["final"]
    no_probs = no_markets["final"]
    calibration.overall_calibration(yes_probs, no_probs)
    print()

    # Calibration at midpoint
    print("Midpoint")
    yes_probs = yes_markets["midway"]
    no_probs = no_markets["midway"]
    calibration.overall_calibration(yes_probs, no_probs)


if __name__ == "__main__":
    run_bundle()
