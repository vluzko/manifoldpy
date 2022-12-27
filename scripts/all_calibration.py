"""Calibration across all resolved markets.
Currently only works for binary markets.
"""
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from manifoldpy import api, calibration, config


def plot_beta_binomial(
    upper_lower: np.ndarray, means: np.ndarray, decimals
):  # pragma: no cover
    _, ax = plt.subplots()
    num_bins = 10**decimals
    x_axis = np.arange(0, 1 + 1 / num_bins, 1 / num_bins)
    ax.scatter(x_axis, means, color="blue")
    ax.scatter(x_axis, upper_lower[:, 0], color="black", marker="_")  # type: ignore
    ax.scatter(x_axis, upper_lower[:, 1], color="black", marker="_")  # type: ignore
    plt.vlines(x_axis, upper_lower[:, 0], upper_lower[:, 1], color="black")

    ax.set_xticks(np.arange(0, 1 + 1 / 10, 1 / 10))
    ax.set_xlabel("Market probability")
    ax.set_yticks(np.arange(0, 1 + 1 / 10, 1 / 10))
    ax.set_ylabel("Beta binomial means and 0.95 intervals")

    l = np.arange(0, x_axis.max(), 0.0001)
    ax.scatter(l, l, color="green", s=0.01, label="Perfect calibration")
    plt.show()


def plot_calibration(
    c_table: np.ndarray, bins: np.ndarray, path: Optional[Path] = None
):  # pragma: no cover
    _, ax = plt.subplots()
    ax.scatter(bins, c_table, label="Market calibration")
    # Perfect calibration line
    l = np.arange(0, bins.max(), 0.0001)
    ax.scatter(l, l, color="green", s=0.01, label="Perfect calibration")

    ax.set_xticks(np.arange(0, 1 + 1 / 10, 1 / 10))
    ax.set_xlabel("Market probability")
    ax.set_yticks(np.arange(0, 1 + 1 / 10, 1 / 10))
    ax.set_ylabel("Empirical probability")
    ax.legend()

    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def group_calibration_at_close():
    market_cache = pickle.load(config.CACHE_LOC.open("rb"))
    full_markets = [x["market"] for x in market_cache.values()]
    binary = [
        m for m in full_markets if isinstance(m, api.BinaryMarket) and m.isResolved
    ]
    df, _histories = calibration.build_dataframe(binary)  # type: ignore
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
        res = calibration.binary_calibration(yes_markets["final"], no_markets["final"])[  # type: ignore
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
    df, histories = calibration.build_dataframe(binary)  # type: ignore
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
    res = calibration.market_set_accuracy(yes_probs, no_probs)  # type: ignore

    # Calibration at end
    yes_probs = yes_markets["final"]
    no_probs = no_markets["final"]
    res = calibration.market_set_accuracy(yes_probs, no_probs)  # type: ignore

    # Calibration at midpoint
    f_yes = yes_markets[yes_markets["num_traders"] > 3]
    f_no = no_markets[no_markets["num_traders"] > 3]
    yes_probs = f_yes["midway"]
    no_probs = f_no["midway"]
    res = calibration.market_set_accuracy(yes_probs, no_probs)  # type: ignore
    midway_10 = res["10% calibration"]
    plot_calibration(
        midway_10,
        calibration.perfect_calibration(1),
        Path(__file__).parent.parent / "docs" / "midway_calibration.png",
    )


if __name__ == "__main__":
    overall_calibration()
