from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from manifoldpy import api, cache_utils, calibration


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


def main():
    # Load markets
    markets = cache_utils.load_full_markets()
    # Filter for binary resolved markets
    binary = [m for m in markets if m.outcomeType == "BINARY" and m.isResolved == True]
    # Filter for long markets (30 days+)
    long = [
        m for m in binary if (m.closeTime - m.createdTime) > 1000 * 60 * 60 * 24 * 30
    ]
    df, histories = calibration.build_dataframe(long)  # type: ignore
    df["histories"] = [h[1] for h in histories]

    starts = np.array([h[0][0] for h in histories])
    ends = np.array([h[0][-1] for h in histories])
    midpoints = (starts + ends) * 0.5
    df["midway"] = calibration.probability_at_time(histories, midpoints)
    yes_markets = df[df["resolution"] == "YES"]
    no_markets = df[df["resolution"] == "NO"]

    f_yes = yes_markets[yes_markets["num_traders"] > 3]
    f_no = no_markets[no_markets["num_traders"] > 3]
    yes_probs = f_yes["midway"]
    no_probs = f_no["midway"]
    beta_means, beta_intervals = calibration.market_set_accuracy(yes_probs, no_probs)[
        "beta-binomial"
    ]
    plot_beta_binomial(beta_intervals, beta_means, decimals=1)


if __name__ == "__main__":
    main()
