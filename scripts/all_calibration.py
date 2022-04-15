"""Calibration across all resolved markets.
Currently only works for binary markets.
"""
import numpy as np
import pickle
import signal
import sys

from functools import partial
from matplotlib import pyplot as plt
from manifold import api, calibration, markets
from os import environ
from pathlib import Path
from time import time

# TODO: Move the cache code into the main library.
DATA = Path(environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "manifold"
DATA.mkdir(exist_ok=True, parents=True)
CACHE_LOC = DATA / "full_markets.pkl"


def cache_objs(full_markets, signum, frame):
    pickle.dump(full_markets, CACHE_LOC.open('wb'))
    sys.exit(0)


def cache_full_markets():
    try:
        full_markets = pickle.load(CACHE_LOC.open('rb'))
    except FileNotFoundError:
        full_markets = {}
    signal.signal(signal.SIGINT, partial(cache_objs, full_markets))

    lite_markets = api.get_markets()
    print(f"Fetching {len(lite_markets)} markets")
    try:
        for lmarket in lite_markets:
            if lmarket.id in full_markets:
                continue
            else:
                full_market = api.get_market(lmarket.id)
                full_markets[full_market.id] = {"market": full_market, "cache_time": time()}
    # Happens sometimes, probably a rate limit on their end, just restart the script.
    except ConnectionResetError:
        pass
    pickle.dump(full_markets, CACHE_LOC.open('wb'))


def plot_calibration(c_table: np.ndarray, bins: np.ndarray):
    _, ax = plt.subplots()
    ax.scatter(bins, c_table)
    # Perfect calibration line
    l = np.arange(0, bins.max(), 0.0001)
    ax.scatter(l, l, color="green", s=0.01, label="Perfect calibration")

    ax.set_xticks(np.arange(0, 1+1/10, 1/10))
    ax.set_xlabel("Market probability")
    ax.set_yticks(np.arange(0, 1 + 1/10, 1/10))
    ax.set_ylabel("Empirical probability")

    plt.show()


def plot_beta_binomial(upper_lower: np.ndarray, means: np.ndarray, decimals):
    _, ax = plt.subplots()
    num_bins = 10**decimals
    x_axis = np.arange(0, 1+1/num_bins, 1/num_bins)
    ax.scatter(x_axis, means, color="blue")
    ax.scatter(x_axis, upper_lower[:, 0], color="black", marker="_")
    ax.scatter(x_axis, upper_lower[:, 1], color="black", marker="_")
    plt.vlines(x_axis, upper_lower[:, 0], upper_lower[:, 1], color="black")

    ax.set_xticks(np.arange(0, 1+1/10, 1/10))
    ax.set_xlabel("Market probability")
    ax.set_yticks(np.arange(0, 1 + 1/10, 1/10))
    ax.set_ylabel('Beta binomial means and 0.95 intervals')

    l = np.arange(0, x_axis.max(), 0.0001)
    ax.scatter(l, l, color="green", s=0.01, label="Perfect calibration")
    plt.show()


def main():
    binary = [x for x in api.get_markets() if isinstance(x, markets.BinaryMarket)]
    brier_score = calibration.brier_score(binary)
    log_score = calibration.log_score(binary)
    print(f"\
        Brier score: {brier_score}\n\
        Log score:   {log_score}"
    )
    # Calibration with 100 bins
    one_percent = calibration.binary_calibration(binary, decimals=2)
    # plot_calibration(one_percent, bins=np.arange(0, 1+ 1/100, 1/100))

    # Calibration with 10 bins
    ten_percent = calibration.binary_calibration(binary, decimals=1)
    # plot_calibration(ten_percent, bins=np.arange(0, 1 + 1/10, 1/10))

    # Calibration when we model each bin as with a beta binomial model
    beta_interval, beta_means = calibration.beta_binomial_calibration(binary, decimals=1)
    plot_beta_binomial(beta_interval, beta_means, decimals=1)


if __name__ == "__main__":
    main()
