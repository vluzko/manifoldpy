"""Calibration across all resolved markets.
Currently only works for binary markets.
"""
import numpy as np

from matplotlib import pyplot as plt
from manifold import api, calibration


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


def calibration_at_close():
    binary = [x for x in api.get_markets() if isinstance(x, api.BinaryMarket)]
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


def calibration_at_start():
    binary = [x for x in api.get_full_markets_cached() if isinstance(x, api.BinaryMarket) and x.resolved]
    import pdb
    pdb.set_trace()
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
    calibration_at_start()
