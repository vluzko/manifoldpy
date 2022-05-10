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


def calibration_at_close():
    binary = [x for x in api.get_markets() if isinstance(x, api.BinaryMarket)]
    yes_probs, no_probs = calibration.extract_binary_probabilities(binary)
    calibration.overall_calibration(yes_probs, no_probs)


def calibration_at_start():
    binary = [m for m in api.get_full_markets_cached() if isinstance(m, api.BinaryMarket) and m.isResolved]
    yes_probs = np.array([m.start_probability() for m in binary if m.resolution == 'YES'])
    no_probs = np.array([m.start_probability() for m in binary if m.resolution == 'NO'])
    calibration.overall_calibration(yes_probs, no_probs)


if __name__ == "__main__":
    calibration_at_start()
