"""Calibration across all resolved markets
Currently only works for binary markets.
"""
import numpy as np
from matplotlib import pyplot as plt
from manifold import api, calibration


def plot_calibration(c_table, bins):
    fig, ax = plt.subplots()
    if bins is None:
        bins = np.arange(0, 1.01, 0.01)
    ax.scatter(bins, c_table)
    # Perfect calibration line
    l = np.arange(0, bins.max(), 0.0001)
    ax.scatter(np.arange(0, bins.max(), 0.0001), np.arange(0, bins.max(), 0.0001), color='red', s=0.01)

    plt.show()


def main():
    binary, _ = api.get_markets()
    brier_score = calibration.brier_score(binary)
    c_table = calibration.binary_calibration(binary)
    ten_perc_bins = np.arange(0.0, 1.01, 0.1)
    ten_percent = calibration.binary_calibration(binary, bins=ten_perc_bins)
    # plot_calibration(c_table)
    plot_calibration(ten_percent, ten_perc_bins)



if __name__ == "__main__":
    main()
