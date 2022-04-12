"""Calibration across all resolved markets
Currently only works for binary markets.
"""
import numpy as np
from matplotlib import pyplot as plt
from manifold import api, calibration


def plot_calibration(c_table):
    fig, ax = plt.subplots()
    ax.scatter(np.arange(0, 1.01, 0.01), c_table)

    plt.show()
    import pdb
    pdb.set_trace()


def main():
    binary, _ = api.get_markets()
    brier_score = calibration.brier_score(binary)
    c_table = calibration.binary_calibration(binary)
    plot_calibration(c_table)


if __name__ == "__main__":
    main()
