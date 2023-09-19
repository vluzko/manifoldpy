"""Calibration across all resolved markets.
Currently only works for binary markets.
"""
from pathlib import Path

from manifoldpy import api, calibration, cache_utils


def overall_calibration():
    """Evaluate calibration at start, end, midway, and for individual groups"""
    full_markets = cache_utils.load_full_markets()
    binary = [
        m for m in full_markets if isinstance(m, api.BinaryMarket) and m.isResolved
    ]
    df, histories = calibration.build_dataframe(binary)  # type: ignore
    # Probabilities at the halfway point
    df["midway"] = calibration.probability_at_fraction_completed(histories, 0.5)

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
    calibration.plot_calibration(
        midway_10,
        calibration.perfect_calibration(1),
        Path(__file__).parent.parent / "docs" / "midway_calibration.png",
    )
    beta_means, beta_intervals = calibration.market_set_accuracy(yes_probs, no_probs)[
        "beta-binomial"
    ]
    calibration.plot_beta_binomial(
        beta_intervals,
        beta_means,
        decimals=1,
        path=Path(__file__).parent.parent / "docs" / "beta_binomial_midway.png",
    )


if __name__ == "__main__":
    overall_calibration()
