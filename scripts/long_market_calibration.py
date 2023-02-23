import numpy as np

from manifoldpy import cache_utils, calibration


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
    calibration.plot_beta_binomial(beta_intervals, beta_means, decimals=1)


if __name__ == "__main__":
    main()
