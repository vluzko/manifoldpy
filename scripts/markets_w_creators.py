import numpy as np
import scipy

from manifoldpy import cache_utils, calibration, config


def load():
    has_counts = np.load(config.DATA / "has_counts.npy") + 1
    lacks_counts = np.load(config.DATA / "lacks_counts.npy") + 1

    perfect_has = calibration.best_possible_beta(has_counts, decimals=1)
    perfect_lacks = calibration.best_possible_beta(lacks_counts, decimals=1)

    has_kl = calibration.kl_beta(has_counts, perfect_has)

    lacks_kl = calibration.kl_beta(lacks_counts, perfect_lacks)
    print(f"Has KL: {has_kl.mean()}")
    print(f"Lacks KL: {lacks_kl.mean()}")


def main():
    full_markets = cache_utils.load_full_markets()
    binary = [
        m
        for m in full_markets
        if m.outcomeType == "BINARY" and m.isResolved and len(m.bets) > 5
    ]
    traders = [{b.userId for b in m.bets} for m in binary]
    df, histories = calibration.build_dataframe(binary)
    df["probabilities"] = calibration.probability_at_fraction_completed(histories, 0.75)
    df["has_creator"] = [m.creatorId in traders[i] for i, m in enumerate(binary)]

    has = df[df["has_creator"]]
    lacks = df[~df["has_creator"]]

    has_means, has_intervals = calibration.market_set_accuracy(
        has[has["resolution"] == "YES"]["probabilities"],
        has[has["resolution"] == "NO"]["probabilities"],
    )["beta-binomial"]
    lacks_means, lacks_intervals = calibration.market_set_accuracy(
        lacks[lacks["resolution"] == "YES"]["probabilities"],
        lacks[lacks["resolution"] == "NO"]["probabilities"],
    )["beta-binomial"]

    calibration.plot_beta_binomial(has_intervals, has_means, 1)
    calibration.plot_beta_binomial(lacks_intervals, lacks_means, 1)

    has_counts = calibration.bet_counts(
        has[has["resolution"] == "YES"]["probabilities"],
        has[has["resolution"] == "NO"]["probabilities"],
        decimals=1,
    )
    lacks_counts = calibration.bet_counts(
        lacks[lacks["resolution"] == "YES"]["probabilities"],
        lacks[lacks["resolution"] == "NO"]["probabilities"],
        decimals=1,
    )

    np.save(config.DATA / "has_counts.npy", has_counts)
    np.save(config.DATA / "lacks_counts.npy", lacks_counts)


if __name__ == "__main__":
    # main()
    load()
