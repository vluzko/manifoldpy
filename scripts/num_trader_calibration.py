"""Calibration across all resolved markets.
Currently only works for binary markets.
"""
from pathlib import Path
import numpy as np
import pickle

from manifoldpy import api, calibration, config


def double_check(df):
    """Double check binary calibration against library method"""
    relevant = df[df["resolution"].isin({"YES", "NO"})]
    total_counts = relevant["p_of_ith"].round(decimals=1).value_counts().sort_index()
    yes_counts = (
        relevant[relevant["resolution"] == "YES"]["p_of_ith"]
        .round(decimals=1)
        .value_counts()
        .sort_index()
    )
    return yes_counts / total_counts


def main():
    """Evaluate calibration at start, end, midway, and for individual groups"""
    market_cache = pickle.load(config.CACHE_LOC.open("rb"))
    full_markets = [x["market"] for x in market_cache.values()]
    binary = [
        m for m in full_markets if isinstance(m, api.BinaryMarket) and m.isResolved
    ]
    df, histories = calibration.build_dataframe(binary)  # type: ignore
    df["histories"] = [h[1] for h in histories]
    traders = [
        [b.userId for b in sorted(m.bets, key=lambda b: b.createdTime)] for m in binary
    ]
    new_trader = [
        np.array([x not in t[:i] for i, x in enumerate(t)]).cumsum() for t in traders
    ]
    df["new_trader"] = new_trader
    last = len(df)
    for i in range(1, 100):
        print(f"After {i} traders:")
        sub_df = df[df["num_traders"] > i].copy()

        # Should monotonically decrease
        assert len(sub_df) <= last
        last = len(sub_df)

        index_of_ith = sub_df["new_trader"].map(lambda x: np.where(x == i)[0][0])

        # Check trade index
        for df_idx, t_idx in index_of_ith.items():
            row = df.loc[df_idx]
            assert row["new_trader"][t_idx] == i
            if t_idx > 0:
                assert row["new_trader"][t_idx] == 1 + row["new_trader"][t_idx - 1]

        # Grab the probability corresponding to the i-th trader
        sub_df["p_of_ith"] = [
            sub_df["histories"][i][t_idx] for i, t_idx in index_of_ith.items()
        ]

        yes_markets = sub_df[sub_df["resolution"] == "YES"]
        no_markets = sub_df[sub_df["resolution"] == "NO"]

        # Calibration at start
        yes_probs = yes_markets["p_of_ith"]
        no_probs = no_markets["p_of_ith"]
        dc = double_check(sub_df)
        ten_percent = calibration.binary_calibration(yes_probs, no_probs, decimals=1)
        assert (dc.array == ten_percent).all()
        perfect = calibration.perfect_calibration(decimals=1)
        diff = np.abs(ten_percent - perfect)
        print("\n".join(f"{i:.3f}: {d:.4f}" for i, d in zip(perfect, diff)))
        total_good = len(np.where(diff <= 0.05)[0])
        print(f"Total in range: {total_good}")

        if total_good >= 7:
            print(f"Stopping at {i}")
            break


if __name__ == "__main__":
    main()
