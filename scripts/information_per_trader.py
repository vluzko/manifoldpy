import numpy as np
import pandas as pd

from manifoldpy import api, cache_utils, calibration


def main():
    # Get top 50 users
    # Note that this is *current* top 50 users. I don't believe there's an easy way to get historical top 50 users
    top_users = sorted(api.get_all_users(), key=lambda x: x.profitCached["allTime"], reverse=True)[:50]  # type: ignore
    user_ids = {u.id for u in top_users}
    user_map = {u.id: u.name for u in top_users}
    # Load their bets
    cache = cache_utils.load_cache()
    bets_filtered = [
        (m_id, b_id, b["userId"], b["probBefore"], b["probAfter"])
        for m_id, m_bets in cache["bets"].items()
        for b_id, b in m_bets.items()
        if b["userId"] in user_ids
    ]
    columns = ["market_id", "bet_id", "user_id", "prob_before", "prob_after"]
    bets_df = pd.DataFrame(bets_filtered, columns=columns)

    # Filter for markets closed before market creation
    # MARKET_CREATION = 1668841200000
    bets_df["market_close"] = bets_df["market_id"].apply(
        lambda x: cache["lite_markets"][x]["closeTime"]
    )
    # before_creation = bets_df[bets_df['market_close'] < MARKET_CREATION]
    before_creation = bets_df

    # Filter for resolved markets
    resolved = before_creation[
        before_creation["market_id"].apply(
            lambda x: cache["lite_markets"][x]["isResolved"]
        )
    ]

    # Add resolution
    resolved["resolution"] = resolved["market_id"].apply(
        lambda x: cache["lite_markets"][x]["resolution"]
    )

    # Filter for binary markets
    resolved["outcome_type"] = resolved["market_id"].apply(
        lambda x: cache["lite_markets"][x]["outcomeType"]
    )
    binary_markets = resolved[resolved["outcome_type"] == "BINARY"]

    # Filter for YES/NO markets
    yes_no = binary_markets[binary_markets["resolution"].isin({"YES", "NO"})]
    no_filt = yes_no["resolution"] == "NO"
    # Flip NO probabilities
    yes_no.loc[no_filt, "prob_before"] = 1 - yes_no.loc[no_filt, "prob_before"]
    yes_no.loc[no_filt, "prob_after"] = 1 - yes_no.loc[no_filt, "prob_after"]

    yes_no["info_gain"] = np.log2(yes_no["prob_after"]) - np.log2(yes_no["prob_before"])  # type: ignore

    per_bet_gain = (
        yes_no.groupby("user_id")["info_gain"].mean().sort_values(ascending=False)
    )
    full_order = [user_map[u_id] for u_id in per_bet_gain.index]
    top_user = user_map[per_bet_gain.index[0]]
    print(top_user)
    print(full_order)

    per_mkt_gain = (
        yes_no.groupby(["user_id", "market_id"])["info_gain"].sum().reset_index()
    )
    per_mkt_avg = (
        per_mkt_gain.groupby("user_id")["info_gain"].mean().sort_values(ascending=False)
    )
    mkt_order = [user_map[u_id] for u_id in per_mkt_avg.index]
    print(mkt_order)

    import pdb

    pdb.set_trace()

    # Calculate information gain (per bet)

    # Calculate information gain (per market)


if __name__ == "__main__":
    main()
