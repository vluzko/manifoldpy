"""Tools for calculating calibration and other accuracy metrics."""
import bisect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  # type: ignore
from scipy.stats import beta  # type: ignore

from manifoldpy.api import BinaryMarket, Market


def perfect_calibration(decimals: int) -> np.ndarray:
    p = np.linspace(0, 1, 10**decimals + 1)
    p[0] = p[1] * 0.25
    t = (p[-2] + 1) / 2
    p[-1] = (t + 1) / 2
    return p


def extract_binary_probabilities(
    markets: List[BinaryMarket],
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the closing probabilities from all binary markets."""
    yes_probs: np.ndarray = np.array(
        [x.probability for x in markets if x.resolution == "YES"]
    )
    no_probs: np.ndarray = np.array(
        [x.probability for x in markets if x.resolution == "NO"]
    )
    return yes_probs, no_probs


def brier_score(yes_probs: np.ndarray, no_probs: np.ndarray) -> float:
    """Calculate brier score.
    Brier score is 1/n * sum((outcome - prob)^2) where outcome is 0 or 1.
    """
    yes_scores = (1 - yes_probs) ** 2
    no_scores = no_probs**2

    num_mkts = len(yes_probs) + len(no_probs)
    score = 1 / num_mkts * (np.sum(yes_scores) + np.sum(no_scores))
    return score  # type: ignore


def log_score(yes_probs: np.ndarray, no_probs: np.ndarray) -> float:
    """Calculate log score.
    Log score is sum(-log(p)), where p is the probability on the actual outcome (so 1-p in the case of markets that resolve NO).
    """
    all_probs = np.concatenate((yes_probs, 1 - no_probs))
    return np.sum(-np.log(all_probs))


def relative_log_score(markets: List[Market]) -> float:
    """Relative log score of all markets against the opening position
    This is the number of nats the market adds against the opening position.
    Probably not very meaningful, given that the opening position is usually 0.5
    """
    raise NotImplementedError


def bet_counts(
    yes_probs: np.ndarray, no_probs: np.ndarray, decimals: int
) -> np.ndarray:
    """Number of bets that have resolved YES/NO within each bin"""
    yes_idx, yes_counts = np.unique(
        yes_probs.round(decimals=decimals), return_counts=True
    )
    no_idx, no_counts = np.unique(no_probs.round(decimals=decimals), return_counts=True)

    all_vals = np.zeros((10**decimals + 1, 2))
    all_vals[(yes_idx * 10**decimals).astype(int), 0] = yes_counts
    all_vals[(no_idx * 10**decimals).astype(int), 1] = no_counts
    return all_vals


def binary_calibration(
    yes_probs: np.ndarray, no_probs: np.ndarray, decimals: int = 1
) -> np.ndarray:
    """Calculate binary calibration across all passed markets

    Returns:
        An array containing the fraction of markets that actually resolved yes for the corresponding bucket.
        Buckets are constructed based on the value of `decimals`, and generally will be 10^decimals + 1.
        The `i`th element of the returned array is the fraction of markets at that confidence level that resolved true.
    """
    all_vals = bet_counts(yes_probs, no_probs, decimals)
    pred_true = all_vals[:, 0]
    all_preds = all_vals.sum(axis=1)
    calibration = np.divide(
        pred_true, all_preds, out=np.zeros_like(all_preds), where=all_preds != 0
    )

    return calibration


def beta_binomial_calibration(
    yes_probs: np.ndarray, no_probs: np.ndarray, decimals: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate calibration with a beta-binomial model

    Args:
        yes_probs: The probabilities assigned to all YES outcomes
        no_probs: The probabilities assigned to all NO outcomes
        decimals: The

    Returns
        0.95 confidence interval and means for the distributions.
        Intervals are centered around the median.
    """
    all_vals = bet_counts(yes_probs, no_probs, decimals)
    # Beta(1, 1) prior
    alpha_beta = all_vals + 1
    upper_lower = np.zeros_like(alpha_beta)
    for i, (alpha_val, beta_val) in enumerate(alpha_beta):
        dist = beta(alpha_val, beta_val)
        upper_lower[i] = dist.interval(0.95)
    return upper_lower, alpha_beta[:, 0] / alpha_beta.sum(axis=1)


def market_set_accuracy(yes_probs: np.ndarray, no_probs: np.ndarray) -> Dict[str, Any]:
    """Compute common metrics for a set of markets

    Args:
        yes_probs: The probabilities of all markets that resolved YES.
        no_probs: The probabilities of all markets that resolved NO.

    Returns:
        1% calibration, 10% calibration, beta-binomial model means, beta-binomial model 95% CI, and proper scores.
    """
    scoring_rules = {"Brier score": brier_score, "Log score": log_score}
    scores = {name: rule(yes_probs, no_probs) for name, rule in scoring_rules.items()}

    # Calibration at 1%
    one_percent = binary_calibration(yes_probs, no_probs, decimals=2)

    # Calibration at 10%
    ten_percent = binary_calibration(yes_probs, no_probs, decimals=1)

    # Beta-binomial calibration at 10%
    beta_interval, beta_means = beta_binomial_calibration(
        yes_probs, no_probs, decimals=1
    )
    return {
        "1% calibration": one_percent,
        "10% calibration": ten_percent,
        "beta-binomal": (beta_means, beta_interval),
        **scores,
    }


def build_dataframe(
    markets: List[Market],
) -> Tuple[pd.DataFrame, List[Tuple[np.ndarray, np.ndarray]]]:
    """Build a dataframe from all passed markets.

    Args:
        markets: A list of resolved markets

    Returns:
        Tuple[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]: The dataframe, and a list of betting histories for every market.
    """
    columns = ["id", "num_trades", "resolution", "volume", "tags"]
    simple_fields = [(x.id, len(x.bets), x.resolution, x.volume, tuple(y.lower() for y in x.tags)) for x in markets]  # type: ignore
    df = pd.DataFrame(simple_fields, columns=columns)

    histories = [x.probability_history() for x in markets]
    df["start"] = [p[1][0] for p in histories]
    df["final"] = [p[1][-1] for p in histories]

    df["num_traders"] = [x.num_traders() for x in markets]

    return df, histories


def probability_at_time(
    histories: List[Tuple[np.ndarray, np.ndarray]], midpoints: np.ndarray
) -> np.ndarray:
    indices = [bisect.bisect_left(h[0], m) for h, m in zip(histories, midpoints)]  # type: ignore
    # Slower, but probably easier to understand
    probabilities = np.array([h[1][i] for h, i in zip(histories, indices)])
    return probabilities


def markets_by_group(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Get a dict group -> group's markets"""
    all_tags = {y for x in df.tags.unique() for y in x}
    filters = {x: df.tags.apply(lambda y: x in y) for x in all_tags}
    return filters


def plot_beta_binomial(
    upper_lower: np.ndarray, means: np.ndarray, decimals
):  # pragma: no cover
    _, ax = plt.subplots()
    num_bins = 10**decimals
    x_axis = np.arange(0, 1 + 1 / num_bins, 1 / num_bins)
    ax.scatter(x_axis, means, color="blue")
    ax.scatter(x_axis, upper_lower[:, 0], color="black", marker="_")  # type: ignore
    ax.scatter(x_axis, upper_lower[:, 1], color="black", marker="_")  # type: ignore
    plt.vlines(x_axis, upper_lower[:, 0], upper_lower[:, 1], color="black")

    ax.set_xticks(np.arange(0, 1 + 1 / 10, 1 / 10))
    ax.set_xlabel("Market probability")
    ax.set_yticks(np.arange(0, 1 + 1 / 10, 1 / 10))
    ax.set_ylabel("Beta binomial means and 0.95 intervals")

    l = np.arange(0, x_axis.max(), 0.0001)
    ax.scatter(l, l, color="green", s=0.01, label="Perfect calibration")
    plt.show()


def plot_calibration(
    c_table: np.ndarray, bins: np.ndarray, path: Optional[Path] = None
):  # pragma: no cover
    _, ax = plt.subplots()
    ax.scatter(bins, c_table, label="Market calibration")
    # Perfect calibration line
    l = np.arange(0, bins.max(), 0.0001)
    ax.scatter(l, l, color="green", s=0.01, label="Perfect calibration")

    ax.set_xticks(np.arange(0, 1 + 1 / 10, 1 / 10))
    ax.set_xlabel("Market probability")
    ax.set_yticks(np.arange(0, 1 + 1 / 10, 1 / 10))
    ax.set_ylabel("Empirical probability")
    ax.legend()

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
