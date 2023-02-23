"""Tools for calculating calibration and other accuracy metrics."""
import bisect
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
from matplotlib import pyplot as plt  # type: ignore
from scipy.stats import beta  # type: ignore

from manifoldpy.api import BinaryMarket, Market


def perfect_calibration(decimals: int) -> np.ndarray:
    p = np.linspace(0, 1, 10**decimals + 1)
    p[0] = p[1] * 0.25
    t = (p[-2] + 1) / 2
    p[-1] = (t + 1) / 2
    return p


def best_possible_beta(actual_beta: np.ndarray, decimals: int) -> np.ndarray:
    """Compute the best possible beta distribution for the given number of bets"""
    fraction_true = perfect_calibration(decimals).reshape(-1, 1)
    total_bets = actual_beta.sum(axis=1).reshape(-1, 1)
    true = np.round(fraction_true * total_bets)
    false = total_bets - true

    assert (true + false == total_bets).all()

    return np.concatenate((true, false), axis=1)


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
        "beta-binomial": (beta_means, beta_interval),
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


def probability_at_fraction_completed(
    histories: List[Tuple[np.ndarray, np.ndarray]], fraction: float
) -> np.ndarray:
    """Find the probability of each market at the given timepoints.

    Args:
        histories: A tuple of (timepoints, probabilities) for each market.
        fraction: The fraction of the market's runtime that has completed. e.g. 0.5 for halfway finished.

    Returns:
        An array of probabilities. array[i] = probabilities[i] at midpoints[i].
    """
    assert 0.0 <= fraction <= 1.0
    starts: npt.NDArray[np.float64] = np.array([h[0][0] for h in histories])
    ends: npt.NDArray[np.float64] = np.array([h[0][-1] for h in histories])
    midpoints = starts + (ends - starts) * fraction
    return probability_at_time(histories, midpoints)


def probability_at_time(
    histories: List[Tuple[np.ndarray, np.ndarray]], midpoints: np.ndarray
) -> np.ndarray:
    """Find the probability of each market at the given timepoints.

    Args:
        histories: A tuple of (timepoints, probabilities) for each market.
        midpoints: An array of timepoints for each market.

    Returns:
        An array of probabilities. array[i] = probabilities[i] at midpoints[i].
    """
    indices = [bisect.bisect_left(h[0], m) for h, m in zip(histories, midpoints)]  # type: ignore
    # Slower, but probably easier to understand
    probabilities = np.array([h[1][i] for h, i in zip(histories, indices)])
    return probabilities


def markets_by_group(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Get a dict group -> group's markets"""
    all_tags = {y for x in df.tags.unique() for y in x}
    filters = {x: df.tags.apply(lambda y: x in y) for x in all_tags}
    return filters


def kl_beta(
    dist_1: npt.NDArray[np.float64], dist_2: npt.NDArray[np.float64]
) -> np.ndarray:
    """Calculate KL(dist_1 || dist_2) for each row of dist_1 and dist_2.
    Formula taken from https://math.stackexchange.com/questions/257821/kullback-liebler-divergence#comment564291_257821

    Args:
        dist_1: An array of shape [n, 2]. Each row is [alpha, beta] for a beta distribution.
        dist_2: An array of shape [n, 2]. Each row is [alpha, beta] for a beta distribution.

    Returns:
        An array of shape [n]. array[i] is the KL divergence between dist_1[i] and dist_2[i].
    """
    alpha_1 = dist_1[:, 0]
    beta_1 = dist_1[:, 1]
    alpha_2 = dist_2[:, 0]
    beta_2 = dist_2[:, 1]
    numer = (
        scipy.special.gammaln(alpha_1 + beta_1)
        + scipy.special.gammaln(alpha_2)
        + scipy.special.gammaln(beta_2)
    )
    denom = (
        scipy.special.gammaln(alpha_2 + beta_2)
        + scipy.special.gammaln(alpha_1)
        + scipy.special.gammaln(beta_1)
    )

    term_1 = numer - denom

    term_2 = (alpha_1 - alpha_2) * (
        scipy.special.psi(alpha_1) - scipy.special.psi(alpha_1 + beta_1)
    )

    term_3 = (beta_1 - beta_2) * (
        scipy.special.psi(beta_1) - scipy.special.psi(alpha_1 + beta_1)
    )
    return term_1 + term_2 + term_3


def plot_beta_binomial(  # pragma: no cover
    upper_lower: np.ndarray, means: np.ndarray, decimals
):
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
