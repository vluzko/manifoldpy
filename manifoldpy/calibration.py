import numpy as np
from manifoldpy.api import Market, BinaryMarket
from matplotlib import pyplot as plt
from scipy.stats import beta
from typing import List, Tuple


def extract_binary_probabilities(
    markets: List[BinaryMarket],
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the closing probabilities from all binary markets.
    Markets that resolve NO have their probabilities flipped
    """
    yes_probs = np.array([x.probability for x in markets if x.resolution == "YES"])
    no_probs = np.array([x.probability for x in markets if x.resolution == "NO"])
    return yes_probs, no_probs


def brier_score(yes_probs: np.ndarray, no_probs: np.ndarray) -> float:
    """Calculate brier score.
    Brier score is 1/n * sum((outcome - prob)^2) where outcome is 0 or 1.
    """
    yes_scores = (1 - yes_probs) ** 2
    no_scores = no_probs**2

    num_mkts = len(yes_probs) + len(no_probs)
    score = 1 / num_mkts * (np.sum(yes_scores) + np.sum(no_scores))
    return score


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
    """Number of bets that have resolved YES/NO within each bins"""
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
    """Calculate binary calibration across all passed markets"""
    all_vals = bet_counts(yes_probs, no_probs, decimals)

    calibration = all_vals[:, 0] / all_vals.sum(axis=1)

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


def plot_beta_binomial(upper_lower: np.ndarray, means: np.ndarray, decimals):
    _, ax = plt.subplots()
    num_bins = 10**decimals
    x_axis = np.arange(0, 1 + 1 / num_bins, 1 / num_bins)
    ax.scatter(x_axis, means, color="blue")
    ax.scatter(x_axis, upper_lower[:, 0], color="black", marker="_")
    ax.scatter(x_axis, upper_lower[:, 1], color="black", marker="_")
    plt.vlines(x_axis, upper_lower[:, 0], upper_lower[:, 1], color="black")

    ax.set_xticks(np.arange(0, 1 + 1 / 10, 1 / 10))
    ax.set_xlabel("Market probability")
    ax.set_yticks(np.arange(0, 1 + 1 / 10, 1 / 10))
    ax.set_ylabel("Beta binomial means and 0.95 intervals")

    l = np.arange(0, x_axis.max(), 0.0001)
    ax.scatter(l, l, color="green", s=0.01, label="Perfect calibration")
    plt.show()


def overall_calibration(yes_probs: np.ndarray, no_probs: np.ndarray, decimals: int = 1):
    scoring_rules = {"Brier": brier_score, "Log": log_score}
    scores = {k: v(yes_probs, no_probs) for k, v in scoring_rules.items()}
    print("\n".join(f"{k}: {v}" for k, v in scores.items()))

    # Calibration with 100 bins
    one_percent = binary_calibration(yes_probs, no_probs, decimals=2)
    # plot_calibration(one_percent, bins=np.arange(0, 1+ 1/100, 1/100))

    # Calibration with 10 bins
    ten_percent = binary_calibration(yes_probs, no_probs, decimals=1)
    # plot_calibration(ten_percent, bins=np.arange(0, 1 + 1/10, 1/10))

    # Calibration when we model each bin as with a beta binomial model
    beta_interval, beta_means = beta_binomial_calibration(
        yes_probs, no_probs, decimals=1
    )
    plot_beta_binomial(beta_interval, beta_means, decimals=1)
