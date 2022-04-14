import numpy as np
from scipy.stats import beta
from typing import List
from manifold.markets import Market


def extract_binary_probabilities(markets: List[Market]) -> np.ndarray:
    """Get the probabilities from all binary markets.
    Markets that resolve NO have their probabilities flipped
    """
    yes_probs = np.array([x.probability for x in markets if x.resolution == "YES"])
    no_probs = np.array([x.probability for x in markets if x.resolution == "NO"])
    return yes_probs, no_probs


def brier_score(markets: List[Market]) -> float:
    """Calculate brier score.
    Brier score is 1/n * sum((outcome - prob)^2) where outcome is 0 or 1.
    """
    yes_probs, no_probs = extract_binary_probabilities(markets)
    yes_scores = (1 - yes_probs) ** 2
    no_scores = no_probs ** 2

    num_mkts = len(yes_probs) + len(no_probs)
    score = 1 / num_mkts * (np.sum(yes_scores) + np.sum(no_scores))
    return score


def log_score(markets: List[Market]) -> float:
    """Calculate log score.
    Log score is sum(-log(p)), where p is the probability on the actual outcome (so 1-p in the case of markets that resolve NO).
    """
    yes_probs, no_probs = extract_binary_probabilities(markets)
    all_probs = np.concatenate((yes_probs, 1 - no_probs))
    return np.sum(-np.log(all_probs))


def relative_log_score(markets: List[Market]) -> float:
    """Relative log score of all markets against the opening position
    This is the number of nats the market adds against the opening position.
    Probably not very meaningful, given that the opening position is usually 0.5
    """
    raise NotImplementedError


def bet_counts(markets: List[Market], decimals: int) -> np.ndarray:
    """Number of bets that have resolved YES/NO within each bins"""
    yes_probs, no_probs = extract_binary_probabilities(markets)
    yes_idx, yes_counts = np.unique(yes_probs.round(decimals=decimals), return_counts=True)
    no_idx, no_counts = np.unique(no_probs.round(decimals=decimals), return_counts=True)

    all_vals = np.zeros((10**decimals + 1, 2))
    all_vals[(yes_idx * 10**decimals).astype(int), 0] = yes_counts
    all_vals[(no_idx * 10**decimals).astype(int), 1] = no_counts
    return all_vals


def binary_calibration(markets: List[Market], decimals: int=1) -> np.ndarray:
    """Calculate binary calibration across all passed markets"""
    all_vals = bet_counts(markets, decimals)

    calibration = all_vals[:, 0] / all_vals.sum(axis=1)

    return calibration


def beta_binomial_calibration(markets: List[Market], decimals: int=1) -> np.ndarray:
    """Calculate calibration with a beta-binomial model
    Returns
        0.95 confidence interval and means for the distributions.
        Intervals are centered around the median.
    """
    all_vals = bet_counts(markets, decimals)
    # Beta(1, 1) prior
    alpha_beta = all_vals + 1
    upper_lower = np.zeros_like(alpha_beta)
    for i, (alpha_val, beta_val) in enumerate(alpha_beta):
        dist = beta(alpha_val, beta_val)
        upper_lower[i] = dist.interval(0.95)
    return upper_lower, alpha_beta[:, 0] / alpha_beta.sum(axis=1)
