import numpy as np
from typing import Optional


def extract_binary_probabilities(markets) -> np.ndarray:
    """Get the probabilities from all binary markets
    Markets that resolve NO have their probabilities flipped
    """
    yes_probs = np.array([x.probability for x in markets if x.resolution == "YES"])
    no_probs = np.array([1 - x.probability for x in markets if x.resolution == "NO"])
    all_probs = np.concatenate((yes_probs, no_probs))
    return all_probs


def brier_score(markets) -> float:
    """Calculate brier score across all passed markets"""
    all_probs = extract_binary_probabilities(markets)
    num_mkts = len(all_probs)
    score = 1 / num_mkts * np.sum((np.ones(num_mkts) - all_probs) ** 2)
    return score


def log_score(markets) -> float:
    """Calculate log score across all passed markets"""
    all_probs = extract_binary_probabilities(markets)
    return np.sum(-np.log(all_probs))


def binary_calibration(markets, bins: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate binary calibration across all passed markets
    TODO: Ideally this would be beta-binomial model
    """
    yes_probs = np.array([x.probability for x in markets if x.resolution == "YES"])
    no_probs = np.array([x.probability for x in markets if x.resolution == "NO"])
    if bins is None:
        bins = np.arange(0.0, 1.01, 0.01)
    yes_idx, yes_counts = np.unique(np.digitize(yes_probs, bins), return_counts=True)
    no_idx, no_counts = np.unique(np.digitize(no_probs, bins), return_counts=True)

    all_vals = np.zeros((len(bins), 2))
    all_vals[yes_idx, 0] = yes_counts
    all_vals[no_idx, 1] = no_counts

    calibration = all_vals[:, 0] / all_vals.sum(axis=1)

    return calibration
