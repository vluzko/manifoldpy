import numpy as np
from pytest import fixture
from manifold import calibration


class MockMarket:
    def __init__(self, p: float, r: str):
        self.probability = p
        self.resolution = r


def test_brier_score():
    mkt = MockMarket(0.7, "YES")
    brier1 = calibration.brier_score(*calibration.extract_binary_probabilities([mkt]))
    assert np.isclose(brier1, 0.09)

    brier2 = calibration.brier_score(
        *calibration.extract_binary_probabilities([MockMarket(0.7, "NO")])
    )
    assert np.isclose(brier2, 0.49)


def test_log_score():
    mkt = MockMarket(0.7, "YES")
    log1 = calibration.log_score(*calibration.extract_binary_probabilities([mkt]))
    assert np.isclose(log1, 0.3566749)


def test_calibration():
    mkts = [
        MockMarket(0.7, "YES"),
        MockMarket(0.7, "YES"),
        MockMarket(0.7, "NO"),
    ]
    c = calibration.binary_calibration(*calibration.extract_binary_probabilities(mkts))
    assert np.isclose(c[7], 2 / 3)


def test_beta_binomial_calibration():
    mkts = [
        MockMarket(0.7, "YES"),
        MockMarket(0.7, "YES"),
        MockMarket(0.7, "NO"),
    ]
    intervals, means = calibration.beta_binomial_calibration(
        *calibration.extract_binary_probabilities(mkts)
    )
    for i in range(len(intervals)):
        if i == 7:
            continue
        else:
            assert np.isclose(intervals[i], [0.025, 0.975]).all()
            assert means[i] == 0.5
    alpha = 1 + len([x for x in mkts if x.resolution == "YES"])
    beta = len(mkts) - (alpha - 1) + 1
    assert means[7] == alpha / (alpha + beta)
