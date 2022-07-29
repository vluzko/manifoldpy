import numpy as np
from pytest import fixture
from manifold import calibration


class MockMarket:
    def __init__(self, p: float, r: str):
        self.probability = p
        self.resolution = r


def test_brier_score():
    mkt = MockMarket(0.7, "YES")
    brier1 = calibration.brier_score([mkt])
    assert np.isclose(brier1, 0.09)

    brier2 = calibration.brier_score([MockMarket(0.7, "NO")])
    assert np.isclose(brier2, 0.49)


def test_log_score():
    raise NotImplementedError


def test_calibration():
    mkts = [
        MockMarket(0.7, "YES"),
        MockMarket(0.7, "YES"),
        MockMarket(0.7, "NO"),
    ]
    c = calibration.binary_calibration(mkts)
    assert np.isclose(c[7], 2 / 3)


def test_beta_binomial_calibration():
    raise NotImplementedError
