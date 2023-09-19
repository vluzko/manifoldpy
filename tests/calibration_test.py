import numpy as np
from pytest import fixture

from manifoldpy import api, calibration


class MockMarket:
    def __init__(self, p: float, r: str):
        self.probability = p
        self.resolution = r


@fixture
def test_markets():
    mkts = [
        MockMarket(0.7, "YES"),
        MockMarket(0.7, "YES"),
        MockMarket(0.7, "NO"),
    ]
    return mkts


def test_best_possible_beta():
    actual_beta = np.ones((11, 2))
    res = calibration.best_possible_beta(actual_beta, 1)
    assert np.array_equal(
        res,
        np.array(
            [
                [0, 2],
                [0, 2],
                [0, 2],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [2, 0],
                [2, 0],
                [2, 0],
            ]
        ),
    )


def test_brier_score():
    mkt = MockMarket(0.7, "YES")
    brier1 = calibration.brier_score(*calibration.extract_binary_probabilities([mkt]))  # type: ignore
    assert np.isclose(brier1, 0.09)

    brier2 = calibration.brier_score(
        *calibration.extract_binary_probabilities([MockMarket(0.7, "NO")])  # type: ignore
    )
    assert np.isclose(brier2, 0.49)


def test_log_score():
    mkt = MockMarket(0.7, "YES")
    log1 = calibration.log_score(*calibration.extract_binary_probabilities([mkt]))  # type: ignore
    assert np.isclose(log1, 0.3566749)


def test_calibration():
    mkts = [
        MockMarket(0.7, "YES"),
        MockMarket(0.7, "YES"),
        MockMarket(0.7, "NO"),
    ]
    c = calibration.binary_calibration(*calibration.extract_binary_probabilities(mkts))  # type: ignore
    assert np.isclose(c[7], 2 / 3)


def test_beta_binomial_calibration():
    mkts = [
        MockMarket(0.7, "YES"),
        MockMarket(0.7, "YES"),
        MockMarket(0.7, "NO"),
    ]
    intervals, means = calibration.beta_binomial_calibration(
        *calibration.extract_binary_probabilities(mkts)  # type: ignore
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


def test_perfect_calibration():
    assert np.isclose(
        calibration.perfect_calibration(1),
        np.array([0.025, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.975]),
    ).all()


def test_market_set_accuracy(test_markets):
    res = calibration.market_set_accuracy(*calibration.extract_binary_probabilities(test_markets))  # type: ignore
    assert np.isclose(res["10% calibration"][7], 0.66666)
    assert np.isclose(res["Brier score"], 0.22333333)


def test_build_df():
    market = api.get_full_market("6qEWrk0Af7eWupuSWxQm")
    df, _ = calibration.build_dataframe([market])
    assert df.iloc[0]["id"] == "6qEWrk0Af7eWupuSWxQm"


def test_probability_at_time():
    market = api.get_full_market("6qEWrk0Af7eWupuSWxQm")
    df, histories = calibration.build_dataframe([market])
    starts = np.array([h[0][0] for h in histories])
    ends = np.array([h[0][-1] for h in histories])
    midpoints = starts + (ends - starts) * 0.5  # type: ignore
    df["midway"] = calibration.probability_at_time(histories, midpoints)
    assert np.isclose(df["midway"][0], 0.222646)


def test_probability_at_fraction():
    market = api.get_full_market("6qEWrk0Af7eWupuSWxQm")
    df, histories = calibration.build_dataframe([market])

    df["midway"] = calibration.probability_at_fraction_completed(histories, 0.5)
    assert np.isclose(df["midway"][0], 0.222646)


def test_kl_beta():
    dist_1 = np.ones((5, 2))
    dist_2 = np.ones((5, 2)) + 2
    kl = calibration.kl_beta(dist_1, dist_2)
    assert np.isclose(kl, 0.59880262).all()
