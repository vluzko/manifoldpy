from api_test import SKIP_LONG

from manifoldpy import api


@SKIP_LONG
def test_mechanisms():
    markets = api.get_all_markets()
    all_mechanisms = {x.mechanism for x in markets}
    assert all_mechanisms <= {"cpmm-1", "dpm-2"}
