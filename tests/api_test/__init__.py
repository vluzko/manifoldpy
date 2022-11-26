import pytest

SKIP_LONG = pytest.mark.skipif("not config.getoption('runlong')")
