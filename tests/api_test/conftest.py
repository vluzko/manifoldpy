def pytest_addoption(parser):
    parser.addoption(
        "--runlong", action="store_true", default=False, help="skip long tests"
    )
