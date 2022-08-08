from time import time
from sys import argv
from manifold import api


def make_market():
    key = argv[1]
    wrapper = api.APIWrapper(key)
    close = int(time()) * 1000 + 10000
    resp = wrapper.create_market(
        "BINARY",
        "POST API test question",
        "This is test question.",
        close,
        ["test", "api"],
        initialProb=50,
    )
    print(resp.text)


def resolve_market():
    key = argv[1]
    wrapper = api.APIWrapper(key)
    wrapper.resolve_market("QVVNxtYnbcZhFVkZMsB7", "YES")


if __name__ == "__main__":
    resolve_market()
