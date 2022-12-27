import timeit
from functools import partial

from manifoldpy import api

markets = api.get_markets(limit=100)


def get_by_market():
    i = 0
    for market in markets:
        bets = api.get_bets(marketId=market.id)
        i += len(bets)
    return i


def by_bets(j):
    while True:
        if j <= 1000:
            return api.get_bets(limit=j)
        else:
            api.get_bets(limit=1000)
            j -= 1000


def main():
    print(timeit.timeit(get_by_market, number=1))
    j = get_by_market()
    print(timeit.timeit(partial(by_bets, j), number=1))


if __name__ == "__main__":
    main()
