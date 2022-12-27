from manifoldpy import cache_utils


def main():
    cache_utils.update_lite_markets()
    cache = cache_utils.update_bets()
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
