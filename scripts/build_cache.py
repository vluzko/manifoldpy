from manifoldpy import cache_utils


def main():
    cache_utils.update_lite_markets()
    # cache_utils.backfill_bets()


if __name__ == "__main__":
    main()
