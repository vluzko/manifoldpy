from manifoldpy import cache_utils


def main():
    print("Updating cached markets")
    cache_utils.update_cached()
    print("Getting new markets")
    cache_utils.get_full_markets()


if __name__ == "__main__":
    main()
