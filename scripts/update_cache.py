from manifoldpy import api


def main():
    print("Updating cached markets")
    api.update_cached()
    print("Getting new markets")
    api.get_full_markets()


if __name__ == "__main__":
    main()
