from manifoldpy import api


def main():
    api.get_full_markets(reset_cache=True)


if __name__ == "__main__":
    main()
