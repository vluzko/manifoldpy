import numpy as np
import requests
import pickle

from typing import Dict, List, Tuple
from time import time
from manifold import config
from attr import define, field
from typing import List, Optional, TypeVar, Type, Any


V0_URL = "https://manifold.markets/api/v0/"
USERNAME_URL = V0_URL + "user/{}"
USER_ID_URL = V0_URL + "user/by-id/{}"
USERS_URL = V0_URL + "users"
ALL_MARKETS_URL = V0_URL + "markets"
SINGLE_MARKET_URL = V0_URL + "market/{}"
MARKET_SLUG_URL = V0_URL + "slug/{}"
BETS_URL = V0_URL + "bets"


MarketT = TypeVar("MarketT", bound="Market")


@define
class User:
    """A manifold user"""

    id: str
    createdTime: int
    name: str
    username: str
    url: str
    avatarUrl: str
    balance: float
    totalDeposits: float
    profitCached: Dict[str, float]
    creatorVolumeCached: Dict[str, float]
    bio: Optional[str] = None
    twitterHandle: Optional[str] = None
    discordHandle: Optional[str] = None
    bannerUrl: Optional[str] = None
    website: Optional[str] = None

    @classmethod
    def from_json(cls, json: Any) -> "User":
        return cls(**json)


@define
class Bet:
    """A single bet"""

    contractId: str
    createdTime: int
    shares: float
    amount: int
    probAfter: float
    probBefore: float
    id: str
    outcome: str
    isLiquidityProvision: Optional[bool] = None
    isCancelled: Optional[bool] = None
    orderAmount: Optional[float] = None
    fills: Optional[float] = None
    isFilled: Optional[bool] = None
    limitProb: Optional[float] = None
    dpmShares: Optional[float] = None
    # TODO: Define Fees class
    fees: Optional[dict] = None
    # TODO: Define Sale class
    sale: Optional[dict] = None
    isSold: Optional[bool] = None
    loanAmount: Optional[float] = None
    isRedemption: Optional[bool] = None
    isAnte: Optional[bool] = None
    userId: Optional[str] = None

    @classmethod
    def from_json(cls, json: Any) -> "Bet":
        return cls(**json)


@define
class Comment:
    """A comment on a market"""

    id: str
    contractId: str
    userUsername: str
    userAvatarUrl: str
    userId: str
    text: str
    createdTime: int
    userName: str
    betId: Optional[str] = None
    answerOutcome: Optional[str] = None

    @classmethod
    def from_json(cls, json: Any) -> "Comment":
        return cls(**json)


@define
class Answer:
    """An answer to a free response market"""


@define
class Market:
    """A market"""

    id: str
    creatorUsername: str
    creatorName: str
    createdTime: int
    question: str
    description: str
    tags: List[str]
    url: str
    pool: Dict[str, float]
    volume: float
    volume7Days: float
    volume24Hours: float
    outcomeType: str
    mechanism: str
    isResolved: bool
    resolutionProbability: Optional[float] = field(kw_only=True, default=None)
    p: Optional[float] = field(kw_only=True, default=None)
    totalLiquidity: Optional[float] = field(kw_only=True, default=None)
    closeTime: Optional[int] = field(kw_only=True, default=None)
    creatorAvatarUrl: Optional[str] = field(kw_only=True, default=None)
    resolution: Optional[str] = field(kw_only=True, default=None)
    resolutionTime: Optional[int] = field(kw_only=True, default=None)
    # Separating into Lite and Full market types would be pointlessly annoying
    bets: Optional[List[Bet]] = field(kw_only=True, default=None)
    comments: Optional[List[Bet]] = field(kw_only=True, default=None)

    @property
    def slug(self) -> str:
        return self.url.split("/")[-1]

    def get_updates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all updates to this market.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The time of each update, and the probabilities after each update.
        """
        raise NotImplementedError

    def start_probability(self) -> float:
        """Get the starting probability of the market"""
        raise NotImplementedError

    def final_probability(self) -> float:
        """Get the final probability of this market"""
        raise NotImplementedError

    @classmethod
    def from_json(cls: Type[MarketT], json: Any) -> MarketT:
        # TODO: *Maybe* clean this up. The API is pretty inconsistent and I don't really see the
        # benefit of handling all the idiosyncracies.
        # if 'bets' in json:
        #     json['bets'] = [Bet.from_json(bet) for bet in json['bets']]
        # if 'comments' in json:
        #     json['comments'] = [Comment.from_json(comment) for comment in json['comments']]
        return cls(**json)  # type: ignore


@define
class BinaryMarket(Market):
    """A market with a binary resolution
    Attributes:
        probability: The current resolution probability
        p: Something to do with CFMM markets
        totalLiquidity: Also something to do with CFMM markets
    """

    probability: float

    def get_updates(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.bets is None:
            full = get_market(self.id)
            self.bets = full.bets
            self.comments = full.comments

        assert self.bets is not None
        if len(self.bets) == 0:
            return np.array([self.createdTime]), np.array([self.probability])
        else:
            # TODO: Fix the string access after the API is cleaned up
            start_prob = self.bets[0]["probBefore"]
            start_time = self.createdTime
            times, probabilities = zip(
                *[(bet["createdTime"], bet["probAfter"]) for bet in self.bets]
            )
            return np.array((start_time, *times)), np.array(
                (start_prob, *probabilities)
            )

    def start_probability(self) -> float:
        return self.get_updates()[1][0]

    def final_probability(self) -> float:
        return self.probability


@define
class FreeResponseMarket(Market):
    """A market with multiple possible resolutions"""

    answers: Optional[List[Any]] = None

    def final_probability(self) -> float:
        if self.bets is None:
            pass
        raise NotImplementedError


def get_user_by_name(username: str) -> User:
    """Get the data for one user from their username
    [API reference](https://docs.manifold.markets/api#get-v0userusername)

    Args:
        username:
    """
    resp = requests.get(USERNAME_URL.format(username))
    resp.raise_for_status()
    return User.from_json(resp.json())


def get_user_by_id(user_id: str) -> User:
    """Get the data for one user from their username
    [API reference](https://docs.manifold.markets/api#get-v0userby-idid)

    Args:
        user_id:
    """
    resp = requests.get(USER_ID_URL.format(user_id))
    resp.raise_for_status()
    return User.from_json(resp.json())


def get_users() -> List[User]:
    """Get all users
    [API reference](https://docs.manifold.markets/api#get-v0users)
    """
    resp = requests.get(USERS_URL)
    resp.raise_for_status()
    return [User.from_json(x) for x in resp.json()]


def get_markets(limit: int = 1000, before: Optional[str] = None) -> List[Market]:
    """Get a list of markets
    [API reference](https://docs.manifold.markets/api#get-v0markets)

    Args:
        limit:
        before:

    """
    if before is not None:
        params = {"limit": limit, "before": before}
    else:
        params = {"limit": limit}
    json = requests.get(ALL_MARKETS_URL, params=params).json()  # type: ignore

    # If this fails, the code is out of date.
    all_mechanisms = {x["mechanism"] for x in json}
    assert all_mechanisms == {"cpmm-1", "dpm-2"}

    markets = [
        BinaryMarket.from_json(x)
        if "probability" in x
        else FreeResponseMarket.from_json(x)
        for x in json
    ]

    return markets


def get_slug(slug: str) -> Market:
    """Get a market by its slug
    [API reference](https://docs.manifold.markets/api#get-v0slugmarketslug)
    """
    market = requests.get(MARKET_SLUG_URL.format(slug)).json()
    if "probability" in market:
        return BinaryMarket.from_json(market)
    else:
        return FreeResponseMarket.from_json(market)


def get_market(market_id: str) -> Market:
    """Get a single full market

    Raises:
        HTTPError: If the API gives a bad response.
            This is known to happen with markets with a very large number of bets.
    """
    resp = requests.get(SINGLE_MARKET_URL.format(market_id), timeout=20)
    resp.raise_for_status()
    market = resp.json()
    if "probability" in market:
        return BinaryMarket.from_json(market)
    else:
        return FreeResponseMarket.from_json(market)


def get_bets(
    username: Optional[str] = None,
    market: Optional[str] = None,
    limit: int = 1000,
    before: Optional[str] = None,
) -> List[Bet]:
    """Get all bets.
    [API reference](https://docs.manifold.markets/api#get-v0bets)

    Args:
        username:
        market:
        limit:
        before:
    """
    params: Dict[str, Any] = {"limit": limit}
    if username is not None:
        params["username"] = username
    if market is not None:
        params["market"] = market
    if before is not None:
        params["before"] = before
    resp = requests.get(BETS_URL, params=params)
    resp.raise_for_status()

    return [Bet.from_json(x) for x in resp.json()]


def make_bet():
    """Make a bet.
    [API reference](https://docs.manifold.markets/api#post-v0bet)
    """
    raise NotImplementedError


def create_market():
    """Create a new market
    [API reference](https://docs.manifold.markets/api#post-v0market)
    """
    raise NotImplementedError


def resolve_market():
    """Resolve an existing market.
    [API reference](https://docs.manifold.markets/api#post-v0marketmarketidresolve)
    """
    raise NotImplementedError


def sell_shares():
    """Sell shares on a market.
    [API reference](https://docs.manifold.markets/api#post-v0marketmarketidsell)
    """
    raise NotImplementedError


def get_all_markets() -> List[Market]:
    """Get all markets.
    Unlike get_markets, this will get all available markets, without a limit
    on the number fetched.
    Automatically calls the markets endpoint until all data has been read.
    """
    markets = get_markets(limit=1000)
    i = markets[0].id
    while True:
        new_markets = get_markets(limit=1000, before=i)
        markets.extend(new_markets)
        if len(new_markets) < 1000:
            break
        else:
            i = markets[-1].id
    return markets


def get_full_markets(reset_cache: bool = False, cache_every: int = 500) -> List[Market]:
    """Get all full markets, and cache the results.
    Cache is not timestamped.

    Args:
        reset_cache: Whether or not to overwrite the existing cache
        cache_every: How frequently to cache the updated markets.
    """

    if not reset_cache:
        try:
            full_markets = pickle.load(config.CACHE_LOC.open("rb"))
        except FileNotFoundError:
            full_markets = {}
        # Indicates the cache is out of date and needs to be overwritten anyway
        except ModuleNotFoundError:
            full_markets = {}
            pickle.dump(full_markets, config.CACHE_LOC.open("wb"))
    else:
        full_markets = {}
        pickle.dump(full_markets, config.CACHE_LOC.open("wb"))

    lite_markets = get_markets()
    print(f"Fetching {len(lite_markets)} markets")
    missed_markets = []
    for i, lmarket in enumerate(lite_markets):
        if lmarket.id in full_markets:
            continue
        else:
            try:
                full_market = get_market(lmarket.id)
                full_markets[full_market.id] = {
                    "market": full_market,
                    "cache_time": time(),
                }
            # If we get an HTTP Error, just skip that market
            except requests.HTTPError:
                missed_markets.append(lmarket.id)

        if i % cache_every == 0:
            print(i)
            pickle.dump(full_markets, config.CACHE_LOC.open("wb"))
    pickle.dump(full_markets, config.CACHE_LOC.open("wb"))
    market_list = [x["market"] for x in full_markets.values()]
    print(f"Could not get {len(missed_markets)} markets.")
    return market_list
