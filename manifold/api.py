import numpy as np
import requests
import pickle
import signal
import sys

from functools import partial
from typing import List, Tuple
from time import time
from manifold import config


ALL_MARKETS_URL = "https://manifold.markets/api/v0/markets"
SINGLE_MARKET_URL = "https://manifold.markets/api/v0/market/{}"


from attr import define, field
from typing import List, Optional, TypeVar, Type, Any


MarketT = TypeVar("MarketT", bound="Market")


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
    dpmShares: Optional[float]=None
    # TODO: Define Fees class
    fees: Optional[dict]=None
    # TODO: Define Sale class
    sale: Optional[dict]=None
    isSold: Optional[bool]=None
    loanAmount: Optional[float]=None
    isRedemption: Optional[bool]=None
    isAnte: Optional[bool]=None
    userId: Optional[str]=None

    @classmethod
    def from_json(cls, json: Any) -> "Bet":
        return cls(**json)  # type: ignore


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
    betId: Optional[str]=None
    answerOutcome: Optional[str]=None

    @classmethod
    def from_json(cls, json: Any) -> "Comment":
        return cls(**json)  # type: ignore


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
    pool: float
    volume7Days: float
    volume24Hours: float
    mechanism: str
    isResolved: bool
    closeTime: Optional[int] = field(kw_only=True, default=None)
    creatorAvatarUrl: Optional[str] = field(kw_only=True, default=None)
    resolution: Optional[str] = field(kw_only=True, default=None)
    resolutionTime: Optional[int] = field(kw_only=True, default=None)
    # Separating into two FullMarket types would be pointlessly annoying
    bets: Optional[List[Bet]] = field(kw_only=True, default=None)
    comments: Optional[List[Bet]] = field(kw_only=True, default=None)
    outcomeType: str
    volume: float

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
    p: Optional[float] = None
    totalLiquidity: Optional[float] = None

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
            times, probabilities = zip(*[(bet['createdTime'], bet['probAfter']) for bet in self.bets])  # type: ignore
            return np.array(times), np.array(probabilities)

    def start_probability(self) -> float:
        return self.get_updates()[1][0]

    def final_probability(self) -> float:
        return self.probability


@define
class MultiMarket(Market):
    """A market with multiple possible resolutions"""

    def final_probability(self) -> float:
        if self.bets is None:
            pass
        import pdb
        pdb.set_trace()
        raise NotImplementedError


def get_markets() -> List[Market]:
    json = requests.get(ALL_MARKETS_URL).json()

    # If this fails, the code is out of date.
    all_mechanisms = {x["mechanism"] for x in json}
    assert all_mechanisms == {"cpmm-1", "dpm-2"}

    markets = [BinaryMarket.from_json(x) if 'probability' in x else MultiMarket.from_json(x) for x in json]

    return markets


def get_market(market_id: str) -> Market:
    market = requests.get(SINGLE_MARKET_URL.format(market_id)).json()
    # market['bets'] = [Bet.from_json(x) for x in market['bets']]
    if "probability" in market:
        return BinaryMarket.from_json(market)
    else:
        return MultiMarket.from_json(market)


def get_market_cached(market_id: str) -> Market:
    try:
        full_markets = pickle.load(config.CACHE_LOC.open('rb'))
        if market_id in full_markets:
            return full_markets[market_id]
        else:
            # TODO: Update cache
            return get_market(market_id)
    except FileNotFoundError:
        return get_market(market_id)


def get_full_markets() -> List[Market]:
    """Get all markets, including bets and comments.
    Not part of the API, but handy. Takes a while to run.
    """
    markets = get_markets()

    return [get_market(x.id) for x in markets]


def get_full_markets_cached(use_cache: bool = True) -> List[Market]:
    """Get all full markets, and cache the results.
    Cache is not timestamped.
    """
    def cache_objs(full_markets, _signum, _frame):
        pickle.dump(full_markets, config.CACHE_LOC.open('wb'))
        sys.exit(0)

    if use_cache:
        try:
            full_markets = pickle.load(config.CACHE_LOC.open('rb'))
        except (FileNotFoundError, ModuleNotFoundError):
            full_markets = {}
    else:
        full_markets = {}

    # This is unnecessary in hindsight but I'll leave it in unless it gets annoying to support.
    signal.signal(signal.SIGINT, partial(cache_objs, full_markets))

    lite_markets = get_markets()
    print(f"Fetching {len(lite_markets)} markets")
    try:
        for i, lmarket in enumerate(lite_markets):
            if lmarket.id in full_markets:
                continue
            else:
                full_market = get_market(lmarket.id)
                full_markets[full_market.id] = {"market": full_market, "cache_time": time()}

            if i % 500 == 0:
                print(i)
                pickle.dump(full_markets, config.CACHE_LOC.open('wb'))
    # Happens sometimes, probably a rate limit on their end, just restart the script.
    except ConnectionResetError:
        pass
    pickle.dump(full_markets, config.CACHE_LOC.open('wb'))
    market_list = [x["market"] for x in full_markets.values()]
    return market_list