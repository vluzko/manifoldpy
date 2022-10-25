"""API bindings"""
import numpy as np
import requests
import pickle

from attr import define, field
from typing import Dict, List, Tuple, List, Optional, TypeVar, Type, Any, Literal
from time import time
from manifoldpy import config


V0_URL = "https://manifold.markets/api/v0/"
USERNAME_URL = V0_URL + "user/{}"
USER_ID_URL = V0_URL + "user/by-id/{}"
USERS_URL = V0_URL + "users"
ALL_MARKETS_URL = V0_URL + "markets"
SINGLE_MARKET_URL = V0_URL + "market/{}"
MARKET_SLUG_URL = V0_URL + "slug/{}"
BETS_URL = V0_URL + "bets"

MAKE_BET_URL = V0_URL + "bet"
CREATE_MARKET_URL = V0_URL + "market"
RESOLVE_MARKET_URL = V0_URL + "market/{}/resolve"
SELL_SHARES_URL = V0_URL + "market/{}/sell"

MarketT = TypeVar("MarketT", bound="Market")
OutcomeType = Literal[
    "BINARY", "FREE_RESPONSE", "NUMERIC", "PSEUDO_NUMERIC", "MULTIPLE_CHOICE"
]
T = TypeVar("T")


def weak_structure(json: dict, cls: Type[T]) -> T:
    fields = {}
    for f in cls.__attrs_attrs__:  # type: ignore
        val = json.get(f.name, f.default)
        fields[f.name] = val
    return cls(**fields)


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
    profitCached: Dict[str, Optional[float]]
    creatorVolumeCached: Dict[str, float]
    bio: Optional[str] = None
    twitterHandle: Optional[str] = None
    discordHandle: Optional[str] = None
    bannerUrl: Optional[str] = None
    website: Optional[str] = None


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
    challengeSlug: Optional[str] = None
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
    tags: List[str]
    url: str
    pool: Dict[str, float]
    volume: float
    volume7Days: float
    volume24Hours: float
    outcomeType: OutcomeType
    mechanism: str
    isResolved: bool
    resolutionProbability: Optional[float] = field(kw_only=True, default=None)
    p: Optional[float] = field(kw_only=True, default=None)
    totalLiquidity: Optional[float] = field(kw_only=True, default=None)
    closeTime: Optional[int] = field(kw_only=True, default=None)
    creatorId: Optional[str] = field(kw_only=True, default=None)
    lastUpdatedTime: Optional[int] = field(kw_only=True, default=None)
    creatorAvatarUrl: Optional[str] = field(kw_only=True, default=None)
    resolution: Optional[str] = field(kw_only=True, default=None)
    resolutionTime: Optional[int] = field(kw_only=True, default=None)
    min: Optional[int] = field(kw_only=True, default=None)
    max: Optional[int] = field(kw_only=True, default=None)
    isLogScale: Optional[bool] = field(kw_only=True, default=None)
    # Separating into Lite and Full market types would be pointlessly annoying
    description: Optional[str] = field(kw_only=True, default=None)
    textDescription: Optional[str] = field(kw_only=True, default=None)
    bets: Optional[List[Bet]] = field(kw_only=True, default=None)
    comments: Optional[List[Comment]] = field(kw_only=True, default=None)

    @property
    def slug(self) -> str:
        return self.url.split("/")[-1]

    def get_updates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all updates to this market.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The time of each update, and the probabilities after each update.
        """
        raise NotImplementedError

    def num_traders(self) -> int:
        raise NotImplementedError

    def probability_history(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def start_probability(self) -> float:
        """Get the starting probability of the market"""
        raise NotImplementedError

    def final_probability(self) -> float:
        """Get the final probability of this market"""
        raise NotImplementedError

    @staticmethod
    def from_json(json: Any) -> "Market":
        market: Market
        if "bets" in json:
            json["bets"] = [weak_structure(x, Bet) for x in json["bets"]]
        if "comments" in json:
            json["comments"] = [weak_structure(x, Comment) for x in json["comments"]]

        cls: Type["Market"]
        if json["outcomeType"] == "BINARY":
            cls = BinaryMarket
        elif json["outcomeType"] == "FREE_RESPONSE":
            cls = FreeResponseMarket
        elif json["outcomeType"] == "NUMERIC":
            cls = NumericMarket
        elif json["outcomeType"] == "PSEUDO_NUMERIC":
            cls = PseudoNumericMarket
        elif json["outcomeType"] == "MULTIPLE_CHOICE":
            cls = MultipleChoiceMarket
        else:
            raise ValueError(
                f'{json["outcomeType"]} isn\'t a known market outcome type. Submit a bug report if the json came from the API.'
            )
        market = weak_structure(json, cls)
        return market


@define
class BinaryMarket(Market):
    """A market with a binary resolution
    Attributes:
        probability: The current resolution probability
    """

    probability: float

    def num_traders(self) -> int:
        if self.bets is None:
            return 0
        else:
            unique_traders = {b.userId for b in self.bets}
            return len(unique_traders)

    def probability_history(self) -> Tuple[np.ndarray, np.ndarray]:
        assert (
            self.bets is not None
        ), "Call get_market before accessing probability history"
        if len(self.bets) == 0:
            times, probabilities = np.array([self.createdTime]), np.array(
                [self.probability]
            )
        else:
            s_bets = sorted(self.bets, key=lambda x: x.createdTime)
            start_prob = s_bets[0].probBefore
            start_time = self.createdTime
            t_iter, p_iter = zip(*[(bet.createdTime, bet.probAfter) for bet in s_bets])
            times, probabilities = np.array((start_time, *t_iter)), np.array(
                (start_prob, *p_iter)
            )
        return times, probabilities

    def start_probability(self) -> float:
        return self.probability_history()[1][0]

    def final_probability(self) -> float:
        return self.probability_history()[1][-1]

    def probability_at_time(self, timestamp: int):
        times, probs = self.probability_history()
        if timestamp <= times[0]:
            raise ValueError("Timestamp before market creation")
        elif timestamp >= times[-1]:
            return probs[-1]
        else:
            raise NotImplementedError


@define
class FreeResponseMarket(Market):
    """A market with multiple possible resolutions"""

    answers: Optional[List[Any]] = None

    def final_probability(self) -> float:
        if self.bets is None:
            pass
        raise NotImplementedError


@define
class NumericMarket(Market):
    pass


@define
class PseudoNumericMarket(Market):
    pass


@define
class MultipleChoiceMarket(Market):
    pass


def get_user_by_name(username: str) -> User:
    """Get the data for one user from their username
    [API reference](https://docs.manifold.markets/api#get-v0userusername)

    Args:
        username:
    """
    resp = requests.get(USERNAME_URL.format(username))
    resp.raise_for_status()
    return weak_structure(resp.json(), User)


def get_user_by_id(user_id: str) -> User:
    """Get the data for one user from their username
    [API reference](https://docs.manifold.markets/api#get-v0userby-idid)

    Args:
        user_id:
    """
    resp = requests.get(USER_ID_URL.format(user_id))
    resp.raise_for_status()
    return weak_structure(resp.json(), User)


def get_users() -> List[User]:
    """Get all users
    [API reference](https://docs.manifold.markets/api#get-v0users)
    """
    resp = requests.get(USERS_URL)
    resp.raise_for_status()
    return [weak_structure(x, User) for x in resp.json()]


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
        else weak_structure(x, FreeResponseMarket)
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
    return Market.from_json(market)


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

    return [weak_structure(x, Bet) for x in resp.json()]


@define
class APIWrapper:
    key: str

    @property
    def headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": f"Key {self.key}"}

    def _prep_bet(
        self,
        amount: float,
        contractId: str,
        outcome: str,
        limitProb: Optional[float] = None,
    ) -> requests.PreparedRequest:
        """Prepare a bet POST request.
        See `make_bet` for details.
        """
        data = {"amount": amount, "contractId": contractId, "outcome": outcome}
        if limitProb is not None:
            data["limitProb"] = limitProb
        req = requests.Request("POST", MAKE_BET_URL, headers=self.headers, json=data)
        prepped = req.prepare()
        return prepped

    def make_bet(
        self,
        amount: float,
        contractId: str,
        outcome: str,
        limitProb: Optional[float] = None,
    ) -> requests.Response:
        """Make a bet.
        [API reference](https://docs.manifold.markets/api#post-v0bet)
        Args:
            amount: The amount to bet
            contractId: The market id.
            outcome: The outcome to bet on. YES or NO for binary markets
            limitProb: A limit probability for the bet. If spending the full amount would push the market past this probability, then only enough to push the market to this probability will be bought. Any additional funds will be left often as a bet that can later be matched by an opposing offer.
        """
        prepped = self._prep_bet(amount, contractId, outcome, limitProb=limitProb)
        s = requests.Session()
        return s.send(prepped)

    def _prep_create(
        self,
        outcomeType: str,
        question: str,
        description: str,
        closeTime: int,
        tags: Optional[List[str]] = None,
        initialProb: Optional[int] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> requests.PreparedRequest:
        """Prepare a create market POST request
        See `create_market` for details.
        """
        data = {
            "outcomeType": outcomeType,
            "question": question,
            "description": {
                "type": "doc",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": description,
                            },
                        ],
                    },
                ],
            },
            "closeTime": closeTime,
        }
        if tags is not None:
            data["tags"] = tags

        if outcomeType == "BINARY":
            assert initialProb is not None
            assert 1 <= initialProb <= 99
            data["initialProb"] = initialProb

        if outcomeType == "NUMERIC":
            assert min is not None
            assert max is not None
            data["min"] = min
            data["max"] = max
        req = requests.Request(
            "POST", CREATE_MARKET_URL, headers=self.headers, json=data
        )
        prepped = req.prepare()
        return prepped

    def create_market(
        self,
        outcomeType: OutcomeType,
        question: str,
        description: str,
        closeTime: int,
        tags: Optional[List[str]] = None,
        initialProb: Optional[float] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> requests.Response:
        """Create a new market
        [API reference](https://docs.manifold.markets/api#post-v0market)

        Args:
            outcomeType: The kind of market. Must be one of BINARY, FREE_RESPONSE, or NUMERIC
            question: The market question.
            description: Additional details about the market
            closeTime: When the market closes (milliseconds since epoch)
            tags: Any tags for the market.
        """
        prepped = self._prep_create(
            outcomeType,
            question,
            description,
            closeTime,
            tags=tags,
            initialProb=initialProb,
            min=min,
            max=max,
        )
        s = requests.Session()
        return s.send(prepped)

    def _prep_resolve(
        self,
        market_id: str,
        outcome: str,
        probabilityInt: Optional[int] = None,
        resolutions: Optional[List[Any]] = None,
        value: Optional[Any] = None,
    ) -> requests.PreparedRequest:
        """Prepare a resolve market POST request
        See `resolve_market` for details
        """
        # At most one of these should be set
        assert (
            (probabilityInt is not None)
            + (resolutions is not None)
            + (value is not None)
        ) <= 1
        data: Dict[str, Any] = {"outcome": outcome}
        if probabilityInt is not None:
            data["probabilityInt"] = probabilityInt
        elif resolutions is not None:
            data["resolutions"] = resolutions
        elif value is not None:
            data["value"] = value

        req = requests.Request(
            "POST",
            RESOLVE_MARKET_URL.format(market_id),
            headers=self.headers,
            json=data,
        )
        prepped = req.prepare()
        return prepped

    def resolve_market(
        self,
        market_id: str,
        outcome: str,
        probabilityInt: Optional[int] = None,
        resolutions: Optional[List[Any]] = None,
        value: Optional[Any] = None,
    ) -> requests.Response:
        """Resolve an existing market.
        [API reference](https://docs.manifold.markets/api#post-v0marketmarketidresolve)
        Args:
            market_id: The id of the market to resolve.
            outcome: The outcome to resolve with.
            probabilityInt: The probability to resolve with (if outcome is MKT)
            resolutions: An array of responses and weights for each response (for resolving free responses with multiple outcomes)
            value: The value the market resolves to (for numeric markets)
        """
        prepped = self._prep_resolve(
            market_id,
            outcome,
            probabilityInt=probabilityInt,
            resolutions=resolutions,
            value=value,
        )
        s = requests.Session()
        return s.send(prepped)

    def _prep_sell(
        self,
        market_id: str,
        outcome: Optional[str] = None,
        shares: Optional[int] = None,
    ) -> requests.PreparedRequest:
        """Prepare a sell POST request.
        See `sell_shares` for details.
        """
        data: Dict[str, Any] = {}
        if outcome is not None:
            data["outcome"] = outcome
        if shares is not None:
            data["shares"] = shares

        req = requests.Request(
            "POST", SELL_SHARES_URL.format(market_id), headers=self.headers, json=data
        )
        prepped = req.prepare()
        return prepped

    def sell_shares(
        self,
        market_id: str,
        outcome: Optional[str] = None,
        shares: Optional[int] = None,
    ) -> requests.Response:
        """Sell shares in a particular market

        Args:
            market_id: The market to sell shares in
            outcome: The kind of shares to sell. Must be YES or NO.
        """
        prepped = self._prep_sell(market_id, outcome, shares=shares)
        s = requests.Session()
        return s.send(prepped)


def make_bet(
    key: str,
    amount: float,
    contractId: str,
    outcome: str,
    limitProb: Optional[float] = None,
):
    """See `APIWrapper.make_bet`"""
    wrapper = APIWrapper(key)
    return wrapper.make_bet(amount, contractId, outcome, limitProb=limitProb)


def create_market(
    key: str,
    outcomeType: OutcomeType,
    question: str,
    description: str,
    closeTime: int,
    tags: Optional[List[str]] = None,
    initialProb: Optional[float] = None,
    min: Optional[float] = None,
    max: Optional[float] = None,
):
    """See `APIWrapper.create_market`"""
    wrapper = APIWrapper(key)
    return wrapper.create_market(outcomeType, question, description, closeTime, tags)


def resolve_market(
    key: str,
    market_id: str,
    outcome: str,
    probabilityInt: Optional[int] = None,
    resolutions: Optional[List[Any]] = None,
    value: Optional[Any] = None,
):
    """See `APIWrapper.resolve_market`"""
    wrapper = APIWrapper(key)
    return wrapper.resolve_market(
        market_id,
        outcome,
        probabilityInt=probabilityInt,
        resolutions=resolutions,
        value=value,
    )


def sell_shares(key: str, market_id: str, outcome: str, shares: Optional[int] = None):
    """See `APIWrapper.sell_shares`"""
    wrapper = APIWrapper(key)
    return wrapper.sell_shares(market_id, outcome, shares=shares)


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
    else:
        full_markets = {}
        import pdb

        pdb.set_trace()
        pickle.dump(full_markets, config.CACHE_LOC.open("wb"))
    lite_markets = get_all_markets()
    print(f"Fetched {len(lite_markets)} markets")
    cached_ids = {x["market"].id for x in full_markets.values()}
    lite_ids = {x.id for x in lite_markets}
    missing_markets = lite_ids - cached_ids
    print(f"Need {len(missing_markets)} new markets.")
    missed = []
    for i, lmarket_id in enumerate(missing_markets):
        try:
            full_market = get_market(lmarket_id)
            full_markets[full_market.id] = {
                "market": full_market,
                "cache_time": time(),
            }
        # If we get an HTTP Error, just skip that market
        except requests.HTTPError:
            missed.append(lmarket_id)

        if i % cache_every == 0:
            print(i)
            pickle.dump(full_markets, config.CACHE_LOC.open("wb"))
    pickle.dump(full_markets, config.CACHE_LOC.open("wb"))
    market_list = [x["market"] for x in full_markets.values()]
    print(f"Could not get {len(missed)} markets.")
    return market_list
