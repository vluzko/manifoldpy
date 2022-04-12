import attr
import requests

# from dataclasses import dataclass, field
from attr import define, field
from typing import List, Optional, Tuple, TypeVar, Type


MarketT = TypeVar("MarketT", bound="Market")


@define
class Bet:
    """A single bet"""

    contractId: str
    createdTime: int
    isAnte: bool
    shares: float
    userId: str
    amount: int
    probAfter: float
    probBefore: float
    id: str
    outcome: str

    @classmethod
    def from_json(cls, json) -> "Bet":
        return cls(**json)


@define
class Comment:
    contractId: str
    userUsername: str
    userAvatarUrl: str
    userId: str
    text: str
    createdTime: int
    betId: str
    userName: str

    @classmethod
    def from_json(cls, json) -> "Comment":
        return cls(**json)


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

    @classmethod
    def from_json(cls: Type[MarketT], json) -> MarketT:
        return cls(**json)


@define
class BinaryMarket(Market):
    """A market with a binary resolution
    Attributes:
        probability: The current resolution probability
        p: Appears to also be resolution probability. Isn't present on newer markets, I assume this is deprecated
        totalLiquidity:
    """

    probability: float
    p: Optional[float] = None
    totalLiquidity: Optional[float] = None


@define
class MultiMarket(Market):
    """A market with multiple possible resolutions"""


def get_markets() -> Tuple[List[BinaryMarket], List[MultiMarket]]:
    URL = "https://manifold.markets/api/v0/markets"
    json = requests.get(URL).json()

    # If this fails, the code is out of date.
    all_mechanisms = {x["mechanism"] for x in json}
    assert all_mechanisms == {"cpmm-1", "dpm-2"}

    binary_markets = [BinaryMarket.from_json(x) for x in json if "probability" in x]
    multi_markets = [MultiMarket.from_json(x) for x in json if "probability" not in x]

    return binary_markets, multi_markets


def get_market(market_id: str) -> Market:
    URL = f"https://manifold.markets/api/v0/market/{market_id}"
    market = requests.get(URL).json()
    if "probability" in market:
        return BinaryMarket.from_json(market)
    else:
        return MultiMarket.from_json(market)
