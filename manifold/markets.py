from attr import define, field
from typing import List, Optional, TypeVar, Type, Any


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
    def from_json(cls, json: Any) -> "Bet":
        return cls(**json)  # type: ignore


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

    @classmethod
    def from_json(cls: Type[MarketT], json: Any) -> MarketT:
        return cls(**json)  # type: ignore


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