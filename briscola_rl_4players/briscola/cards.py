"""Card definitions for the Italian 40-card Briscola deck.

The code uses English rank and suit names to keep identifiers ASCII-only and
easy to serialize. The mathematical rules are the standard Briscola rules.
"""

from __future__ import annotations

from dataclasses import dataclass


SUITS: tuple[str, ...] = ("cups", "coins", "clubs", "swords")

# Order is strongest to weakest for taking a trick.
RANKS: tuple[str, ...] = (
    "ace",
    "three",
    "king",
    "knight",
    "jack",
    "seven",
    "six",
    "five",
    "four",
    "two",
)

POINTS: dict[str, int] = {
    "ace": 11,
    "three": 10,
    "king": 4,
    "knight": 3,
    "jack": 2,
    "seven": 0,
    "six": 0,
    "five": 0,
    "four": 0,
    "two": 0,
}

RANK_TO_STRENGTH: dict[str, int] = {
    rank: len(RANKS) - index for index, rank in enumerate(RANKS)
}


@dataclass(frozen=True, order=True)
class Card:
    """A single card.

    Cards are immutable so they can safely be used in sets, dictionaries, and
    replay records.
    """

    suit: str
    rank: str

    def __post_init__(self) -> None:
        if self.suit not in SUITS:
            raise ValueError(f"Unknown suit: {self.suit}")
        if self.rank not in RANKS:
            raise ValueError(f"Unknown rank: {self.rank}")

    @property
    def points(self) -> int:
        return POINTS[self.rank]

    @property
    def strength(self) -> int:
        return RANK_TO_STRENGTH[self.rank]

    @property
    def id(self) -> str:
        return f"{self.rank}_of_{self.suit}"

    def short(self) -> str:
        """Compact display form used in logs and notebooks."""

        return f"{self.rank[0].upper()}{self.suit[0].upper()}"


@dataclass(frozen=True)
class PlayedCard:
    """A public card play made by a player."""

    player_id: int
    card: Card


def make_deck() -> list[Card]:
    """Return a fresh ordered 40-card deck."""

    return [Card(suit=suit, rank=rank) for suit in SUITS for rank in RANKS]


def card_from_id(card_id: str) -> Card:
    """Parse a card id created by ``Card.id``."""

    rank, _, suit = card_id.partition("_of_")
    if not suit:
        raise ValueError(f"Invalid card id: {card_id}")
    return Card(suit=suit, rank=rank)


def total_deck_points() -> int:
    return sum(card.points for card in make_deck())

