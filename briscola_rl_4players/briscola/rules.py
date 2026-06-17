"""Pure Briscola rules.

This module contains no hidden state and no learning logic. Keeping rules pure
makes them easy to test and reuse from the environment, baselines, and features.
"""

from __future__ import annotations

from .cards import Card, PlayedCard


TEAM_A = 0
TEAM_B = 1
NUM_PLAYERS = 4


def team_of(player_id: int) -> int:
    """Return the team id for a zero-based player id.

    Players 0 and 2 form Team A. Players 1 and 3 form Team B.
    """

    if player_id not in range(NUM_PLAYERS):
        raise ValueError(f"Invalid player id: {player_id}")
    return TEAM_A if player_id % 2 == 0 else TEAM_B


def partner_of(player_id: int) -> int:
    return (player_id + 2) % NUM_PLAYERS


def left_opponent_of(player_id: int) -> int:
    return (player_id + 1) % NUM_PLAYERS


def right_opponent_of(player_id: int) -> int:
    return (player_id - 1) % NUM_PLAYERS


def table_order_from(player_id: int) -> list[int]:
    """Return the four players in table order starting from ``player_id``."""

    return [(player_id + offset) % NUM_PLAYERS for offset in range(NUM_PLAYERS)]


def next_player(player_id: int) -> int:
    return (player_id + 1) % NUM_PLAYERS


def trick_points(trick: tuple[PlayedCard, ...] | list[PlayedCard]) -> int:
    return sum(play.card.points for play in trick)


def card_beats(
    challenger: Card,
    incumbent: Card,
    lead_suit: str,
    trump_suit: str,
) -> bool:
    """Return whether ``challenger`` beats ``incumbent`` in a trick."""

    challenger_is_trump = challenger.suit == trump_suit
    incumbent_is_trump = incumbent.suit == trump_suit

    if challenger_is_trump and not incumbent_is_trump:
        return True
    if incumbent_is_trump and not challenger_is_trump:
        return False

    if challenger_is_trump and incumbent_is_trump:
        return challenger.strength > incumbent.strength

    # With no trump involved, only cards of the leading suit can win.
    if challenger.suit == lead_suit and incumbent.suit != lead_suit:
        return True
    if challenger.suit != lead_suit:
        return False

    return challenger.strength > incumbent.strength


def trick_winner(
    trick: tuple[PlayedCard, ...] | list[PlayedCard],
    trump_suit: str,
) -> PlayedCard:
    """Return the public play currently winning the trick."""

    if not trick:
        raise ValueError("Cannot resolve an empty trick")

    lead_suit = trick[0].card.suit
    winner = trick[0]
    for play in trick[1:]:
        if card_beats(play.card, winner.card, lead_suit, trump_suit):
            winner = play
    return winner

