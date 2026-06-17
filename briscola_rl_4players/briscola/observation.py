"""Legal player observations.

The environment owns the hidden state. Policies only receive ``PlayerObservation``
objects, which deliberately omit opponent hands and deck order.

Project rule: during the final phase, when the draw deck is empty and each team
is playing the last three tricks, teammates may see each other's current hand.
Before that phase, the partner hand is still hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cards import Card, PlayedCard


@dataclass(frozen=True)
class PlayerObservation:
    """Information legally available to one player at a decision point."""

    player_id: int
    partner_id: int
    left_opponent_id: int
    right_opponent_id: int
    hand: tuple[Card, ...]
    partner_hand_visible: bool
    partner_hand: tuple[Card, ...]
    trump_suit: str
    revealed_trump: Card | None
    current_trick: tuple[PlayedCard, ...]
    played_cards: tuple[PlayedCard, ...]
    trick_winners: tuple[int, ...]
    team_id: int
    opponent_team_id: int
    team_score: int
    opponent_score: int
    trick_leader: int
    current_player: int
    deck_count: int
    trick_index: int
    position_in_trick: int

    @property
    def legal_actions(self) -> tuple[Card, ...]:
        return self.hand

    @property
    def score_difference(self) -> int:
        return self.team_score - self.opponent_score

    @property
    def cards_played_count(self) -> int:
        return len(self.played_cards)

    @property
    def opponents(self) -> tuple[int, int]:
        return (self.left_opponent_id, self.right_opponent_id)
