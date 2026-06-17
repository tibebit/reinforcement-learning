"""Four-player Briscola environment with hidden internal state."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .cards import Card, PlayedCard, make_deck
from .observation import PlayerObservation
from .rules import (
    NUM_PLAYERS,
    left_opponent_of,
    next_player,
    partner_of,
    right_opponent_of,
    table_order_from,
    team_of,
    trick_points,
    trick_winner,
)


@dataclass(frozen=True)
class StepResult:
    """Result returned by ``env.step``."""

    observation: PlayerObservation | None
    done: bool
    trick_completed: bool
    trick_winner: int | None
    trick_points: int
    scores: tuple[int, int]


@dataclass
class FourPlayerBriscolaEnv:
    """Environment for four-player Briscola with drawing.

    The code uses zero-based players:

    ```text
    player 0 and player 2: Team A
    player 1 and player 3: Team B
    ```
    """

    seed: int | None = None
    start_player: int = 0
    rng: random.Random = field(init=False)
    hands: list[list[Card]] = field(init=False)
    deck: list[Card] = field(init=False)
    trump_suit: str = field(init=False)
    revealed_trump: Card | None = field(init=False)
    current_trick: list[PlayedCard] = field(init=False)
    played_cards: list[PlayedCard] = field(init=False)
    trick_winners: list[int] = field(init=False)
    scores: list[int] = field(init=False)
    trick_leader: int = field(init=False)
    current_player: int = field(init=False)
    trick_index: int = field(init=False)
    done: bool = field(init=False)

    def __post_init__(self) -> None:
        self.reset(self.seed, self.start_player)

    def reset(
        self,
        seed: int | None = None,
        start_player: int | None = None,
    ) -> PlayerObservation:
        """Start a new match and return the first player's observation."""

        if seed is not None:
            self.seed = seed
        if start_player is not None:
            self.start_player = start_player

        self.rng = random.Random(self.seed)
        self.hands = [[] for _ in range(NUM_PLAYERS)]
        self.deck = make_deck()
        self.rng.shuffle(self.deck)

        # Initial deal: three cards to each player in table order.
        for _ in range(3):
            for player_id in range(NUM_PLAYERS):
                self.hands[player_id].append(self.deck.pop(0))

        # Reveal trump and place it at the bottom of the draw deck, so it is
        # visible now and drawn as the last card.
        self.revealed_trump = self.deck.pop(0)
        self.trump_suit = self.revealed_trump.suit
        self.deck.append(self.revealed_trump)

        self.current_trick = []
        self.played_cards = []
        self.trick_winners = []
        self.scores = [0, 0]
        self.trick_leader = self.start_player
        self.current_player = self.start_player
        self.trick_index = 0
        self.done = False
        return self.observe(self.current_player)

    def observe(self, player_id: int) -> PlayerObservation:
        """Return a legal observation for ``player_id``.

        This method is the only public observation API policies should use.
        """

        player_team = team_of(player_id)
        opponent_team = 1 - player_team
        partner_hand_visible = self.partner_hand_visible()
        partner_id = partner_of(player_id)
        return PlayerObservation(
            player_id=player_id,
            partner_id=partner_id,
            left_opponent_id=left_opponent_of(player_id),
            right_opponent_id=right_opponent_of(player_id),
            hand=tuple(self.hands[player_id]),
            partner_hand_visible=partner_hand_visible,
            partner_hand=tuple(self.hands[partner_id]) if partner_hand_visible else (),
            trump_suit=self.trump_suit,
            revealed_trump=self.revealed_trump,
            current_trick=tuple(self.current_trick),
            played_cards=tuple(self.played_cards),
            trick_winners=tuple(self.trick_winners),
            team_id=player_team,
            opponent_team_id=opponent_team,
            team_score=self.scores[player_team],
            opponent_score=self.scores[opponent_team],
            trick_leader=self.trick_leader,
            current_player=self.current_player,
            deck_count=len(self.deck),
            trick_index=self.trick_index,
            position_in_trick=len(self.current_trick),
        )

    def partner_hand_visible(self) -> bool:
        """Return whether teammates can legally see each other's cards.

        In this variant, teammate cards become public inside the team when the
        draw deck is empty. This happens after the seventh trick has been
        resolved and the last cards have been drawn, leaving the final three
        tricks to play.
        """

        return len(self.deck) == 0

    def legal_actions(self, player_id: int | None = None) -> tuple[Card, ...]:
        if player_id is None:
            player_id = self.current_player
        return tuple(self.hands[player_id])

    def step(self, card: Card) -> StepResult:
        """Play ``card`` for the current player."""

        if self.done:
            raise RuntimeError("Cannot step a finished match")

        player_id = self.current_player
        if card not in self.hands[player_id]:
            raise ValueError(f"Illegal action: player {player_id} does not hold {card}")

        self.hands[player_id].remove(card)
        play = PlayedCard(player_id=player_id, card=card)
        self.current_trick.append(play)
        self.played_cards.append(play)

        trick_completed = len(self.current_trick) == NUM_PLAYERS
        completed_winner: int | None = None
        completed_points = 0

        if trick_completed:
            winning_play = trick_winner(self.current_trick, self.trump_suit)
            completed_winner = winning_play.player_id
            completed_points = trick_points(self.current_trick)
            self.scores[team_of(completed_winner)] += completed_points
            self.trick_winners.append(completed_winner)

            self._draw_after_trick(completed_winner)

            self.current_trick = []
            self.trick_index += 1
            self.trick_leader = completed_winner
            self.current_player = completed_winner
        else:
            self.current_player = next_player(player_id)

        self.done = self._is_terminal()
        observation = None if self.done else self.observe(self.current_player)
        return StepResult(
            observation=observation,
            done=self.done,
            trick_completed=trick_completed,
            trick_winner=completed_winner,
            trick_points=completed_points,
            scores=tuple(self.scores),
        )

    def _draw_after_trick(self, winner: int) -> None:
        """Draw one card per player, starting from the trick winner."""

        if not self.deck:
            return

        for player_id in table_order_from(winner):
            if not self.deck:
                break
            drawn = self.deck.pop(0)
            self.hands[player_id].append(drawn)
            if self.revealed_trump == drawn:
                self.revealed_trump = None

    def _is_terminal(self) -> bool:
        return (
            not self.deck
            and not self.current_trick
            and all(len(hand) == 0 for hand in self.hands)
            and self.trick_index == 10
        )

    def winner_team(self) -> int | None:
        """Return winning team id, or ``None`` for a 60-60 tie."""

        if self.scores[0] == self.scores[1]:
            return None
        return 0 if self.scores[0] > self.scores[1] else 1

    def score_difference_for_team(self, team_id: int) -> int:
        return self.scores[team_id] - self.scores[1 - team_id]

    def cards_played_by_player(self) -> list[int]:
        counts = [0 for _ in range(NUM_PLAYERS)]
        for play in self.played_cards:
            counts[play.player_id] += 1
        return counts

    def assert_invariants(self) -> None:
        """Raise an error if a core game invariant is violated."""

        all_cards: list[Card] = []
        for hand in self.hands:
            all_cards.extend(hand)
        all_cards.extend(self.deck)
        all_cards.extend(play.card for play in self.played_cards)

        if len(all_cards) != 40:
            raise AssertionError(f"Expected 40 total cards, found {len(all_cards)}")
        if len(set(all_cards)) != 40:
            raise AssertionError("Duplicate card detected")
        if self.done and sum(self.scores) != 120:
            raise AssertionError(f"Final score must sum to 120, got {self.scores}")
        if self.done and self.cards_played_by_player() != [10, 10, 10, 10]:
            raise AssertionError("Each player must play exactly 10 cards")
