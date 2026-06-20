"""Feature extraction for the linear Briscola policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from .cards import Card, PlayedCard, make_deck
from .observation import PlayerObservation
from .rules import trick_points, trick_winner


MAX_CARD_POINTS = 11
MAX_TRICK_POINTS = 44
MAX_DECK_COUNT = 28
TOTAL_POINTS = 120
TOTAL_TRUMPS = 10


@dataclass
class BriscolaFeatureExtractor:
    """Build ``phi(observation, action)`` for the linear softmax policy.

    Every feature is computed from the player's legal observation and the
    candidate card only. Partner-hand features are active only in the final
    phase, when the rules make the partner hand visible. Opponent hands and deck
    order are never used.
    """

    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.feature_names:
            self.feature_names = self._default_feature_names()

    def size(self) -> int:
        return len(self.feature_names)

    def extract(self, obs: PlayerObservation, card: Card) -> list[float]:
        if card not in obs.hand:
            raise ValueError("Features can only be extracted for legal hand cards")

        current_winner = self._current_winner(obs)
        candidate_winner = self._candidate_winner(obs, card)
        observed_cards = self._observed_cards(obs)

        partner_winning = current_winner == obs.partner_id
        opponent_winning = current_winner in obs.opponents
        candidate_wins = candidate_winner == obs.player_id
        candidate_beats_partner = partner_winning and candidate_wins
        candidate_beats_opponent = opponent_winning and candidate_wins

        players_after_me = 3 - obs.position_in_trick
        players_after = [
            (obs.player_id + offset) % 4 for offset in range(1, players_after_me + 1)
        ]
        opponents_after_me = sum(1 for player in players_after if player in obs.opponents)
        partner_still_to_act = float(obs.partner_id in players_after)
        opponent_still_to_act = float(opponents_after_me > 0)
        partner_hand_visible = float(obs.partner_hand_visible)
        partner_hand_points = sum(card.points for card in obs.partner_hand) / 32
        partner_trump_count = sum(1 for card in obs.partner_hand if card.suit == obs.trump_suit)
        partner_high_count = sum(1 for card in obs.partner_hand if card.rank in {"ace", "three", "king"})
        partner_has_trump = float(partner_trump_count > 0)
        partner_has_high_card = float(partner_high_count > 0)
        partner_can_win_now = float(self._partner_can_win_now(obs))

        card_points = card.points / MAX_CARD_POINTS
        trick_value = trick_points(obs.current_trick) / MAX_TRICK_POINTS
        rank_strength = card.strength / 10
        score_diff = obs.score_difference / TOTAL_POINTS
        deck_count = obs.deck_count / MAX_DECK_COUNT

        is_trump = float(card.suit == obs.trump_suit)
        is_high = float(card.rank in {"ace", "three", "king"})
        is_ace_or_three = float(card.rank in {"ace", "three"})
        is_zero = float(card.points == 0)
        risky_card = float(card.points >= 10 or card.suit == obs.trump_suit)

        trumps_played = self._count_public_trumps(obs)
        trumps_observed = self._count_observed_trumps(obs, observed_cards)
        high_cards_played = self._count_public_high_cards(obs)
        aces_played = self._count_public_rank(obs, "ace")
        threes_played = self._count_public_rank(obs, "three")

        stronger_same_suit = self._stronger_same_suit_not_observed(card, observed_cards)
        stronger_trumps = self._stronger_trumps_not_observed(obs.trump_suit, card, observed_cards)

        early_phase = float(obs.trick_index <= 2)
        middle_phase = float(3 <= obs.trick_index <= 6)
        late_phase = float(obs.trick_index >= 7)
        deck_empty = float(obs.deck_count == 0)
        final_tricks = float(obs.trick_index >= 8)

        values = {
            "bias": 1.0,
            "card_points": card_points,
            "rank_strength": rank_strength,
            "is_trump": is_trump,
            "is_ace": float(card.rank == "ace"),
            "is_three": float(card.rank == "three"),
            "is_king_knight_jack": float(card.rank in {"king", "knight", "jack"}),
            "is_zero_point": is_zero,
            "is_high_card": is_high,
            "is_low_card": float(card.points == 0 and not is_trump),
            "position_first": float(obs.position_in_trick == 0),
            "position_second": float(obs.position_in_trick == 1),
            "position_third": float(obs.position_in_trick == 2),
            "position_fourth": float(obs.position_in_trick == 3),
            "cards_in_trick": obs.position_in_trick / 3,
            "trick_points": trick_value,
            "partner_winning": float(partner_winning),
            "opponent_winning": float(opponent_winning),
            "candidate_wins_now": float(candidate_wins),
            "candidate_beats_partner": float(candidate_beats_partner),
            "candidate_beats_opponent": float(candidate_beats_opponent),
            "players_after_me": players_after_me / 3,
            "opponents_after_me": opponents_after_me / 2,
            "partner_still_to_act": partner_still_to_act,
            "opponent_still_to_act": opponent_still_to_act,
            "partner_hand_visible": partner_hand_visible,
            "partner_hand_points": partner_hand_points,
            "partner_trump_count": partner_trump_count / 3,
            "partner_high_count": partner_high_count / 3,
            "partner_has_trump": partner_has_trump,
            "partner_has_high_card": partner_has_high_card,
            "partner_can_win_now": partner_can_win_now,
            "cards_played": obs.cards_played_count / 40,
            "trumps_played": trumps_played / TOTAL_TRUMPS,
            "trumps_not_observed": (TOTAL_TRUMPS - trumps_observed) / TOTAL_TRUMPS,
            "aces_played": aces_played / 4,
            "threes_played": threes_played / 4,
            "high_cards_played": high_cards_played / 12,
            "stronger_same_suit_not_observed": stronger_same_suit / 9,
            "stronger_trumps_not_observed": stronger_trumps / 9,
            "team_score": obs.team_score / TOTAL_POINTS,
            "opponent_score": obs.opponent_score / TOTAL_POINTS,
            "score_difference": score_diff,
            "team_ahead": float(score_diff > 0),
            "team_behind": float(score_diff < 0),
            "deck_count": deck_count,
            "early_phase": early_phase,
            "middle_phase": middle_phase,
            "late_phase": late_phase,
            "deck_empty": deck_empty,
            "final_tricks": final_tricks,
            # Interaction features let a linear policy represent conditional
            # decisions such as "use trump when the opponent is winning points".
            "trump_x_trick_points": is_trump * trick_value,
            "trump_x_opponent_winning": is_trump * float(opponent_winning),
            "trump_x_partner_winning": is_trump * float(partner_winning),
            "high_card_x_final_phase": is_high * final_tricks,
            "partner_winning_x_card_points": float(partner_winning) * card_points,
            "opponent_winning_x_can_win": float(opponent_winning) * float(candidate_wins),
            "can_win_x_trick_points": float(candidate_wins) * trick_value,
            "score_advantage_x_final_phase": max(score_diff, 0.0) * final_tricks,
            "score_deficit_x_final_phase": max(-score_diff, 0.0) * final_tricks,
            "players_after_me_x_risky_card": (players_after_me / 3) * risky_card,
            "opponents_after_me_x_high_card": (opponents_after_me / 2) * is_high,
            "partner_still_to_act_x_card_points": partner_still_to_act * card_points,
            "partner_visible_x_card_points": partner_hand_visible * card_points,
            "partner_can_win_x_card_points": partner_can_win_now * card_points,
            "partner_has_trump_x_opponent_winning": partner_has_trump * float(opponent_winning),
            "deck_empty_x_high_card": deck_empty * is_high,
            "deck_empty_x_trump": deck_empty * is_trump,
        }

        return [values[name] for name in self.feature_names]

    def _default_feature_names(self) -> list[str]:
        return [
            "bias",
            "card_points",
            "rank_strength",
            "is_trump",
            "is_ace",
            "is_three",
            "is_king_knight_jack",
            "is_zero_point",
            "is_high_card",
            "is_low_card",
            "position_first",
            "position_second",
            "position_third",
            "position_fourth",
            "cards_in_trick",
            "trick_points",
            "partner_winning",
            "opponent_winning",
            "candidate_wins_now",
            "candidate_beats_partner",
            "candidate_beats_opponent",
            "players_after_me",
            "opponents_after_me",
            "partner_still_to_act",
            "opponent_still_to_act",
            "partner_hand_visible",
            "partner_hand_points",
            "partner_trump_count",
            "partner_high_count",
            "partner_has_trump",
            "partner_has_high_card",
            "partner_can_win_now",
            "cards_played",
            "trumps_played",
            "trumps_not_observed",
            "aces_played",
            "threes_played",
            "high_cards_played",
            "stronger_same_suit_not_observed",
            "stronger_trumps_not_observed",
            "team_score",
            "opponent_score",
            "score_difference",
            "team_ahead",
            "team_behind",
            "deck_count",
            "early_phase",
            "middle_phase",
            "late_phase",
            "deck_empty",
            "final_tricks",
            "trump_x_trick_points",
            "trump_x_opponent_winning",
            "trump_x_partner_winning",
            "high_card_x_final_phase",
            "partner_winning_x_card_points",
            "opponent_winning_x_can_win",
            "can_win_x_trick_points",
            "score_advantage_x_final_phase",
            "score_deficit_x_final_phase",
            "players_after_me_x_risky_card",
            "opponents_after_me_x_high_card",
            "partner_still_to_act_x_card_points",
            "partner_visible_x_card_points",
            "partner_can_win_x_card_points",
            "partner_has_trump_x_opponent_winning",
            "deck_empty_x_high_card",
            "deck_empty_x_trump",
        ]

    def _current_winner(self, obs: PlayerObservation) -> int | None:
        if not obs.current_trick:
            return None
        return trick_winner(obs.current_trick, obs.trump_suit).player_id

    def _candidate_winner(self, obs: PlayerObservation, card: Card) -> int:
        return self._candidate_winner_for_player(obs, card, obs.player_id)

    def _candidate_winner_for_player(
        self,
        obs: PlayerObservation,
        card: Card,
        player_id: int,
    ) -> int:
        candidate_trick = tuple(obs.current_trick) + (
            PlayedCard(player_id=player_id, card=card),
        )
        return trick_winner(candidate_trick, obs.trump_suit).player_id

    def _observed_cards(self, obs: PlayerObservation) -> set[Card]:
        observed = set(obs.hand)
        if obs.partner_hand_visible:
            observed.update(obs.partner_hand)
        observed.update(play.card for play in obs.played_cards)
        if obs.revealed_trump is not None:
            observed.add(obs.revealed_trump)
        return observed

    def _partner_can_win_now(self, obs: PlayerObservation) -> bool:
        if not obs.partner_hand_visible or not obs.partner_hand:
            return False
        return any(
            self._candidate_winner_for_player(obs, card, obs.partner_id) == obs.partner_id
            for card in obs.partner_hand
        )

    def _count_public_trumps(self, obs: PlayerObservation) -> int:
        return sum(1 for play in obs.played_cards if play.card.suit == obs.trump_suit)

    def _count_observed_trumps(self, obs: PlayerObservation, observed: set[Card]) -> int:
        return sum(1 for card in observed if card.suit == obs.trump_suit)

    def _count_public_high_cards(self, obs: PlayerObservation) -> int:
        return sum(1 for play in obs.played_cards if play.card.rank in {"ace", "three", "king"})

    def _count_public_rank(self, obs: PlayerObservation, rank: str) -> int:
        return sum(1 for play in obs.played_cards if play.card.rank == rank)

    def _stronger_same_suit_not_observed(self, card: Card, observed: set[Card]) -> int:
        return sum(
            1
            for other in make_deck()
            if other.suit == card.suit
            and other.strength > card.strength
            and other not in observed
        )

    def _stronger_trumps_not_observed(
        self,
        trump_suit: str,
        card: Card,
        observed: set[Card],
    ) -> int:
        if card.suit != trump_suit:
            return sum(
                1
                for other in make_deck()
                if other.suit == trump_suit and other not in observed
            )
        return sum(
            1
            for other in make_deck()
            if other.suit == trump_suit
            and other.strength > card.strength
            and other not in observed
        )
