"""Baseline policies for evaluation and sanity checks."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .cards import Card, PlayedCard
from .observation import PlayerObservation
from .rules import trick_points, trick_winner


@dataclass
class RandomPolicy:
    name: str = "random"

    def action_probabilities(self, obs: PlayerObservation) -> dict[Card, float]:
        probability = 1.0 / len(obs.legal_actions)
        return {card: probability for card in obs.legal_actions}

    def select_action(
        self,
        obs: PlayerObservation,
        rng: random.Random,
        greedy: bool = False,
    ) -> Card:
        return rng.choice(list(obs.legal_actions))


@dataclass
class GreedyTrickPolicy:
    """A myopic policy that only evaluates the current trick."""

    name: str = "greedy_trick"

    def action_probabilities(self, obs: PlayerObservation) -> dict[Card, float]:
        best = self._best_cards(obs)
        probability = 1.0 / len(best)
        return {card: probability if card in best else 0.0 for card in obs.legal_actions}

    def select_action(
        self,
        obs: PlayerObservation,
        rng: random.Random,
        greedy: bool = False,
    ) -> Card:
        return rng.choice(self._best_cards(obs))

    def _best_cards(self, obs: PlayerObservation) -> list[Card]:
        scores = {card: self._score_card(obs, card) for card in obs.legal_actions}
        best_score = max(scores.values())
        return [card for card, score in scores.items() if score == best_score]

    def _score_card(self, obs: PlayerObservation, card: Card) -> float:
        candidate_trick = tuple(obs.current_trick) + (
            PlayedCard(player_id=obs.player_id, card=card),
        )
        current_points = trick_points(candidate_trick)
        winner = trick_winner(candidate_trick, obs.trump_suit).player_id

        # If this card is currently winning, value immediate points. Otherwise,
        # prefer sacrificing low-value cards.
        if winner == obs.player_id:
            return current_points - 0.05 * card.points
        if winner == obs.partner_id:
            return 0.5 * card.points
        return -card.points - 0.1 * card.strength


@dataclass
class SimpleHeuristicPolicy:
    """Small documented heuristic used as an external evaluation reference."""

    name: str = "simple_heuristic"

    def action_probabilities(self, obs: PlayerObservation) -> dict[Card, float]:
        best = self._best_cards(obs)
        probability = 1.0 / len(best)
        return {card: probability if card in best else 0.0 for card in obs.legal_actions}

    def select_action(
        self,
        obs: PlayerObservation,
        rng: random.Random,
        greedy: bool = False,
    ) -> Card:
        return rng.choice(self._best_cards(obs))

    def _best_cards(self, obs: PlayerObservation) -> list[Card]:
        scores = {card: self._score_card(obs, card) for card in obs.legal_actions}
        best_score = max(scores.values())
        return [card for card, score in scores.items() if score == best_score]

    def _score_card(self, obs: PlayerObservation, card: Card) -> float:
        candidate_trick = tuple(obs.current_trick) + (
            PlayedCard(player_id=obs.player_id, card=card),
        )
        candidate_points = trick_points(candidate_trick)
        candidate_winner = trick_winner(candidate_trick, obs.trump_suit).player_id

        card_is_trump = card.suit == obs.trump_suit
        card_is_load = card.points >= 10
        final_phase = obs.deck_count == 0 or obs.trick_index >= 7

        if candidate_winner == obs.partner_id:
            # Load points on the partner if the trick already looks safe.
            return 2.0 * card.points - 0.3 * card_is_trump

        if candidate_winner == obs.player_id:
            # Win valuable tricks, especially late. Avoid wasting a trump on an
            # empty trick unless it is late and necessary.
            score = 2.0 * candidate_points + 0.5 * final_phase
            if card_is_trump and candidate_points <= 2:
                score -= 2.0
            return score

        # If an opponent is currently winning, do not donate expensive cards.
        penalty = 2.0 * card.points + 0.5 * card.strength
        if card_is_load:
            penalty += 5.0
        return -penalty

