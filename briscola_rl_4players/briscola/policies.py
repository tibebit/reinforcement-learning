"""Policies used by training and evaluation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

from .cards import Card
from .features import BriscolaFeatureExtractor
from .observation import PlayerObservation


class Policy(Protocol):
    """Minimal policy interface used by the environment runners."""

    name: str

    def action_probabilities(self, obs: PlayerObservation) -> dict[Card, float]:
        ...

    def select_action(
        self,
        obs: PlayerObservation,
        rng: random.Random,
        greedy: bool = False,
    ) -> Card:
        ...


@dataclass
class LinearSoftmaxPolicy:
    """Linear softmax policy with action masking.

    The architecture is shared by every player. Different players may still use
    different parameter vectors, for example current learner theta versus frozen
    historical theta-minus snapshots.
    """

    theta: list[float]
    feature_extractor: BriscolaFeatureExtractor = field(default_factory=BriscolaFeatureExtractor)
    name: str = "linear_softmax"

    @classmethod
    def initialize(
        cls,
        feature_extractor: BriscolaFeatureExtractor | None = None,
        rng: random.Random | None = None,
        scale: float = 0.01,
        name: str = "linear_softmax",
    ) -> "LinearSoftmaxPolicy":
        feature_extractor = feature_extractor or BriscolaFeatureExtractor()
        rng = rng or random.Random()
        theta = [rng.uniform(-scale, scale) for _ in range(feature_extractor.size())]
        return cls(theta=theta, feature_extractor=feature_extractor, name=name)

    def copy(self, name: str | None = None) -> "LinearSoftmaxPolicy":
        return LinearSoftmaxPolicy(
            theta=list(self.theta),
            feature_extractor=self.feature_extractor,
            name=name or self.name,
        )

    def action_preferences(self, obs: PlayerObservation) -> dict[Card, float]:
        preferences: dict[Card, float] = {}
        for card in obs.legal_actions:
            features = self.feature_extractor.extract(obs, card)
            preferences[card] = dot(self.theta, features)
        return preferences

    def action_probabilities(self, obs: PlayerObservation) -> dict[Card, float]:
        preferences = self.action_preferences(obs)
        if not preferences:
            raise ValueError("No legal actions available")

        # Stable softmax: subtract the max preference before exponentiating.
        max_pref = max(preferences.values())
        exp_values = {
            card: math.exp(value - max_pref) for card, value in preferences.items()
        }
        total = sum(exp_values.values())
        return {card: value / total for card, value in exp_values.items()}

    def select_action(
        self,
        obs: PlayerObservation,
        rng: random.Random,
        greedy: bool = False,
    ) -> Card:
        probabilities = self.action_probabilities(obs)
        if greedy:
            return max(probabilities, key=probabilities.get)

        threshold = rng.random()
        cumulative = 0.0
        last_card = next(iter(probabilities))
        for card, probability in probabilities.items():
            cumulative += probability
            last_card = card
            if threshold <= cumulative:
                return card
        return last_card

    def log_probability(self, obs: PlayerObservation, action: Card) -> float:
        probability = self.action_probabilities(obs)[action]
        return math.log(max(probability, 1e-15))

    def grad_log_probability(self, obs: PlayerObservation, action: Card) -> list[float]:
        """Gradient of log pi(action | observation) for linear softmax."""

        probabilities = self.action_probabilities(obs)
        action_features = self.feature_extractor.extract(obs, action)
        expected_features = [0.0 for _ in self.theta]

        for card, probability in probabilities.items():
            features = self.feature_extractor.extract(obs, card)
            for index, value in enumerate(features):
                expected_features[index] += probability * value

        return [
            action_features[index] - expected_features[index]
            for index in range(len(self.theta))
        ]

    def apply_gradient(
        self,
        gradient: list[float],
        learning_rate: float,
        max_update_norm: float | None = None,
    ) -> None:
        if max_update_norm is not None:
            gradient = clip_vector(gradient, max_update_norm)
        for index, value in enumerate(gradient):
            self.theta[index] += learning_rate * value


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def add_scaled_in_place(target: list[float], source: list[float], scale: float) -> None:
    for index, value in enumerate(source):
        target[index] += scale * value


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def clip_vector(vector: list[float], max_norm: float) -> list[float]:
    norm = vector_norm(vector)
    if norm <= max_norm or norm == 0:
        return vector
    scale = max_norm / norm
    return [value * scale for value in vector]

