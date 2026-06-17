"""Evaluation helpers for Briscola policies."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean, pstdev

from .env import FourPlayerBriscolaEnv
from .policies import Policy


@dataclass(frozen=True)
class MatchResult:
    scores: tuple[int, int]
    winner_team: int | None
    point_diff_team_a: int


@dataclass(frozen=True)
class EvaluationResult:
    games: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    mean_point_difference: float
    standard_error: float
    confidence_interval_95: tuple[float, float]


def play_match(
    policies_by_player: dict[int, Policy],
    seed: int,
    start_player: int | None = None,
    greedy: bool = True,
) -> MatchResult:
    """Run one match with fixed policies and no learning."""

    if start_player is None:
        start_player = seed % 4
    env = FourPlayerBriscolaEnv(seed=seed, start_player=start_player)
    rng = random.Random(seed + 20_000)

    while not env.done:
        player_id = env.current_player
        obs = env.observe(player_id)
        action = policies_by_player[player_id].select_action(obs, rng, greedy=greedy)
        env.step(action)

    return MatchResult(
        scores=tuple(env.scores),
        winner_team=env.winner_team(),
        point_diff_team_a=env.scores[0] - env.scores[1],
    )


def evaluate_team_match(
    policy_a: Policy,
    policy_b: Policy,
    seeds: list[int],
    paired: bool = True,
    greedy: bool = True,
) -> EvaluationResult:
    """Evaluate policy A against policy B from A's perspective."""

    point_differences: list[int] = []
    outcomes: list[int] = []

    for seed in seeds:
        result = play_match(
            policies_by_player={0: policy_a, 1: policy_b, 2: policy_a, 3: policy_b},
            seed=seed,
            greedy=greedy,
        )
        point_differences.append(result.point_diff_team_a)
        outcomes.append(_outcome_from_difference(result.point_diff_team_a))

        if paired:
            swapped = play_match(
                policies_by_player={0: policy_b, 1: policy_a, 2: policy_b, 3: policy_a},
                seed=seed,
                greedy=greedy,
            )
            # Here policy A is Team B, so the perspective is the negative Team A
            # score difference.
            perspective_diff = -swapped.point_diff_team_a
            point_differences.append(perspective_diff)
            outcomes.append(_outcome_from_difference(perspective_diff))

    games = len(point_differences)
    wins = sum(1 for outcome in outcomes if outcome > 0)
    draws = sum(1 for outcome in outcomes if outcome == 0)
    losses = sum(1 for outcome in outcomes if outcome < 0)
    avg = mean(point_differences)
    stderr = _standard_error(point_differences)
    interval = (avg - 1.96 * stderr, avg + 1.96 * stderr)

    return EvaluationResult(
        games=games,
        win_rate=wins / games,
        draw_rate=draws / games,
        loss_rate=losses / games,
        mean_point_difference=avg,
        standard_error=stderr,
        confidence_interval_95=interval,
    )


def matchup_matrix(
    policies: dict[str, Policy],
    seeds: list[int],
    greedy: bool = True,
) -> dict[tuple[str, str], EvaluationResult]:
    """Build a pairwise evaluation matrix."""

    matrix: dict[tuple[str, str], EvaluationResult] = {}
    for name_a, policy_a in policies.items():
        for name_b, policy_b in policies.items():
            matrix[(name_a, name_b)] = evaluate_team_match(
                policy_a=policy_a,
                policy_b=policy_b,
                seeds=seeds,
                paired=True,
                greedy=greedy,
            )
    return matrix


def _outcome_from_difference(point_difference: int) -> int:
    if point_difference > 0:
        return 1
    if point_difference < 0:
        return -1
    return 0


def _standard_error(values: list[int]) -> float:
    if len(values) <= 1:
        return 0.0
    return pstdev(values) / math.sqrt(len(values))

