"""REINFORCE training utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean

from .cards import Card
from .env import FourPlayerBriscolaEnv
from .observation import PlayerObservation
from .policies import LinearSoftmaxPolicy, Policy, add_scaled_in_place
from .rules import partner_of, team_of


@dataclass(frozen=True)
class RewardConfig:
    """Reward configuration used by training."""

    mode: str = "combined_terminal"
    lambda_margin: float = 0.2


@dataclass
class TrajectoryStep:
    """One learner decision stored for policy-gradient updates."""

    observation: PlayerObservation
    action: Card
    global_index: int
    reward_to_go: float = 0.0


@dataclass
class EpisodeResult:
    steps: list[TrajectoryStep]
    rewards: list[float]
    final_scores: tuple[int, int]
    learner_seat: int
    learner_team: int
    episode_return: float


@dataclass
class TrainStats:
    episodes: int
    learner_decisions: int
    mean_return: float
    mean_score_diff: float
    gradient_norm: float


def collect_episode(
    learner_policy: LinearSoftmaxPolicy,
    partner_policy: Policy,
    opponent_policy: Policy,
    seed: int,
    learner_seat: int,
    reward_config: RewardConfig,
    greedy_non_learner: bool = False,
) -> EpisodeResult:
    """Play one full match and collect only the active learner's decisions."""

    env = FourPlayerBriscolaEnv(seed=seed, start_player=seed % 4)
    policy_rng = random.Random(seed + 10_000)
    learner_team = team_of(learner_seat)
    partner_seat = partner_of(learner_seat)

    steps: list[TrajectoryStep] = []
    rewards: list[float] = []
    global_index = 0

    while not env.done:
        player_id = env.current_player
        obs = env.observe(player_id)

        if player_id == learner_seat:
            action = learner_policy.select_action(obs, policy_rng, greedy=False)
            steps.append(
                TrajectoryStep(
                    observation=obs,
                    action=action,
                    global_index=global_index,
                )
            )
        elif player_id == partner_seat:
            action = partner_policy.select_action(
                obs,
                policy_rng,
                greedy=greedy_non_learner,
            )
        else:
            action = opponent_policy.select_action(
                obs,
                policy_rng,
                greedy=greedy_non_learner,
            )

        result = env.step(action)
        immediate_reward = 0.0

        if reward_config.mode == "trick_point_dense" and result.trick_completed:
            winner_team = team_of(result.trick_winner)  # type: ignore[arg-type]
            sign = 1.0 if winner_team == learner_team else -1.0
            immediate_reward = sign * result.trick_points

        if result.done and reward_config.mode in {"terminal_win", "combined_terminal"}:
            immediate_reward = terminal_reward(
                scores=result.scores,
                learner_team=learner_team,
                reward_config=reward_config,
            )

        rewards.append(immediate_reward)
        global_index += 1

    for step in steps:
        step.reward_to_go = sum(rewards[step.global_index :])

    score_diff = env.score_difference_for_team(learner_team)
    episode_return = steps[0].reward_to_go if steps else 0.0
    return EpisodeResult(
        steps=steps,
        rewards=rewards,
        final_scores=tuple(env.scores),
        learner_seat=learner_seat,
        learner_team=learner_team,
        episode_return=episode_return,
    )


def terminal_reward(
    scores: tuple[int, int],
    learner_team: int,
    reward_config: RewardConfig,
) -> float:
    team_score = scores[learner_team]
    opponent_score = scores[1 - learner_team]
    difference = team_score - opponent_score

    if reward_config.mode == "terminal_win":
        if difference > 0:
            return 1.0
        if difference < 0:
            return -1.0
        return 0.0

    if reward_config.mode == "combined_terminal":
        if difference > 0:
            sign = 1.0
        elif difference < 0:
            sign = -1.0
        else:
            sign = 0.0
        return sign + reward_config.lambda_margin * (difference / 120.0)

    raise ValueError(f"Unsupported terminal reward mode: {reward_config.mode}")


def reinforce_update(
    policy: LinearSoftmaxPolicy,
    episodes: list[EpisodeResult],
    learning_rate: float,
    baseline: str = "batch_mean",
    max_update_norm: float | None = 5.0,
) -> TrainStats:
    """Apply one REINFORCE update from a batch of collected episodes."""

    steps = [step for episode in episodes for step in episode.steps]
    if not steps:
        raise ValueError("Cannot update from an empty learner trajectory")

    if baseline == "batch_mean":
        baseline_value = mean(step.reward_to_go for step in steps)
    elif baseline == "none":
        baseline_value = 0.0
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")

    gradient = [0.0 for _ in policy.theta]

    for step in steps:
        advantage = step.reward_to_go - baseline_value
        grad_log_prob = policy.grad_log_probability(step.observation, step.action)
        add_scaled_in_place(gradient, grad_log_prob, advantage / len(steps))

    norm = sum(value * value for value in gradient) ** 0.5
    policy.apply_gradient(
        gradient,
        learning_rate=learning_rate,
        max_update_norm=max_update_norm,
    )

    score_diffs = [
        episode.final_scores[episode.learner_team]
        - episode.final_scores[1 - episode.learner_team]
        for episode in episodes
    ]

    return TrainStats(
        episodes=len(episodes),
        learner_decisions=len(steps),
        mean_return=mean(episode.episode_return for episode in episodes),
        mean_score_diff=mean(score_diffs),
        gradient_norm=norm,
    )

