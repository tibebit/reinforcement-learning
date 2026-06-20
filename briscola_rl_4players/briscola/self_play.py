"""Self-play training with a historical snapshot pool."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .features import BriscolaFeatureExtractor
from .policies import LinearSoftmaxPolicy
from .reinforce import RewardConfig, TrainStats, collect_episode, reinforce_update


@dataclass
class Snapshot:
    """Frozen policy parameters stored in the pool."""

    name: str
    theta: list[float]
    update_index: int
    evaluation_score: float | None = None


@dataclass
class SnapshotPool:
    """Pool of frozen historical policies.

    The pool stores parameter vectors, not live policy objects. Sampling creates
    fresh policy instances, which prevents accidental updates to snapshots.
    """

    feature_extractor: BriscolaFeatureExtractor
    max_size: int = 20
    best_count: int | None = None
    recent_count: int | None = None
    keep_initial: bool = True
    snapshots: list[Snapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_size < 1:
            raise ValueError("SnapshotPool max_size must be at least 1")
        if self.best_count is not None and self.best_count < 0:
            raise ValueError("best_count cannot be negative")
        if self.recent_count is not None and self.recent_count < 0:
            raise ValueError("recent_count cannot be negative")

    def add_policy(
        self,
        policy: LinearSoftmaxPolicy,
        name: str,
        update_index: int,
        evaluation_score: float | None = None,
    ) -> None:
        self.snapshots.append(
            Snapshot(
                name=name,
                theta=list(policy.theta),
                update_index=update_index,
                evaluation_score=evaluation_score,
            )
        )
        self._trim()

    def update_evaluation_score(self, name: str, evaluation_score: float) -> None:
        """Attach an external evaluation score to a stored snapshot."""

        for snapshot in self.snapshots:
            if snapshot.name == name:
                snapshot.evaluation_score = evaluation_score
                self._trim()
                return
        raise ValueError(f"Unknown snapshot name: {name}")

    def sample_policy(self, rng: random.Random) -> LinearSoftmaxPolicy:
        if not self.snapshots:
            raise ValueError("Cannot sample from an empty snapshot pool")
        snapshot = rng.choice(self.snapshots)
        return LinearSoftmaxPolicy(
            theta=list(snapshot.theta),
            feature_extractor=self.feature_extractor,
            name=snapshot.name,
        )

    def _trim(self) -> None:
        if len(self.snapshots) <= self.max_size:
            return

        if self.max_size == 1:
            # This supports latest-only experiments explicitly.
            self.snapshots = [self.snapshots[-1]]
            return

        selected: list[Snapshot] = []

        if self.keep_initial:
            self._append_unique(selected, [self.snapshots[0]], limit=1)

        remaining = self.max_size - len(selected)
        best_limit = min(self._target_best_count(), remaining)
        best = sorted(
            (snapshot for snapshot in self.snapshots if snapshot.evaluation_score is not None),
            key=lambda snapshot: (snapshot.evaluation_score, snapshot.update_index),
            reverse=True,
        )
        self._append_unique(selected, best, limit=best_limit)

        remaining = self.max_size - len(selected)
        recent_limit = min(self._target_recent_count(), remaining)
        recent = sorted(self.snapshots, key=self._snapshot_key, reverse=True)
        self._append_unique(selected, recent, limit=recent_limit)

        remaining = self.max_size - len(selected)
        if remaining > 0:
            selected_keys = {self._snapshot_key(snapshot) for snapshot in selected}
            older_candidates = [
                snapshot
                for snapshot in sorted(self.snapshots, key=self._snapshot_key)
                if self._snapshot_key(snapshot) not in selected_keys
            ]
            self._append_unique(
                selected,
                self._time_spaced_subset(older_candidates, remaining),
                limit=remaining,
            )

        self.snapshots = sorted(selected, key=self._snapshot_key)

    def _target_best_count(self) -> int:
        if self.best_count is not None:
            return self.best_count
        return max(1, self.max_size // 5)

    def _target_recent_count(self) -> int:
        if self.recent_count is not None:
            return self.recent_count
        return max(1, self.max_size // 2)

    @staticmethod
    def _snapshot_key(snapshot: Snapshot) -> tuple[int, str]:
        return (snapshot.update_index, snapshot.name)

    def _append_unique(
        self,
        selected: list[Snapshot],
        candidates: list[Snapshot],
        limit: int,
    ) -> None:
        selected_keys = {self._snapshot_key(snapshot) for snapshot in selected}
        for snapshot in candidates:
            if len(selected) >= self.max_size or limit <= 0:
                return
            key = self._snapshot_key(snapshot)
            if key in selected_keys:
                continue
            selected.append(snapshot)
            selected_keys.add(key)
            limit -= 1

    @staticmethod
    def _time_spaced_subset(
        snapshots: list[Snapshot],
        limit: int,
    ) -> list[Snapshot]:
        """Pick older snapshots spread over training time."""

        if limit <= 0 or not snapshots:
            return []
        if len(snapshots) <= limit:
            return snapshots
        if limit == 1:
            return [snapshots[len(snapshots) // 2]]

        last_index = len(snapshots) - 1
        indexes = {
            round(position * last_index / (limit - 1))
            for position in range(limit)
        }
        return [snapshots[index] for index in sorted(indexes)]


@dataclass
class SelfPlayConfig:
    batch_size: int = 500
    learning_rate: float = 0.01
    snapshot_interval: int = 50
    max_update_norm: float | None = 5.0
    baseline: str = "batch_mean"
    reward_config: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class SelfPlayTrainer:
    """Population-based self-play trainer.

    Each episode has one active learner using current theta. The partner and
    opponents are sampled as frozen snapshots from the historical pool.
    """

    learner: LinearSoftmaxPolicy
    pool: SnapshotPool
    config: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    seed: int = 0
    update_index: int = 0
    snapshot_score_fn: Callable[[LinearSoftmaxPolicy, int], float | None] | None = None
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        if not self.pool.snapshots:
            self.pool.add_policy(
                self.learner,
                name="initial",
                update_index=0,
                evaluation_score=self._snapshot_score(0),
            )

    def train_update(self) -> TrainStats:
        episodes = []
        for _ in range(self.config.batch_size):
            episode_seed = self.rng.randrange(1_000_000_000)
            learner_seat = self.rng.randrange(4)
            partner_policy = self.pool.sample_policy(self.rng)
            opponent_policy = self.pool.sample_policy(self.rng)
            episodes.append(
                collect_episode(
                    learner_policy=self.learner,
                    partner_policy=partner_policy,
                    opponent_policy=opponent_policy,
                    seed=episode_seed,
                    learner_seat=learner_seat,
                    reward_config=self.config.reward_config,
                )
            )

        stats = reinforce_update(
            policy=self.learner,
            episodes=episodes,
            learning_rate=self.config.learning_rate,
            baseline=self.config.baseline,
            max_update_norm=self.config.max_update_norm,
        )

        self.update_index += 1
        if self.update_index % self.config.snapshot_interval == 0:
            self.pool.add_policy(
                self.learner,
                name=f"snapshot_{self.update_index}",
                update_index=self.update_index,
                evaluation_score=self._snapshot_score(self.update_index),
            )
        return stats

    def train(self, updates: int) -> list[TrainStats]:
        return [self.train_update() for _ in range(updates)]

    def _snapshot_score(self, update_index: int) -> float | None:
        if self.snapshot_score_fn is None:
            return None
        return self.snapshot_score_fn(self.learner, update_index)
