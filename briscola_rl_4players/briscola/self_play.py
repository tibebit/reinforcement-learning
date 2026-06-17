"""Self-play training with a historical snapshot pool."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .features import BriscolaFeatureExtractor
from .policies import LinearSoftmaxPolicy
from .reinforce import RewardConfig, TrainStats, collect_episode, reinforce_update


@dataclass
class Snapshot:
    """Frozen policy parameters stored in the pool."""

    name: str
    theta: list[float]
    update_index: int


@dataclass
class SnapshotPool:
    """Pool of frozen historical policies.

    The pool stores parameter vectors, not live policy objects. Sampling creates
    fresh policy instances, which prevents accidental updates to snapshots.
    """

    feature_extractor: BriscolaFeatureExtractor
    max_size: int = 20
    snapshots: list[Snapshot] = field(default_factory=list)

    def add_policy(
        self,
        policy: LinearSoftmaxPolicy,
        name: str,
        update_index: int,
    ) -> None:
        self.snapshots.append(
            Snapshot(
                name=name,
                theta=list(policy.theta),
                update_index=update_index,
            )
        )
        self._trim()

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

        # Preserve the initial snapshot and keep the most recent snapshots. A
        # richer implementation can also preserve a best-evaluation snapshot.
        initial = self.snapshots[0]
        recent = self.snapshots[-(self.max_size - 1) :]
        self.snapshots = [initial] + recent


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
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        if not self.pool.snapshots:
            self.pool.add_policy(self.learner, name="initial", update_index=0)

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
            )
        return stats

    def train(self, updates: int) -> list[TrainStats]:
        return [self.train_update() for _ in range(updates)]

