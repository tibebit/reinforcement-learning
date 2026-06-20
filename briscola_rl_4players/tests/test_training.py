import random
import unittest

from briscola.baselines import RandomPolicy
from briscola.features import BriscolaFeatureExtractor
from briscola.policies import LinearSoftmaxPolicy
from briscola.reinforce import RewardConfig, collect_episode, terminal_reward
from briscola.self_play import SelfPlayConfig, SelfPlayTrainer, SnapshotPool


class TrainingTests(unittest.TestCase):
    def test_one_self_play_update_runs(self):
        extractor = BriscolaFeatureExtractor()
        learner = LinearSoftmaxPolicy.initialize(extractor, rng=random.Random(10))
        pool = SnapshotPool(extractor, max_size=5)
        config = SelfPlayConfig(batch_size=5, learning_rate=0.001, snapshot_interval=10)
        trainer = SelfPlayTrainer(learner=learner, pool=pool, config=config, seed=11)
        stats = trainer.train_update()
        self.assertEqual(stats.episodes, 5)
        self.assertEqual(stats.learner_decisions, 50)
        self.assertGreaterEqual(len(pool.snapshots), 1)

    def test_snapshot_pool_max_size_one_keeps_latest(self):
        extractor = BriscolaFeatureExtractor()
        rng = random.Random(20)
        policy = LinearSoftmaxPolicy.initialize(extractor, rng=rng)
        pool = SnapshotPool(extractor, max_size=1)

        pool.add_policy(policy, name="initial", update_index=0)
        pool.add_policy(policy, name="snapshot_1", update_index=1)
        pool.add_policy(policy, name="snapshot_2", update_index=2)

        self.assertEqual([snapshot.name for snapshot in pool.snapshots], ["snapshot_2"])

    def test_snapshot_pool_keeps_best_recent_and_time_spaced(self):
        extractor = BriscolaFeatureExtractor()
        rng = random.Random(30)
        policy = LinearSoftmaxPolicy.initialize(extractor, rng=rng)
        pool = SnapshotPool(
            extractor,
            max_size=5,
            best_count=1,
            recent_count=2,
        )

        pool.add_policy(policy, name="initial", update_index=0)
        for update in range(1, 10):
            pool.add_policy(
                policy,
                name=f"snapshot_{update}",
                update_index=update,
                evaluation_score=100.0 if update == 4 else None,
            )

        names = {snapshot.name for snapshot in pool.snapshots}
        self.assertEqual(len(pool.snapshots), 5)
        self.assertIn("initial", names)
        self.assertIn("snapshot_4", names)
        self.assertIn("snapshot_8", names)
        self.assertIn("snapshot_9", names)
        self.assertTrue(
            any(
                name not in {"initial", "snapshot_4", "snapshot_8", "snapshot_9"}
                for name in names
            )
        )


if __name__ == "__main__":
    unittest.main()
