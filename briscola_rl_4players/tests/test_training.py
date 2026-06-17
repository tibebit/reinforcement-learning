import random
import unittest

from briscola.features import BriscolaFeatureExtractor
from briscola.policies import LinearSoftmaxPolicy
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


if __name__ == "__main__":
    unittest.main()

