import random
import unittest

from briscola.env import FourPlayerBriscolaEnv
from briscola.features import BriscolaFeatureExtractor
from briscola.policies import LinearSoftmaxPolicy


class PolicyTests(unittest.TestCase):
    def test_probabilities_sum_to_one_over_legal_cards(self):
        env = FourPlayerBriscolaEnv(seed=8)
        obs = env.observe(env.current_player)
        policy = LinearSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(1),
        )
        probabilities = policy.action_probabilities(obs)
        self.assertEqual(set(probabilities), set(obs.hand))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_selected_action_is_legal(self):
        env = FourPlayerBriscolaEnv(seed=9)
        obs = env.observe(env.current_player)
        policy = LinearSoftmaxPolicy.initialize(
            BriscolaFeatureExtractor(),
            rng=random.Random(2),
        )
        action = policy.select_action(obs, random.Random(3))
        self.assertIn(action, obs.hand)


if __name__ == "__main__":
    unittest.main()

