import unittest

from briscola.env import FourPlayerBriscolaEnv
from briscola.features import BriscolaFeatureExtractor


class FeatureTests(unittest.TestCase):
    def test_feature_vector_has_expected_size(self):
        env = FourPlayerBriscolaEnv(seed=5)
        obs = env.observe(env.current_player)
        extractor = BriscolaFeatureExtractor()
        features = extractor.extract(obs, obs.hand[0])
        self.assertEqual(len(features), extractor.size())

    def test_features_are_numeric(self):
        env = FourPlayerBriscolaEnv(seed=6)
        obs = env.observe(env.current_player)
        extractor = BriscolaFeatureExtractor()
        features = extractor.extract(obs, obs.hand[0])
        self.assertTrue(all(isinstance(value, float) for value in features))


if __name__ == "__main__":
    unittest.main()

