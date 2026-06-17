import random
import unittest

from briscola.baselines import RandomPolicy
from briscola.env import FourPlayerBriscolaEnv
from briscola.rules import partner_of


class ObservationTests(unittest.TestCase):
    def test_observation_hides_partner_hand_before_final_phase(self):
        env = FourPlayerBriscolaEnv(seed=3)
        obs = env.observe(0)
        self.assertEqual(obs.hand, tuple(env.hands[0]))
        self.assertFalse(obs.partner_hand_visible)
        self.assertEqual(obs.partner_hand, ())

        # Hidden-state fields intentionally do not exist on PlayerObservation.
        self.assertFalse(hasattr(obs, "deck"))
        self.assertFalse(hasattr(obs, "hands"))
        self.assertFalse(hasattr(obs, "opponent_hands"))

    def test_observation_reveals_partner_hand_when_deck_is_empty(self):
        env = FourPlayerBriscolaEnv(seed=12)
        policy = RandomPolicy()
        rng = random.Random(12)

        while env.deck:
            obs = env.observe(env.current_player)
            action = policy.select_action(obs, rng)
            env.step(action)

        obs = env.observe(0)
        self.assertTrue(obs.partner_hand_visible)
        self.assertEqual(obs.partner_hand, tuple(env.hands[partner_of(0)]))
        self.assertLessEqual(len(obs.partner_hand), 3)

    def test_observation_reports_public_information(self):
        env = FourPlayerBriscolaEnv(seed=4)
        obs = env.observe(env.current_player)
        self.assertEqual(obs.trump_suit, env.trump_suit)
        self.assertEqual(obs.revealed_trump, env.revealed_trump)
        self.assertEqual(obs.deck_count, len(env.deck))
        self.assertEqual(obs.current_trick, tuple(env.current_trick))


if __name__ == "__main__":
    unittest.main()
