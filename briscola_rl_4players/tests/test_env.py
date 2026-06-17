import random
import unittest

from briscola.baselines import RandomPolicy
from briscola.env import FourPlayerBriscolaEnv


class EnvironmentTests(unittest.TestCase):
    def play_random_game(self, seed):
        env = FourPlayerBriscolaEnv(seed=seed)
        policy = RandomPolicy()
        rng = random.Random(seed)
        while not env.done:
            obs = env.observe(env.current_player)
            action = policy.select_action(obs, rng)
            env.step(action)
            env.assert_invariants()
        return env

    def test_complete_random_game_satisfies_invariants(self):
        env = self.play_random_game(seed=7)
        self.assertEqual(env.trick_index, 10)
        self.assertEqual(sum(env.scores), 120)
        self.assertEqual(env.cards_played_by_player(), [10, 10, 10, 10])
        self.assertEqual(len(env.played_cards), 40)

    def test_seed_reproducibility(self):
        first = self.play_random_game(seed=42)
        second = self.play_random_game(seed=42)
        first_actions = [(play.player_id, play.card.id) for play in first.played_cards]
        second_actions = [(play.player_id, play.card.id) for play in second.played_cards]
        self.assertEqual(first_actions, second_actions)
        self.assertEqual(first.scores, second.scores)


if __name__ == "__main__":
    unittest.main()

