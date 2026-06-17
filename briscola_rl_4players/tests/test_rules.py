import unittest

from briscola.cards import Card, PlayedCard
from briscola.rules import trick_points, trick_winner


class RuleTests(unittest.TestCase):
    def test_trump_beats_non_trump(self):
        trick = (
            PlayedCard(0, Card("cups", "ace")),
            PlayedCard(1, Card("coins", "two")),
        )
        self.assertEqual(trick_winner(trick, trump_suit="coins").player_id, 1)

    def test_higher_trump_wins(self):
        trick = (
            PlayedCard(0, Card("coins", "two")),
            PlayedCard(1, Card("coins", "three")),
        )
        self.assertEqual(trick_winner(trick, trump_suit="coins").player_id, 1)

    def test_leading_suit_wins_without_trump(self):
        trick = (
            PlayedCard(0, Card("cups", "king")),
            PlayedCard(1, Card("swords", "ace")),
            PlayedCard(2, Card("cups", "three")),
        )
        self.assertEqual(trick_winner(trick, trump_suit="coins").player_id, 2)

    def test_trick_points_sum_card_values(self):
        trick = (
            PlayedCard(0, Card("cups", "ace")),
            PlayedCard(1, Card("coins", "three")),
            PlayedCard(2, Card("clubs", "jack")),
            PlayedCard(3, Card("swords", "two")),
        )
        self.assertEqual(trick_points(trick), 23)


if __name__ == "__main__":
    unittest.main()

