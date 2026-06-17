import unittest

from briscola.cards import make_deck, total_deck_points


class CardTests(unittest.TestCase):
    def test_deck_has_40_unique_cards(self):
        deck = make_deck()
        self.assertEqual(len(deck), 40)
        self.assertEqual(len(set(deck)), 40)

    def test_total_points_are_120(self):
        self.assertEqual(total_deck_points(), 120)


if __name__ == "__main__":
    unittest.main()

